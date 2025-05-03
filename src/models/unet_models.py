import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
    # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        scale = self.sigmoid_channel(avg_out + max_out)
        x = x * scale

    # Spatial Attention
        avg_out_spatial = torch.mean(x, dim=1, keepdim=True)    # New avg_out for spatial
        max_out_spatial, _ = torch.max(x, dim=1, keepdim=True)  # New max_out for spatial
        concat = torch.cat([avg_out_spatial, max_out_spatial], dim=1)  # (B, 2, H, W)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(concat))  # (B, 1, H, W)
        x = x * spatial_attention

        return x

class UNet_CBAM(nn.Module):
    def __init__(self, in_channels=11, num_classes=12):
        super(UNet_CBAM, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.cbam1 = CBAM(64)
        self.enc2 = conv_block(64, 128)
        self.cbam2 = CBAM(128)
        self.enc3 = conv_block(128, 256)
        self.cbam3 = CBAM(256)
        self.enc4 = conv_block(256, 512)
        self.cbam4 = CBAM(512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.cbam_dec4 = CBAM(512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.cbam_dec3 = CBAM(256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.cbam_dec2 = CBAM(128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.cbam_dec1 = CBAM(64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.cbam1(self.enc1(x))
        enc2 = self.cbam2(self.enc2(self.pool(enc1)))
        enc3 = self.cbam3(self.enc3(self.pool(enc2)))
        enc4 = self.cbam4(self.enc4(self.pool(enc3)))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.cbam_dec4(self.dec4(dec4))

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.cbam_dec3(self.dec3(dec3))

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.cbam_dec2(self.dec2(dec2))

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.cbam_dec1(self.dec1(dec1))

        out = self.final_conv(dec1)
        return out


# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        return self.cbam(self.block(x))

# Full UNet++ with CBAM
class UNetPlusPlus_CBAM(nn.Module):
    def __init__(self, in_channels=11, num_classes=12, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Decoder grid
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])

        # Final classifiers
        if self.deep_supervision:
            self.final = nn.ModuleList([
                nn.Conv2d(filters[0], num_classes, kernel_size=1) for _ in range(4)
            ])
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))

        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, x1_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))

        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, x2_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, x1_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))

        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, x3_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, x2_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, x1_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))

        if self.deep_supervision:
            return [F.interpolate(f(x0), size=x.shape[2:], mode='bilinear', align_corners=False) for f, x0 in zip(self.final, [x0_1, x0_2, x0_3, x0_4])]
        else:
            return self.final(x0_4)