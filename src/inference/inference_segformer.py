import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

# ------------------------------ Test Dataset ------------------------------
class MaridaTestDataset(Dataset):
    """
    Dataset for test-time inference on MARIDA images.
    """

    def __init__(self, image_paths, global_mean, global_std):
        """
        Args:
            image_paths (list): Paths to test images.
            global_mean (Tensor): Channel-wise mean for normalization.
            global_std (Tensor): Channel-wise std for normalization.
        """
        self.image_paths = image_paths
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        """
        Returns number of test samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and normalizes a test image.
        """
        with rasterio.open(self.image_paths[idx]) as img_file:
            image = img_file.read().astype('float32')  # shape: (C, H, W)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.nan_to_num(image, nan=0.0)

        # Normalize using provided mean and std
        image = (image - self.global_mean[:, None, None]) / (self.global_std[:, None, None] + 1e-6)

        # Resize to 512×512
        image = F.interpolate(image.unsqueeze(0), size=[512, 512], mode='bilinear', align_corners=False).squeeze(0)

        return image, self.image_paths[idx]

# ------------------------------ Inference Function ------------------------------
def run_segformer_inference(test_dataset, checkpoint_path, save_dir="predictions_segformer", batch_size=8):
    """
    Runs inference using a fine-tuned SegFormer model and saves predictions.
    
    Args:
        test_dataset (Dataset): Prepared test dataset.
        checkpoint_path (str): Path to trained model checkpoint.
        save_dir (str): Directory to save prediction .tif files.
        batch_size (int): Batch size for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained SegFormer model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=12,
        ignore_mismatched_sizes=True
    )

    # --------update input projection for 11-channel input--------
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
        in_channels=11,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )

    # Initialize new input layer
    nn.init.kaiming_normal_(model.segformer.encoder.patch_embeddings[0].proj.weight, mode='fan_out', nonlinearity='relu')
    if model.segformer.encoder.patch_embeddings[0].proj.bias is not None:
        nn.init.constant_(model.segformer.encoder.patch_embeddings[0].proj.bias, 0)

    # Load trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------inference loop--------
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)

            # Run model
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Resize output to 512×512
            logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype('uint8')

            # --------save predictions--------
            for i in range(preds.shape[0]):
                filename = os.path.basename(paths[i]).replace('.tif', '_pred.tif')
                save_path = os.path.join(save_dir, filename)

                with rasterio.open(paths[i]) as src:
                    meta = src.meta.copy()
                    meta.update({
                        'count': 1,
                        'dtype': 'uint8',
                        'height': 512,
                        'width': 512
                    })

                    with rasterio.open(save_path, 'w', **meta) as dst:
                        dst.write(preds[i], 1)

                print(f"Saved {save_path}")