import os
import torch
import rasterio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------------------------------ Test Dataset ------------------------------
class MaridaTestDataset(Dataset):
    """
    Dataset for test-time prediction on MARIDA images.
    """

    def __init__(self, image_paths, global_mean, global_std):
        """
        Args:
            image_paths (list): List of image paths.
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
        Loads and preprocesses one test image.
        """
        with rasterio.open(self.image_paths[idx]) as img_file:
            image = img_file.read().astype('float32')  # shape: (C, H, W)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.nan_to_num(image, nan=0.0)

        # Normalize and resize to 224x224
        image = (image - self.global_mean[:, None, None]) / (self.global_std[:, None, None] + 1e-6)
        image = F.interpolate(image.unsqueeze(0), size=[224, 224], mode='bilinear', align_corners=False).squeeze(0)

        return image, self.image_paths[idx]

# ------------------------------ Prediction Function ------------------------------
def run_prediction(model, test_dataset, save_dir='predictions_newclass', batch_size=8, device=None):
    """
    Runs inference using the given model and saves the predicted masks.

    Args:
        model (nn.Module): Trained segmentation model.
        test_dataset (Dataset): Preprocessed test dataset.
        save_dir (str): Directory to save output .tif masks.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device for inference.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(save_dir, exist_ok=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype('uint8')

            # --------save predictions--------
            for i in range(preds.shape[0]):
                filename = os.path.basename(paths[i]).replace('.tif', '_pred.tif')
                save_path = os.path.join(save_dir, filename)

                with rasterio.open(paths[i]) as src:
                    meta = src.meta.copy()
                    meta.update({
                        'count': 1,
                        'dtype': 'uint8',
                        'height': 224,
                        'width': 224
                    })

                    with rasterio.open(save_path, 'w', **meta) as dst:
                        dst.write(preds[i], 1)

                print(f"Saved {save_path}")
