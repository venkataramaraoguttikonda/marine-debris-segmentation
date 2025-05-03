import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import rasterio

# ------------------------------ Dataset Definition ------------------------------
class MaridaDataset(Dataset):
    """
    Custom dataset for MARIDA marine debris segmentation.
    """

    def __init__(self, image_paths, label_paths, global_mean=None, global_std=None):
        """
        Args:
            image_paths (list): Paths to input images.
            label_paths (list): Paths to label masks.
            global_mean (tensor, optional): Channel-wise mean. Computed if None.
            global_std (tensor, optional): Channel-wise std. Computed if None.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths

        # --------compute global statistics if not provided--------
        if global_mean is None or global_std is None:
            self.global_mean, self.global_std = self.compute_global_mean_std()
        else:
            self.global_mean = global_mean
            self.global_std = global_std

    def __len__(self):
        """
        Returns the number of samples.
        """
        return len(self.image_paths)

    # ------------------------------ Data Loader Logic ------------------------------
    def __getitem__(self, idx):
        """
        Loads and processes an image-label pair.
        """
        # --------read and normalize image--------
        with rasterio.open(self.image_paths[idx]) as img_file:
            image = img_file.read().astype('float32')  # shape: (C, H, W)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.nan_to_num(image, nan=0.0)

        # --------normalize using global mean and std--------
        image = (image - self.global_mean[:, None, None]) / (self.global_std[:, None, None] + 1e-6)

        # --------resize image to 512x512--------
        image = TF.resize(image, size=[512, 512], interpolation=TF.InterpolationMode.BILINEAR)

        # --------read and preprocess label--------
        with rasterio.open(self.label_paths[idx]) as label_file:
            label = label_file.read(1).astype('int64')

        label = torch.tensor(label, dtype=torch.long)

        # --------remap certain label classes to class 7--------
        for src in [12, 13, 14, 15]:
            label = torch.where(label == src, 7, label)

        # --------set background (0) to ignore index (255)--------
        label = torch.where(label == 0, 255, label)

        # --------resize label to 512x512--------
        label = TF.resize(label.unsqueeze(0), size=[512, 512], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        return {
            'pixel_values': image,  # shape: [C, H, W]
            'labels': label         # shape: [H, W]
        }

    # ------------------------------ Compute Dataset Statistics ------------------------------
    def compute_global_mean_std(self):
        """
        Computes global mean and std across all images.
        """
        total_sum = 0.0
        total_squared_sum = 0.0
        total_pixels = 0

        # --------accumulate per-channel sum and squared sum--------
        for path in self.image_paths:
            with rasterio.open(path) as img_file:
                image = img_file.read().astype('float32')
            image = torch.tensor(image)
            image = torch.nan_to_num(image, nan=0.0)

            total_sum += image.sum(dim=(1, 2))
            total_squared_sum += (image ** 2).sum(dim=(1, 2))
            total_pixels += image.shape[1] * image.shape[2]

        # --------compute mean and std--------
        global_mean = total_sum / total_pixels
        global_var = (total_squared_sum / total_pixels) - (global_mean ** 2)
        global_std = global_var.sqrt()

        return global_mean, global_std
