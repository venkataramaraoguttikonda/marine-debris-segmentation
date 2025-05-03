import torch
from torch.utils.data import Dataset
import rasterio
import torchvision.transforms.functional as TF

# ------------------------------ Dataset Definition ------------------------------
class MaridaDataset(Dataset):
    """
    Dataset for MARIDA segmentation with resizing to 224x224.
    """

    def __init__(self, image_paths, label_paths, global_mean=None, global_std=None):
        """
        Args:
            image_paths (list): Paths to input images.
            label_paths (list): Paths to label masks.
            global_mean (Tensor, optional): Channel-wise mean. Computed if None.
            global_std (Tensor, optional): Channel-wise std. Computed if None.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths

        # --------compute global stats if not provided--------
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
        Loads and processes a single image-label pair.
        """
        # --------read and normalize image--------
        with rasterio.open(self.image_paths[idx]) as img_file:
            image = img_file.read().astype('float32')  # shape: (C, H, W)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.nan_to_num(image, nan=0.0)

        # --------normalize using global mean and std--------
        image = (image - self.global_mean[:, None, None]) / (self.global_std[:, None, None] + 1e-6)

        # --------read and remap label--------
        with rasterio.open(self.label_paths[idx]) as label_file:
            label = label_file.read(1).astype('int64')

        label = torch.tensor(label, dtype=torch.long)

        remap_dict = {12: 7, 13: 7, 14: 7, 15: 7}
        for src, tgt in remap_dict.items():
            label = torch.where(label == src, tgt, label)

        # --------set background to ignore index--------
        label = torch.where(label == 0, 255, label)

        # --------resize both image and label to 224x224--------
        image = TF.resize(image, size=[224, 224])
        label = TF.resize(label.unsqueeze(0), size=[224, 224], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        return {
            'image': image,  # shape: [C, H, W]
            'label': label   # shape: [H, W]
        }

    # ------------------------------ Compute Dataset Statistics ------------------------------
    def compute_global_mean_std(self):
        """
        Computes channel-wise global mean and std.
        """
        total_sum = 0.0
        total_squared_sum = 0.0
        total_pixels = 0

        for idx in range(len(self.image_paths)):
            with rasterio.open(self.image_paths[idx]) as img_file:
                image = img_file.read().astype('float32')

            image = torch.tensor(image, dtype=torch.float32)
            image = torch.nan_to_num(image, nan=0.0)

            total_sum += image.sum(dim=(1, 2))
            total_squared_sum += (image ** 2).sum(dim=(1, 2))
            total_pixels += image.shape[1] * image.shape[2]

        global_mean = total_sum / total_pixels
        global_var = (total_squared_sum / total_pixels) - (global_mean ** 2)
        global_std = global_var.sqrt()

        return global_mean, global_std
