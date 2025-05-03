import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

# ------------------------------ Class Labels ------------------------------
class_labels = [
    'Background', 'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
    'Natural Organic Material', 'Ship', 'Clouds', 'Marine Water',
    'Sediment-Laden Water', 'Foam', 'Turbid Water', 'Shallow Water',
    'Waves', 'Cloud Shadows', 'Wakes', 'Mixed Water'
]

# ------------------------------ Color Mapping ------------------------------
def css_to_rgb255(name):
    """
    Converts CSS color name to RGB tuple in 0-255 scale.
    """
    return tuple(int(c * 255) for c in to_rgb(name))

colors = np.array([
    [0, 0, 0],  # Background
    css_to_rgb255('red'),
    css_to_rgb255('green'),
    css_to_rgb255('limegreen'),
    css_to_rgb255('brown'),
    css_to_rgb255('orange'),
    css_to_rgb255('silver'),
    css_to_rgb255('navy'),
    css_to_rgb255('gold'),
    css_to_rgb255('purple'),
    css_to_rgb255('darkkhaki'),
    css_to_rgb255('darkturquoise'),
    css_to_rgb255('seashell'),
    css_to_rgb255('gray'),
    css_to_rgb255('yellow'),
    css_to_rgb255('rosybrown')
], dtype=np.uint8)

# ------------------------------ Visualization Function ------------------------------
def visualize_image_label_prediction(img_path, label_path, pred_path, save_dir):
    """
    Plots and saves side-by-side comparison of input image, ground truth, and prediction.
    """
    with rasterio.open(img_path) as src:
        image = src.read()

    # Convert to RGB (using bands 3, 2, 1 if available)
    if image.shape[0] >= 3:
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    else:
        rgb = np.stack([image[0]] * 3, axis=-1)

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    # Resize prediction if shape mismatch
    if pred.shape != label.shape:
        pred = torch.nn.functional.interpolate(
            torch.tensor(pred).unsqueeze(0).unsqueeze(0).float(),
            size=label.shape,
            mode='nearest'
        ).squeeze().numpy().astype(np.uint8)

    # Clip values and map to RGB
    label = np.clip(label, 0, len(colors) - 1).astype(np.int64)
    pred = np.clip(pred, 0, len(colors) - 1).astype(np.int64)

    label_rgb = colors[label]
    pred_rgb = colors[pred]
    label_rgb[label == 0] = [0, 0, 0]
    pred_rgb[pred == 0] = [0, 0, 0]

    # Plot side-by-side visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(rgb); axs[0].set_title('Input Image'); axs[0].axis('off')
    axs[1].imshow(label_rgb); axs[1].set_title('Ground Truth'); axs[1].axis('off')
    axs[2].imshow(pred_rgb); axs[2].set_title('Prediction'); axs[2].axis('off')

    # Add legend
    patches = [mpatches.Patch(color=colors[i]/255.0, label=class_labels[i]) for i in range(len(class_labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    patch_name = os.path.basename(img_path).replace('.tif', '')
    save_path = os.path.join(save_dir, f"{patch_name}_vis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {save_path}")

# ------------------------------ Visualize Selected Samples ------------------------------
def visualize_segformer_samples(pred_dir="predictions_segformer", save_dir="vis_outputs/segformer"):
    """
    Generates visualizations for selected test samples.
    """
    samples = [
        'data/patches/S2_12-12-20_16PCC_6.tif',
        'data/patches/S2_22-12-20_18QYF_0.tif',
        'data/patches/S2_27-1-19_16QED_14.tif',
        'data/patches/S2_14-9-18_16PCC_13.tif'
    ]
    for img_path in samples:
        patch_name = os.path.basename(img_path).replace('.tif', '')
        label_path = f"data/patches/{patch_name}_cl.tif"
        pred_path = os.path.join(pred_dir, f"{patch_name}_pred.tif")

        if os.path.exists(label_path) and os.path.exists(pred_path):
            visualize_image_label_prediction(img_path, label_path, pred_path, save_dir)
        else:
            print(f"Missing label or prediction for {patch_name}")