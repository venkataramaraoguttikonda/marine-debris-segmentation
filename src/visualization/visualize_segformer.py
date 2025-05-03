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
    """Convert CSS color name to RGB tuple in 0–255 range."""
    return tuple(int(c * 255) for c in to_rgb(name))

colors = np.array([
    [255, 255, 255],  # Background - white
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
def visualize_image_label_prediction(img_path, label_path, pred_path):
    """
    Display side-by-side visualization of image, ground truth, and prediction.

    Args:
        img_path (str): Path to input image (.tif).
        label_path (str): Path to ground truth label image.
        pred_path (str): Path to prediction mask (.tif).
    """
    with rasterio.open(img_path) as src:
        image = src.read()

    # Convert to RGB
    if image.shape[0] >= 3:
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    else:
        rgb = np.stack([image[0]] * 3, axis=-1)

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    # Resize if prediction size doesn't match label
    if pred.shape != label.shape:
        pred = torch.nn.functional.interpolate(
            torch.tensor(pred).unsqueeze(0).unsqueeze(0).float(),
            size=label.shape,
            mode='nearest'
        ).squeeze().numpy().astype(np.uint8)

    label = np.clip(label, 0, len(colors)-1).astype(np.int64)
    pred = np.clip(pred, 0, len(colors)-1).astype(np.int64)

    label_rgb = colors[label]
    pred_rgb = colors[pred]

    # ------------------------------ Plotting ------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    images = [rgb, label_rgb, pred_rgb]
    titles = ['Input Image', 'Ground Truth', 'Prediction']

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
        rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # ------------------------------ Legend ------------------------------
    skip_labels = {'Shallow Water', 'Waves', 'Cloud Shadows', 'Wakes'}
    patches = [
        mpatches.Patch(color=colors[i]/255.0, label=class_labels[i])
        for i in range(len(class_labels)) if class_labels[i] not in skip_labels
    ]

    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small'
    )

    # ------------------------------ Save Output ------------------------------
    model_tag = os.environ.get("MODEL_TAG", "default")
    vis_dir = os.path.join("vis_outputs", model_tag)
    os.makedirs(vis_dir, exist_ok=True)

    patch_id = os.path.basename(img_path).replace('.tif', '')
    save_path = os.path.join(vis_dir, f"{patch_id}_vis.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# ------------------------------ Visualize Paper Samples ------------------------------
def visualize_paper_samples(pred_dir):
    """
    Visualize fixed MARIDA samples used in the paper.
    
    Args:
        pred_dir (str): Directory containing predicted masks.
    """
    samples_paper = [
        'data/patches/S2_12-12-20_16PCC_6.tif',
        'data/patches/S2_22-12-20_18QYF_0.tif',
        'data/patches/S2_27-1-19_16QED_14.tif',
        'data/patches/S2_14-9-18_16PCC_13.tif'
    ]

    for img_path in samples_paper:
        patch_name = os.path.basename(img_path).replace('.tif', '')
        label_path = f"data/patches/{patch_name}_cl.tif"
        pred_path = os.path.join(pred_dir, f"{patch_name}_pred.tif")

        if os.path.exists(label_path) and os.path.exists(pred_path):
            print(f"Visualizing {patch_name}")
            visualize_image_label_prediction(img_path, label_path, pred_path)
        else:
            print(f"Missing label or prediction for {patch_name}")
