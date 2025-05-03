import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio

# ------------------------------ Confusion Matrix ------------------------------
def compute_confusion_matrix(pred, label, num_classes, ignore_class=0):
    """
    Computes the confusion matrix for segmentation.
    """
    # Mask out invalid labels (exclude class 0)
    mask = (label >= 1) & (label < num_classes)

    # Row: ground truth, Column: prediction
    hist = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist

# ------------------------------ Metrics Computation ------------------------------
def compute_metrics(hist):
    """
    Computes pixel accuracy and IoU from confusion matrix.
    """
    acc = np.diag(hist).sum() / hist.sum()
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    mean_iou = np.nanmean(iou[1:])  # exclude background class 0
    return acc, mean_iou, iou

# ------------------------------ Evaluation Logic ------------------------------
def evaluate_predictions(pred_dir, gt_dir, num_classes=12):
    """
    Evaluates segmentation predictions against ground truth masks.
    
    Args:
        pred_dir (str): Directory with predicted .tif masks.
        gt_dir (str): Directory with ground truth .tif masks.
        num_classes (int): Total number of classes including background.
    """
    hist_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('_pred.tif'):
            continue

        # --------resolve file paths--------
        pred_path = os.path.join(pred_dir, pred_file)
        base_id = pred_file.replace('_pred.tif', '')
        gt_path = os.path.join(gt_dir, f"{base_id}_cl.tif")

        if not os.path.exists(gt_path):
            print(f"Missing ground truth for {base_id}")
            continue

        # --------read prediction and ground truth--------
        with rasterio.open(pred_path) as src:
            pred = src.read(1)

        with rasterio.open(gt_path) as src:
            label = src.read(1)

        # --------remap marine debris classes to class 7--------
        label = np.where(np.isin(label, [12, 13, 14, 15]), 7, label)

        # --------ignore label 255--------
        label = np.where(label == 255, -1, label)

        # --------resize prediction if shapes mismatch--------
        if pred.shape != label.shape:
            pred = F.interpolate(
                torch.tensor(pred).unsqueeze(0).unsqueeze(0).float(),
                size=label.shape,
                mode='nearest'
            ).squeeze().numpy().astype(np.uint8)

        pred = np.clip(pred, 0, num_classes - 1)

        # --------compute confusion matrix--------
        hist = compute_confusion_matrix(pred, label, num_classes)
        hist_total += hist

    # --------compute final metrics--------
    acc, mean_iou, per_class_iou = compute_metrics(hist_total)

    # --------print results--------
    print("\nFinal Evaluation Results:")
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for i, iou in enumerate(per_class_iou):
        print(f"Class {i} IoU: {iou:.4f}")