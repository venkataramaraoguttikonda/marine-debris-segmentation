import os
import numpy as np
import torch
import rasterio
import torch.nn.functional as F

# ------------------------------ Confusion Matrix ------------------------------
def compute_confusion_matrix(pred, label, num_classes, ignore_class=0):
    """
    Computes confusion matrix for multi-class segmentation.
    """
    mask = (label >= 1) & (label < num_classes)
    hist = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

# ------------------------------ Accuracy & IoU ------------------------------
def compute_metrics(hist):
    """
    Computes overall pixel accuracy and IoU scores.
    """
    acc = np.diag(hist).sum() / hist.sum()
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    mean_iou = np.nanmean(iou[1:])  # skip background class
    return acc, mean_iou, iou

# ------------------------------ Macro F1 Score ------------------------------
def compute_f1_scores(hist_total, num_classes):
    """
    Computes macro F1 score (excluding background).
    """
    f1_scores = []
    for c in range(1, num_classes):
        TP = hist_total[c, c]
        FP = hist_total[:, c].sum() - TP
        FN = hist_total[c, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores)

# ------------------------------ Evaluation Logic ------------------------------
def evaluate_predictions(pred_dir, gt_dir, num_classes=12):
    """
    Evaluates predictions against ground truth masks.
    
    Args:
        pred_dir (str): Path to predicted .tif files.
        gt_dir (str): Path to ground truth .tif files.
        num_classes (int): Total number of label classes.
    """
    hist_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred_file in os.listdir(pred_dir):
        if pred_file.endswith('_pred.tif'):
            pred_path = os.path.join(pred_dir, pred_file)
            base_id = pred_file.replace('_pred.tif', '')
            gt_path = os.path.join(gt_dir, f"{base_id}_cl.tif")

            if not os.path.exists(gt_path):
                print(f"Missing ground truth for {base_id}")
                continue

            # --------load predicted and ground truth labels--------
            with rasterio.open(pred_path) as src:
                pred = src.read(1)

            with rasterio.open(gt_path) as src:
                label = src.read(1)

            # --------remap debris classes to 7--------
            label = np.where(np.isin(label, [12, 13, 14, 15]), 7, label)

            # --------set 255 labels to -1 (ignore)--------
            label = np.where(label == 255, -1, label)

            # --------resize prediction if shapes mismatch--------
            if pred.shape != label.shape:
                pred = F.interpolate(
                    torch.tensor(pred).unsqueeze(0).unsqueeze(0).float(),
                    size=label.shape,
                    mode='nearest'
                ).squeeze().numpy().astype(np.uint8)

            # --------clamp predictions to valid class range--------
            pred = np.clip(pred, 0, num_classes - 1)

            # --------update confusion matrix--------
            hist = compute_confusion_matrix(pred, label, num_classes)
            hist_total += hist

    # --------compute final metrics--------
    acc, mean_iou, per_class_iou = compute_metrics(hist_total)
    macro_f1 = compute_f1_scores(hist_total, num_classes)

    # --------print results--------
    print(f"\nFinal Evaluation Results:")
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for i, iou in enumerate(per_class_iou):
        print(f"Class {i} IoU: {iou:.4f}")
    print(f"Macro F1 Score (excluding background): {macro_f1:.4f}")
