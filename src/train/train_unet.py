import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import rasterio
import os
import matplotlib.pyplot as plt

# ------------------------------ Class Weights Computation ------------------------------
def compute_class_weights(label_paths, num_classes):
    """
    Computes class weights based on inverse frequency.
    """
    class_counts = torch.zeros(num_classes)
    for label_path in label_paths:
        with rasterio.open(label_path) as lbl_file:
            label = torch.tensor(lbl_file.read(1), dtype=torch.long)
            for c in range(num_classes):
                class_counts[c] += (label == c).sum()

    class_counts += 1e-6  # avoid division by zero
    total_pixels = class_counts.sum()
    class_weights = (total_pixels - class_counts) / total_pixels
    return class_weights

# ------------------------------ Training Loop ------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_classes=12,
    num_epochs=75,
    save_path="trained_models/model.pth"
):
    """
    Trains a segmentation model and evaluates it on validation set.
    """
    model.to(device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    class_weights = compute_class_weights(train_loader.dataset.label_paths, num_classes)
    print(f"Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    best_iou = 0.0
    train_losses = []
    val_losses = []
    miou_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}] Training Loss: {avg_train_loss:.4f}")

        # --------validation--------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        iou_per_class = torch.zeros(num_classes).to(device)
        total_per_class = torch.zeros(num_classes).to(device)

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                mask = labels != 255
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

                for cls in range(num_classes):
                    pred_inds = (preds == cls)
                    label_inds = (labels == cls)
                    valid_mask = labels != 255
                    intersection = ((pred_inds & label_inds) & valid_mask).sum().float()
                    union = ((pred_inds | label_inds) & valid_mask).sum().float()
                    if union > 0:
                        iou = intersection / (union + 1e-6)
                        iou_per_class[cls] += iou
                        total_per_class[cls] += 1

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_acc = correct / total
        mean_iou = (iou_per_class / (total_per_class + 1e-6)).mean().item()
        miou_scores.append(mean_iou)

        print(f"Epoch [{epoch+1}] Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"Saved new best model to {save_path} with mIoU: {best_iou:.4f}")

    # --------- Save loss and mIoU curves ---------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UNet Loss Curve")
    plt.legend()
    plt.savefig("plots/unet_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(miou_scores, label="Validation mIoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("UNet mIoU Curve")
    plt.legend()
    plt.savefig("plots/unet_miou_curve.png")
    plt.close()

    return model
