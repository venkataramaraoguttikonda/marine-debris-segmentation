import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import rasterio
import os
import matplotlib.pyplot as plt

# ------------------------------ Class Weights Computation ------------------------------
def compute_class_weights(label_paths, num_classes, c=1.02):
    """
    Computes inverse log frequency class weights.
    """
    class_counts = torch.zeros(num_classes)

    for path in label_paths:
        with rasterio.open(path) as src:
            label = torch.tensor(src.read(1), dtype=torch.long)

            # Remap debris classes to 7
            for src_class in [12, 13, 14, 15]:
                label = torch.where(label == src_class, 7, label)

            # Exclude background class 0
            label = label[label != 0]

            for i in range(num_classes):
                class_counts[i] += (label == i).sum()

    class_distribution = class_counts / class_counts.sum()
    class_weights = 1.0 / torch.log(c + class_distribution)

    return class_weights

# ------------------------------ Training Loop ------------------------------
def train_segformer(
    model,
    train_dataset,
    val_dataset,
    device,
    num_epochs=30,
    batch_size=8,
    lr=5e-5,
    save_path="trained_models/segformer_marida.pth"
):
    """
    Trains SegFormer model on MARIDA dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    weights = compute_class_weights(train_dataset.label_paths, num_classes=12)
    print("Class Weights:", weights)

    criterion = CrossEntropyLoss(weight=weights.to(device), ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_miou = 0.0
    train_losses = []
    val_losses = []
    miou_scores = []

    for epoch in range(num_epochs):
        # --------training phase--------
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # --------validation phase--------
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        iou_per_class = torch.zeros(model.config.num_labels).to(device)
        total_per_class = torch.zeros(model.config.num_labels).to(device)

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                logits = model(pixel_values=pixel_values).logits
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                preds = logits.argmax(dim=1)

                loss = criterion(logits, labels)
                val_loss += loss.item()

                mask = labels != 255
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

                for cls in range(model.config.num_labels):
                    pred_inds = (preds == cls)
                    label_inds = (labels == cls)
                    intersection = ((pred_inds & label_inds) & mask).sum().float()
                    union = ((pred_inds | label_inds) & mask).sum().float()

                    if union > 0:
                        iou = intersection / (union + 1e-6)
                        iou_per_class[cls] += iou
                        total_per_class[cls] += 1

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        acc = correct / total
        mean_iou = (iou_per_class / (total_per_class + 1e-6)).mean().item()
        miou_scores.append(mean_iou)
        print(f"Epoch [{epoch+1}] Val Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, mIoU: {mean_iou:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} with mIoU: {best_miou:.4f}")

    # -------- Save loss and mIoU curves --------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SegFormer Training Loss")
    plt.legend()
    plt.savefig("plots/segformer_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SegFormer Validation Loss")
    plt.legend()
    plt.savefig("plots/segformer_val_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(miou_scores, label="Validation mIoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("SegFormer Validation mIoU")
    plt.legend()
    plt.savefig("plots/segformer_miou_curve.png")
    plt.close()

    return model
