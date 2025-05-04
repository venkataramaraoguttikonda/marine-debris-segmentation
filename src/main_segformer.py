import argparse
import torch
import os
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from dataset.dataloader_segformer import MaridaDataset
from models.segformer_model import get_segformer_model
from train.train_segformer import train_segformer
from inference.inference_segformer import MaridaTestDataset, run_segformer_inference
from evaluate.evaluate_segformer import evaluate_predictions
from visualization.visualize_segformer import visualize_paper_samples
from utils import load_split_file, get_image_paths_from_ids

# ------------------------------ Main Pipeline ------------------------------
def main(args):
    """
    Runs the SegFormer segmentation pipeline based on provided flags.
    """
    print("Step 1: Loading dataset split files...")
    train_ids = load_split_file("data/splits/train_X.txt")
    val_ids = load_split_file("data/splits/val_X.txt")
    test_ids = load_split_file("data/splits/test_X.txt")
    print("Dataset splits loaded.")

    patch_root = 'data/patches'
    train_image_paths = get_image_paths_from_ids(train_ids, patch_root)
    val_image_paths = get_image_paths_from_ids(val_ids, patch_root)
    test_image_paths = get_image_paths_from_ids(test_ids, patch_root)

    train_label_paths = [p.replace('.tif', '_cl.tif') for p in train_image_paths]
    val_label_paths = [p.replace('.tif', '_cl.tif') for p in val_image_paths]
    test_label_paths = [p.replace('.tif', '_cl.tif') for p in test_image_paths]

    pred_dir = "predictions_segformer"
    checkpoint_path = args.checkpoint or "trained_models/segformer_marida.pth"
    os.makedirs("trained_models", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- Training --------
    if args.train:
        print("Step 2: Starting training...")
        train_dataset = MaridaDataset(train_image_paths, train_label_paths)
        val_dataset = MaridaDataset(val_image_paths, val_label_paths)
        model = get_segformer_model(device)
        train_segformer(
            model, train_dataset, val_dataset, device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        print("Training complete.")

    # -------- Inference --------
    if args.predict:
        print("Step 3: Running inference on test set...")
        train_dataset = MaridaDataset(train_image_paths, train_label_paths)
        test_dataset = MaridaTestDataset(
            test_image_paths,
            global_mean=train_dataset.global_mean,
            global_std=train_dataset.global_std
        )
        run_segformer_inference(test_dataset, checkpoint_path=checkpoint_path, save_dir=pred_dir)
        print("Inference complete.")

    # -------- Evaluation --------
    if args.evaluate:
        print("Step 4: Evaluating predictions...")
        evaluate_predictions(pred_dir=pred_dir, gt_dir='data/patches', num_classes=12)
        print("Evaluation complete.")

    # -------- Visualization --------
    if args.visualize:
        print("Step 5: Visualizing results...")
        os.environ["MODEL_TAG"] = "segformer"
        visualize_paper_samples(pred_dir=pred_dir)
        print("Visualization complete.")


# ------------------------------ Argument Parser ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer pipeline for marine debris segmentation")
    parser.add_argument("--train", action="store_true", help="Train the SegFormer model")
    parser.add_argument("--predict", action="store_true", help="Run inference on test set")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample predictions")
    parser.add_argument("--checkpoint", type=str, help="Path to trained checkpoint (default: trained_models/segformer_marida.pth)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")

    args = parser.parse_args()
    main(args)