import argparse
import torch
import os
import warnings
warnings.filterwarnings("ignore")

from dataset.dataloader_segformer import MaridaDataset
from models.segformer_model import get_segformer_model
from inference.inference_segformer import MaridaTestDataset, run_segformer_inference
from evaluate.evaluate_segformer import evaluate_predictions
from visualization.visualize_segformer import visualize_paper_samples
from utils import load_split_file, get_image_paths_from_ids

# ------------------------------ Main Pipeline ------------------------------
def main(args):
    """
    Runs the SegFormer pipeline: prediction, evaluation, and visualization.
    """
    print("Step 1: Loading test split...")
    test_ids = load_split_file("data/splits/test_X.txt")

    patch_root = 'data/patches'
    test_image_paths = get_image_paths_from_ids(test_ids, patch_root)
    test_label_paths = [p.replace('.tif', '_cl.tif') for p in test_image_paths]

    pred_dir = "predictions_segformer"
    checkpoint_path = args.checkpoint or "trained_models/segformer_marida.pth"
    os.makedirs("trained_models", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- Inference --------
    if args.predict:
        print("Step 2: Running inference on test set...")
        test_dataset = MaridaTestDataset(test_image_paths)
        run_segformer_inference(test_dataset, checkpoint_path=checkpoint_path, save_dir=pred_dir)
        print("Inference complete.")

    # -------- Evaluation --------
    if args.evaluate:
        print("Step 3: Evaluating predictions...")
        evaluate_predictions(pred_dir=pred_dir, gt_dir='data/patches', num_classes=12)
        print("Evaluation complete.")

    # -------- Visualization --------
    if args.visualize:
        print("Step 4: Visualizing results...")
        os.environ["MODEL_TAG"] = "segformer"
        visualize_paper_samples(pred_dir=pred_dir)
        print("Visualization complete.")


# ------------------------------ Argument Parser ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer pipeline for marine debris segmentation")
    parser.add_argument("--predict", action="store_true", help="Run inference on test set")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample predictions")
    parser.add_argument("--checkpoint", type=str, help="Path to trained checkpoint (default: trained_models/segformer_marida.pth)")

    args = parser.parse_args()
    main(args)
