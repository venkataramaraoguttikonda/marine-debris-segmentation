import argparse
import os
import torch
import warnings
warnings.filterwarnings("ignore")

from dataset.dataloader_unet import MaridaDataset
from models.unet_models import UNet_CBAM, UNetPlusPlus_CBAM
from inference.inference_unet import MaridaTestDataset, run_prediction
from evaluate.evaluate_unet import evaluate_predictions
from utils import load_split_file, get_image_paths_from_ids

# ------------------------------ Model Loader ------------------------------
def load_model(model_name, in_channels, num_classes, device, checkpoint_path):
    """
    Loads a UNet or UNet++ model with pretrained weights.
    """
    if model_name == "unet":
        model = UNet_CBAM(in_channels, num_classes)
    elif model_name == "unet++":
        model = UNetPlusPlus_CBAM(in_channels, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model

# ------------------------------ Default Checkpoint Path ------------------------------
def get_default_checkpoint(model_name):
    """
    Returns the default checkpoint path based on model name.
    """
    return {
        "unet": "trained_models/newclass.pth",
        "unet++": "trained_models/newclass++.pth"
    }[model_name]

# ------------------------------ Main Pipeline ------------------------------
def main(args):
    """
    Executes the pipeline: prediction, evaluation, visualization.
    """
    print("Loading test split...")
    test_ids = load_split_file("data/splits/test_X.txt")
    patch_root = 'data/patches'

    test_image_paths = get_image_paths_from_ids(test_ids, patch_root)
    test_label_paths = [p.replace('.tif', '_cl.tif') for p in test_image_paths]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = args.checkpoint or get_default_checkpoint(args.model)
    pred_dir = f"predictions_{args.model.replace('+', 'p')}"

    # -------- Prediction --------
    if args.predict:
        print("Starting prediction...")
        model = load_model(args.model, in_channels=11, num_classes=12, device=device, checkpoint_path=checkpoint_path)

        test_dataset = MaridaTestDataset(test_image_paths)
        run_prediction(model, test_dataset, save_dir=pred_dir, batch_size=8, device=device)
        print("Prediction complete. Outputs saved to:", pred_dir)

    # -------- Evaluation --------
    if args.evaluate:
        print("Starting evaluation...")
        evaluate_predictions(pred_dir=pred_dir, gt_dir='data/patches', num_classes=12)
        print("Evaluation complete.")

    # -------- Visualization --------
    if args.visualize:
        print("Generating visualizations...")
        os.environ["MODEL_TAG"] = args.model.replace("+", "pp")
        from visualization.visualize_unet import visualize_paper_samples
        visualize_paper_samples(pred_dir)
        print("Visualization complete.")

# ------------------------------ Argument Parser ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction, evaluation, and visualization")
    parser.add_argument("--predict", action="store_true", help="Run prediction on test set")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions vs labels")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet++"], help="Model to use: 'unet' or 'unet++'")
    parser.add_argument("--checkpoint", type=str, help="Optional: path to model checkpoint (defaults to model name)")

    args = parser.parse_args()
    main(args)
