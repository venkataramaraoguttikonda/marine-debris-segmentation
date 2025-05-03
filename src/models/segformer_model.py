import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

# ------------------------------ Model Loader ------------------------------
def get_segformer_model(device):
    """
    Loads and adapts SegFormer model for 11-channel input and 12 output classes.
    """
    # Load pretrained SegFormer model for semantic segmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=12,
        ignore_mismatched_sizes=True
    )

    # --------modify input projection layer from 3 to 11 channels--------
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
        in_channels=11,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )

    # --------initialize weights for the new input layer--------
    nn.init.kaiming_normal_(model.segformer.encoder.patch_embeddings[0].proj.weight, mode='fan_out', nonlinearity='relu')
    if model.segformer.encoder.patch_embeddings[0].proj.bias is not None:
        nn.init.constant_(model.segformer.encoder.patch_embeddings[0].proj.bias, 0)

    model.to(device)
    return model