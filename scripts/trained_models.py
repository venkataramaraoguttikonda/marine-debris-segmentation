import os
from huggingface_hub import hf_hub_download

# Ensure the trained_models directory exists
os.makedirs("trained_models", exist_ok=True)

# UNet_CBAM
hf_hub_download(
    repo_id="venkataramaraoguttikonda/marine-debris-models",
    filename="newclass.pth",
    local_dir="trained_models/",
    local_dir_use_symlinks=False,
)

# UNetPlusPlus_CBAM
hf_hub_download(
    repo_id="venkataramaraoguttikonda/marine-debris-models",
    filename="newclass++.pth",
    local_dir="trained_models/",
    local_dir_use_symlinks=False,
)

# SegFormer
hf_hub_download(
    repo_id="venkataramaraoguttikonda/marine-debris-models",
    filename="segformer_marida.pth",
    local_dir="trained_models/",
    local_dir_use_symlinks=False,
)
