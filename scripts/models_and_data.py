import os
import zipfile
from huggingface_hub import hf_hub_download

# ------------------ Create trained_models directory ------------------
os.makedirs("trained_models", exist_ok=True)

# ------------------ Download Model Checkpoints ------------------
model_files = [
    "newclass.pth",
    "newclass++.pth",
    "segformer_marida.pth"
]

for file in model_files:
    hf_hub_download(
        repo_id="venkataramaraoguttikonda/marine-debris-models",
        filename=file,
        local_dir="trained_models",
        local_dir_use_symlinks=False
    )

# ------------------ Download and Extract data.zip ------------------
zip_path = hf_hub_download(
    repo_id="venkataramaraoguttikonda/marine-debris-models",
    filename="data.zip",
    local_dir=".",
    local_dir_use_symlinks=False
)

# Extract to `data/` folder
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("data")

# Remove the zip file after extraction
os.remove(zip_path)

print("All models downloaded to trained_models/ and data extracted to data/")
