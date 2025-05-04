# Marine Debris Segmentation using Deep Learning

This repository presents a modular pipeline for segmenting marine debris and related oceanographic features from Sentinel-2 satellite imagery. We benchmark three models — **UNet**, **UNet++**, and **SegFormer** — on the MARIDA dataset.

The pipeline includes:
- Custom PyTorch datasets and loaders
- Class-weighted loss for imbalance handling
- Automated training, inference, evaluation, and visualization
- Loss and mIoU plots for training diagnostics

## 📦 Dataset

This project uses the **Marine Debris Archive (MARIDA)** — a weakly-supervised pixel-level semantic segmentation benchmark derived from Sentinel-2 satellite imagery. It is designed to detect marine debris and associated sea surface phenomena like **Sargassum**, **organic matter**, **ships**, **cloud shadows**, and **waves**.

---

### 📥 Download

You can download the MARIDA dataset from:

- **[Zenodo (DOI: 10.5281/zenodo.7053888)](https://doi.org/10.5281/zenodo.7053888)**
- **[Radiant MLHub](https://mlhub.earth/data/marida)**

After downloading, extract it under the `data/` directory:

```bash
marine-debris-segmentation/
└── data/
    ├── patches/                  # Contains image patches and masks
    │   ├── S2_12-12-20_16PCC_6.tif
    │   ├── S2_12-12-20_16PCC_6_cl.tif      # Ground truth mask
    │   └── S2_12-12-20_16PCC_6_conf.tif    # Confidence mask (optional)
    ├── splits/
    │   ├── train_X.txt
    │   ├── val_X.txt
    │   └── test_X.txt
    └── labels_mapping.txt        # (optional - used for multi-label classification)
```

---

### 🗂️ Label Classes

The classification masks (`*_cl.tif`) include the following class IDs:

| ID  | Class Name               |
|-----|---------------------------|
| 0   | Background                |
| 1   | Marine Debris             |
| 2   | Dense Sargassum           |
| 3   | Sparse Sargassum          |
| 4   | Natural Organic Material  |
| 5   | Ship                      |
| 6   | Clouds                    |
| 7   | Marine Water              |
| 8   | Sediment-Laden Water      |
| 9   | Foam                      |
| 10  | Turbid Water              |
| 11  | Shallow Water             |
| 12  | Waves                     |
| 13  | Cloud Shadows             |
| 14  | Wakes                     |
| 15  | Mixed Water               |

> ⚠️ *Class `0` (Background) and the optional `*_conf.tif` confidence maps are not used for training, but are helpful for filtering or visualization.*

---

### 📄 Citation

> R. Veit et al., “MARIDA: A Benchmark for Marine Debris Detection in Sentinel-2 Satellite Imagery,” *IGARSS 2022*.  
> DOI: [10.1109/IGARSS46834.2022.9884340](https://doi.org/10.1109/IGARSS46834.2022.9884340)

## ⚙️ Installation & Setup

This project uses **Python 3.9+** and is managed via **conda**. All dependencies are defined in the `environment.yml` file.

### 🛠️ Create the Environment

```bash
conda env create -f environment.yml
conda activate marinedebri
```

This installs PyTorch, Hugging Face Transformers, rasterio, matplotlib, and other required packages.

### 📁 Initial Folder Structure

After cloning and setting up, your project should look like this:

```bash
marine-debris-segmentation/
├── src/
│   ├── train/
│   ├── inference/
│   ├── evaluate/
│   ├── models/
│   ├── dataset/
│   └── visualization/
├── scripts/
├── data/
│   ├── patches/
│   └── splits/
├── environment.yml
└── README.md
```

> 🔄 Additional folders like `trained_models/`, `predictions_*/`, `vis_outputs/`, and `plots/` are **automatically generated** during training, inference, and visualization steps.

### 📈 Training & Evaluation

You can train, evaluate, and visualize results for any of the supported models using the shell scripts in the `scripts/` directory:

#### 🔁 Train and Evaluate (from scratch)

```bash
# UNet
bash scripts/run_unet.sh

# UNet++
bash scripts/run_unetpp.sh

# SegFormer
bash scripts/run_segformer.sh
```

---

#### 📦 Use Pretrained Models (on test split only)

If you have only the test split (included in this repo) and want to run inference + evaluation + visualization using pretrained models in `trained_models/`:

```bash
# UNet
python src/main.py --model unet --predict --evaluate --visualize

# UNet++
python src/main.py --model unet++ --predict --evaluate --visualize

# SegFormer
python src/main_segformer.py --predict --evaluate --visualize
```

🧠 Note: The outputs will be saved under:

- `predictions_unet/`, `predictions_unetpp/`, `predictions_segformer/`
- `vis_outputs/unet/`, `vis_outputs/unetpp/`, `vis_outputs/segformer/`
- `plots/` for training loss/mIoU curves if training is run