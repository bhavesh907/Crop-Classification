# Crop Classification with Multi-Temporal Satellite Imagery

Multi-class crop classification from 10 multi-temporal RapidEye satellite scenes using deep learning. Three architectures are provided — MLP baseline, 1-D CNN, and Bidirectional LSTM — with the LSTM being the recommended choice because it explicitly models the temporal phenological signal.

![cover](cover1.png)

---

## Why temporal?

Classical single-image approaches use spectral and textural properties and struggle to separate crops with similar reflectance at any one date. Adding the time dimension exposes crop-specific growth curves (phenology), which are highly discriminative. A field of grapes and a field of almonds may look similar in March, but their NDVI trajectories diverge sharply through the growing season.

---

## Dataset

10 RapidEye images (5 spectral bands each: Blue, Green, Red, Red-Edge, NIR) acquired throughout the 2017 growing season, paired with the USDA Cropland Data Layer (CDL) as pixel-level ground truth.

| Date | Approximate Stage |
|------|-------------------|
| 2017-03-06 | Early spring |
| 2017-04-10 | Green-up |
| 2017-06-01 | Early summer |
| 2017-06-15 | Peak growth begins |
| 2017-07-08 | Summer peak |
| 2017-08-07 | Late summer |
| 2017-09-05 | Senescence |
| 2017-09-23 | Late senescence |
| 2017-10-15 | Post-harvest |
| 2017-12-07 | Winter dormancy |

**Target classes** (USDA CDL codes):

| Code | Crop |
|------|------|
| 36 | Alfalfa |
| 69 | Grapes |
| 75 | Almonds |
| 121 | Developed / Open Space |
| 225 | Dbl Crop WinWht/Corn |

**Download via Kaggle (recommended):**

```bash
# Install the Kaggle CLI if you haven't already
pip install kaggle

# Download and unzip into the Dataset/ directory
kaggle datasets download -d bhavesh907/crop-classificationcs2292017usgscroplanddata
unzip crop-classificationcs2292017usgscroplanddata.zip -d Dataset/
```

You need a Kaggle account and your `~/.kaggle/kaggle.json` API token in place.  
See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup instructions.

---

## Installation

### Conda (recommended — handles GDAL automatically)

```bash
conda env create -f environment.yml
conda activate crop-cls
```

### pip

GDAL must be installed separately via conda or your system package manager before pip-installing the Python bindings. See [`GDAL_INSTALLATION.md`](GDAL_INSTALLATION.md) for platform-specific instructions.

```bash
pip install -r requirements.txt
```

---

## Usage

### 1 — Build the dataset

Reads all GeoTIFFs and produces `Dataset/processed.csv` (~500 MB).

```bash
python scripts/preprocess.py
# Options:
#   --samples  Max pixels per class  (default: 100 000)
#   --seed     Random seed           (default: 42)
```

### 2 — Train a model

```bash
# Recommended: Bidirectional LSTM
python scripts/train.py --model lstm

# 1-D CNN
python scripts/train.py --model cnn

# MLP baseline
python scripts/train.py --model mlp

# All options
python scripts/train.py --model lstm --epochs 50 --batch-size 256 \
                        --test-size 0.15 --val-size 0.15 --dropout 0.3
```

Training artefacts written to:

| Path | Contents |
|------|----------|
| `models_saved/{model}_best.keras` | Best checkpoint (lowest val loss) |
| `models_saved/{model}_final.keras` | Final model |
| `models_saved/scaler.joblib` | Fitted `StandardScaler` |
| `models_saved/label_encoder.joblib` | Fitted `LabelEncoder` |
| `results/{model}_history.csv` | Per-epoch loss / accuracy |
| `results/{model}_report.txt` | Per-class precision / recall / F1 |
| `results/{model}_confusion_matrix.png` | Confusion matrix |
| `results/{model}_training_curve.png` | Loss & accuracy curves |

### 3 — Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
Crop-Classification/
├── src/
│   ├── config.py        # paths, class labels, constants
│   ├── preprocess.py    # GeoTIFF → processed CSV  (pixel correspondence fix)
│   ├── features.py      # NDVI and vegetation-index engineering
│   ├── models.py        # MLP, 1-D CNN, Bidirectional LSTM
│   ├── train.py         # training pipeline
│   └── evaluate.py      # metrics, confusion matrix, training curves
├── scripts/
│   ├── preprocess.py    # CLI entry point
│   └── train.py         # CLI entry point
├── tests/
│   ├── test_features.py
│   └── test_preprocess.py
├── Dataset/
│   └── download.txt     # dataset download instructions
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Architecture Comparison

| Model | Input shape | Key property |
|-------|-------------|--------------|
| MLP | `(T×F,)` flat | Fast baseline |
| 1-D CNN | `(T, F)` | Local temporal patterns via convolution |
| **BiLSTM** | `(T, F)` | **Full sequence context; models phenology in both temporal directions** |

Where **T = 10** timestamps, **F = 6** features per step (5 spectral bands + NDVI).

All models share:
- `StandardScaler` fitted on training data only
- Balanced class weights to handle the 13× pixel-count imbalance
- `EarlyStopping` + `ReduceLROnPlateau` to avoid over-training
- Stratified train / val / test splits (70 / 15 / 15)

---

## Key Fixes vs. Original Notebooks

| Issue | Original | Fixed |
|-------|----------|-------|
| **Pixel correspondence** | Each timestamp re-sampled independently — temporal sequences were spatially incoherent | Pixel indices sampled once per class; same locations used for all timestamps |
| **Dropout rate** | `Dropout(5)` — out of range, undefined behaviour | `Dropout(0.3)` |
| **Evaluation** | Only overall accuracy reported | Per-class precision / recall / F1 + confusion matrix |
| **Class imbalance** | Not addressed | Balanced class weights via `compute_class_weight` |
| **Scaler persistence** | Scaler fitted and discarded | Saved to `models_saved/scaler.joblib` |
| **Training callbacks** | Fixed 10 epochs | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| **Model persistence** | Model discarded after training | Saved as `.keras` checkpoint |
| **Framework** | Standalone Keras 2 / TF 1.x (EOL) | TensorFlow 2.x / Keras 3 |
| **Shadowing builtins** | `dict = {}` shadows Python built-in | Renamed to descriptive variable |
| **Column naming** | Generic `col_0 … col_50` | Semantic `t{t}_b{b}` and `t{t}_ndvi` |

---

## Reference

Rose M. Rustowicz et al., *Semantic Segmentation of Satellite Images Using Deep Learning*, Stanford CS229, 2017.  
[Paper PDF](http://cs229.stanford.edu/proj2017/final-reports/5243811.pdf)
