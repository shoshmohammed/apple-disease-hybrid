# Hybrid EfficientNet-Lite + Swin Transformer for apple fruit disease classification

This repository contains the code, data splits and results for the paper:

> S. M. Ali, H. M. Hamza, W. M. Elhady, K. M. Hosny. *An Efficient Hybrid Model for Detection and Classification of Apple Fruit Diseases Using Explainable AI, EfficientNet-Lite, and Swin Transformer.* Scientific Reports (under revision).

It provides the following:
- The proposed model: an EfficientNet-Lite0 branch for local features and a Swin-Tiny branch for global features, combined by attention-based feature fusion.
- The duplicate screening of the three public datasets, using MD5, perceptual hashing over 8 orientations, and DINOv2 similarity.
- The exact split files used in the paper.
- Scripts that reproduce the duplicate-aware five-fold cross-validation, the cross-dataset test, the ablation study, and the Grad-CAM figures.

## Repository layout

```
apple_hybrid/          model, data loading, training loop and evaluation utilities
  config.py            all hyper-parameters used in the paper
  model.py             proposed model (Eqs. 1-4) and the ablation baselines (Table 10)
  engine.py            training with validation-based checkpoint selection; the test set is evaluated once
  evaluation.py        k-fold driver, mean ± SD, t-based 95% CI, bootstrap CI, exact McNemar test
scripts/
  check_duplicates.py  exact and near-duplicate screening (Section 3.1.1)
  make_original_split.py  original Train/Test partition + 10% group-aware validation split
  make_splits.py       duplicate-aware, class-stratified k-fold splits
  train.py             train/evaluate one model on one split file
  run_kfold.py         duplicate-aware five-fold cross-validation (Tables 7 and 8)
  ablation.py          ablation study on identical folds (Table 10)
  gradcam.py           Grad-CAM of correct and incorrect predictions (Figure 16)
splits/                split files used in the paper (paths relative to the data folder)
duplicates/            duplicate-screening reports (image list, duplicate pairs, summary)
results/               metrics and figures reported in the paper
```

## Installation

```bash
pip install -r requirements.txt
```

All experiments were run with Python 3, PyTorch and timm on a single NVIDIA T4 GPU (Google Colab). The ImageNet weights of `tf_efficientnet_lite0` and `swin_tiny_patch4_window7_224` are downloaded automatically by timm.

## Data

The three datasets are public:
- **FDD**: the apple fruit disease dataset of Patidar and Chakravorty (2024), reference [15] of the paper.
- **AFQC and ADEC**: released by Zhang et al., *PLoS One* 20(5): e0322586 (2025), reference [21] of the paper.

Download them and arrange them as follows (each dataset keeps its original `Train/` and `Test/` folders):

```
data/
  ADEC/Train/<class>/...   ADEC/Test/<class>/...
  AFQC/Train/<class>/...   AFQC/Test/<class>/...
  FDD/Train/<class>/...    FDD/Test/<class>/...
```

Every split file in `splits/` lists image paths relative to `data/`. The columns are:
- `path` and `class`;
- `label`;
- `group_id`: the duplicate group of the image;
- `split`: `train`, `val` or `test`;
- `leaked`, in the original partition only: whether the test image has a duplicate in the training set.

Pass `--data-root data` to the scripts.

## Reproducing the paper

| Paper | Command |
|---|---|
| Duplicate screening (Section 3.1.1) | `python scripts/check_duplicates.py --root data/ADEC --layout split --out duplicates/adec --emb` |
| ADEC, original partition (Table 3) | `python scripts/train.py --split splits/adec_original.csv --data-root data --out runs/adec_original` |
| Duplicate-aware 5-fold CV (Tables 7 and 8) | `python scripts/run_kfold.py --data-root data --out runs/adec_cv5` |
| Cross-dataset test (Section 4.5) | `python scripts/train.py --split splits/cross_ADEC_to_FDD.csv --data-root data --out runs/cross_ADEC_to_FDD`, and the same with `cross_FDD_to_ADEC.csv` |
| Ablation study (Table 10) | `python scripts/ablation.py --data-root data --out runs/adec_cv5` |
| Grad-CAM, correct and incorrect (Figure 16) | `python scripts/gradcam.py --runs runs/adec_cv5/proposed --data-root data --out results/gradcam` |

**Duplicates in the cross-validation:**
- In every fold, all copies of an image belong to the same partition.
- Each duplicate group is tested once, using one representative image per group.
- The 1,641 unique ADEC images are therefore each tested exactly once across the five folds.

**Restarting:** finished folds are detected and skipped, so a command can be restarted after an interruption.

### Settings (`apple_hybrid/config.py`)

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-4 |
| Batch size | 16 |
| Epochs | 15 |
| Loss | Cross-entropy |
| Input | 224 × 224, normalized with mean = std = 0.5 |
| Augmentation | None |
| Validation | 10% of the training groups (group-aware, stratified by class) |
| Checkpoint selection | Highest validation accuracy; ties go to the lower validation loss |
| Test evaluation | Once, with the selected checkpoint |
| Seed | 42 |

## Main results

Duplicate-aware five-fold cross-validation on ADEC (mean ± SD over the folds; Table 10 of the paper):

| Model | Params (M) | Accuracy (%) | F1-score (%) | McNemar b / c | p |
|---|---|---|---|---|---|
| EfficientNet-Lite only | 4.16 | 97.20 ± 1.55 | 96.97 ± 1.62 | 33 / 15 | 0.013 |
| Swin Transformer only | 28.05 | 98.35 ± 0.71 | 98.14 ± 0.76 | 16 / 17 | 1.000 |
| EfficientNet-Lite + Swin (concatenation) | 32.20 | 98.60 ± 0.40 | 98.46 ± 0.65 | 13 / 18 | 0.473 |
| MobileNetV3-Small + Swin (attention fusion) | 30.06 | 98.66 ± 0.59 | 98.45 ± 0.66 | 12 / 18 | 0.362 |
| Proposed: EfficientNet-Lite + Swin (attention fusion) | 32.86 | 98.29 ± 0.99 | 98.17 ± 1.10 | – | – |

The McNemar columns compare each variant with the proposed model:
- b = images that only the proposed model classifies correctly.
- c = images that only the variant classifies correctly.

## License

MIT, see `LICENSE`. The datasets keep the licenses of their original publishers.
