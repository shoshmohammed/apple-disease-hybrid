"""Documented split of a dataset that ships with Train/ and Test/ folders: the original test set is kept,
and 10% of the training images (whole duplicate groups, stratified by class) are held out for validation.
Test images whose duplicate group also occurs in Train are flagged as `leaked`.

  python scripts/check_duplicates.py --root data/ADEC --layout split --out dup/adec --emb
  python scripts/make_original_split.py --images dup/adec/images.csv --out splits/adec_original.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_hybrid import config as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="images.csv written by check_duplicates.py")
ap.add_argument("--out", required=True)
ap.add_argument("--val-frac", type=float, default=C.VAL_FRAC)
ap.add_argument("--seed", type=int, default=C.SEED)
a = ap.parse_args()

img = pd.read_csv(a.images)
classes = sorted(img["class"].unique())                       # same order as torchvision ImageFolder
img["label"] = img["class"].map({c: i for i, c in enumerate(classes)})
img["orig_split"] = img["split"].str.lower()
train_groups = set(img.loc[img.orig_split == "train", "group_id"])
img["leaked"] = (img.orig_split == "test") & img["group_id"].isin(train_groups)

tr = img[img.orig_split == "train"].reset_index()
sgkf = StratifiedGroupKFold(n_splits=int(round(1 / a.val_frac)), shuffle=True, random_state=a.seed)
_, val_idx = next(sgkf.split(tr, tr["label"], tr["group_id"]))
img["split"] = np.where(img.orig_split == "test", "test", "train")
img.loc[tr.loc[val_idx, "index"], "split"] = "val"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
img[["path", "class", "label", "group_id", "split", "leaked"]].to_csv(a.out, index=False)
tab = img.groupby(["class", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
print(tab.to_string())
print(f"Test images with a duplicate in Train: {int(img.leaked.sum())} / {int((img.split == 'test').sum())}")
