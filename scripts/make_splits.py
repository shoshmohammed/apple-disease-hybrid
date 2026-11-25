"""
make_splits.py  --  Reviewer 1, Comments 1 & 2 (leakage-free, reproducible splits)

Builds duplicate-aware, class-stratified partitions from images.csv produced by
check_duplicates.py (--layout flat on the WHOLE dataset, so every image is pooled).
All images that share a `group_id` (exact or near duplicates) are always placed in
the SAME partition, so no duplicate can leak between train / validation / test.

Modes
  --mode holdout : one train/val/test split (default 80/10/10), seed-controlled.
  --mode kfold   : K-fold cross-validation (default 5). In every fold one fold is
                   the test set, and --val-frac of the remaining groups becomes the
                   validation set (used ONLY for checkpoint/epoch selection).

Outputs (in --out):
  holdout_seed{S}.csv  or  fold{k}.csv     columns: path, class, label, group_id, split
  split_table.csv                         per-class counts per split (for the paper table)

Usage
  python check_duplicates.py --root /data/ADEC_all --layout flat --out dup_ADEC
  python make_splits.py --images dup_ADEC/images.csv --mode kfold --k 5 --seed 42 --out splits_ADEC
  python make_splits.py --images dup_ADEC/images.csv --mode holdout --seed 42 --out splits_ADEC
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_split(df, frac, seed):
    """Return a boolean mask selecting ~frac of df, stratified by class, grouped by group_id."""
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, take = next(sgkf.split(df, df["label"], df["group_id"]))
    mask = np.zeros(len(df), dtype=bool)
    mask[take] = True
    return mask


def check_no_leak(df):
    per_group = df.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all(), "A duplicate group spans more than one split!"


def split_table(frames):
    t = []
    for name, d in frames.items():
        c = d.groupby(["class", "split"]).size().unstack(fill_value=0)
        c["fold"] = name
        t.append(c.reset_index())
    return pd.concat(t, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="images.csv from check_duplicates.py")
    ap.add_argument("--mode", choices=["holdout", "kfold"], default="kfold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.images)[["path", "class", "group_id"]].reset_index(drop=True)
    classes = sorted(df["class"].unique())
    df["label"] = df["class"].map({c: i for i, c in enumerate(classes)})
    print(f"{len(df)} images, {len(classes)} classes, {df['group_id'].nunique()} groups")
    print("class -> label:", {c: i for i, c in enumerate(classes)})

    frames = {}
    if a.mode == "holdout":
        d = df.copy()
        test = group_split(d, a.test_frac, a.seed)
        rest = d[~test].reset_index()
        val_r = group_split(rest, a.val_frac / (1 - a.test_frac), a.seed + 1)
        d["split"] = "train"
        d.loc[test, "split"] = "test"
        d.loc[rest.loc[val_r, "index"], "split"] = "val"
        check_no_leak(d)
        name = f"holdout_seed{a.seed}"
        d.to_csv(f"{a.out}/{name}.csv", index=False)
        frames[name] = d
    else:
        sgkf = StratifiedGroupKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
        for k, (_, te) in enumerate(sgkf.split(df, df["label"], df["group_id"]), 1):
            d = df.copy()
            d["split"] = "train"
            d.loc[te, "split"] = "test"
            rest = d[d["split"] == "train"].reset_index()
            val_r = group_split(rest, a.val_frac, a.seed + k)
            d.loc[rest.loc[val_r, "index"], "split"] = "val"
            check_no_leak(d)
            d.to_csv(f"{a.out}/fold{k}.csv", index=False)
            frames[f"fold{k}"] = d

    tab = split_table(frames)
    tab.to_csv(f"{a.out}/split_table.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nNo duplicate group crosses a partition. Files saved in {a.out}/")


if __name__ == "__main__":
    main()
