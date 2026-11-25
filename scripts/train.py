"""Train and evaluate one model on one split CSV (columns: path, class, label, split[, group_id, leaked]).

Examples
  # ADEC, original Train/Test partition with a 10% validation split (Table 3, ADEC row)
  python scripts/train.py --split splits/adec_original.csv --data-root data --out runs/adec_original
  # cross-dataset test (Section 4.5)
  python scripts/train.py --split splits/cross_ADEC_to_FDD.csv --data-root data --out runs/cross_ADEC_to_FDD
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_hybrid import config as C  # noqa: E402
from apple_hybrid.data import class_names, load_split  # noqa: E402
from apple_hybrid.engine import train_and_evaluate  # noqa: E402
from apple_hybrid.evaluation import one_image_per_group  # noqa: E402
from apple_hybrid.model import MODELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, help="split CSV")
ap.add_argument("--data-root", default=None, help="folder that contains ADEC/, FDD/ and AFQC/")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="proposed", choices=list(MODELS))
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--one-per-group", action="store_true", help="evaluate each duplicate group once")
a = ap.parse_args()

df = load_split(a.split, a.data_root)
if a.one_per_group:
    df = one_image_per_group(df)
train_and_evaluate(df, a.out, class_names(df), seed=a.seed, model_fn=MODELS[a.model],
                   title=os.path.basename(a.out.rstrip("/")))
