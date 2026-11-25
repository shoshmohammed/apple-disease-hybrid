"""Duplicate-aware five-fold cross-validation on ADEC (Section 4.5, Tables 7 and 8).

  python scripts/run_kfold.py --data-root data --out runs/adec_cv5                 # proposed model
  python scripts/run_kfold.py --data-root data --out runs/adec_cv5 --model swin_only
Each duplicate group is tested once; finished folds are skipped, so the command can be re-run after a disconnect.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_hybrid.data import class_names, load_split  # noqa: E402
from apple_hybrid.evaluation import pooled_predictions, pooled_report, run_folds, summarize_folds  # noqa: E402
from apple_hybrid.model import MODELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="splits/adec_fold{k}.csv", help="pattern of the fold CSVs")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--data-root", default=None)
ap.add_argument("--out", default="runs/adec_cv5")
ap.add_argument("--model", default="proposed", choices=list(MODELS))
a = ap.parse_args()

csvs = [a.folds.format(k=k) for k in range(1, a.k + 1)]
names = class_names(load_split(csvs[0]))
res = run_folds(csvs, a.out, a.model, a.data_root)
out_dir = os.path.join(a.out, a.model, "summary")
summarize_folds(res, names, out_dir)
pp = pooled_predictions(a.out, a.model, a.k)
pp.to_csv(f"{out_dir}/pooled_predictions.csv", index=False)
pooled_report(pp, names, out_dir, f"{a.model} - {a.k}-fold CV (pooled, {len(pp)} unique images)")
