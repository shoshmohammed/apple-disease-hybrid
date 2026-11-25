"""Ablation study on identical folds (Section 4.6, Table 10).

Trains every variant on the same five ADEC folds (finished folds are skipped) and compares each variant with
the proposed model by an exact McNemar test on the pooled out-of-fold predictions.

  python scripts/ablation.py --data-root data --out runs/adec_cv5
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_hybrid.evaluation import KEYS, mcnemar, pooled_predictions, run_folds  # noqa: E402
from apple_hybrid.model import MODEL_LABELS, MODELS, count_parameters  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="splits/adec_fold{k}.csv")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--data-root", default=None)
ap.add_argument("--out", default="runs/adec_cv5")
ap.add_argument("--variants", nargs="+", default=["effnet_only", "swin_only", "concat", "mobilenet_attention", "proposed"])
a = ap.parse_args()

csvs = [a.folds.format(k=k) for k in range(1, a.k + 1)]
res = {v: run_folds(csvs, a.out, v, a.data_root) for v in a.variants}
ref = pooled_predictions(a.out, "proposed", a.k)[["path", "y_true", "y_pred"]]
n_classes = ref.y_true.nunique()
rows = []
for v in a.variants:
    t = pd.DataFrame([r["test"] for r in res[v]])
    row = {"Model": MODEL_LABELS[v], "Params (M)": round(count_parameters(MODELS[v](n_classes)), 2)}
    for m in KEYS:
        row[m] = f"{t[m].mean():.2f} ± {t[m].std(ddof=1):.2f}"
    if v != "proposed":
        b, c, p = mcnemar(ref, pooled_predictions(a.out, v, a.k)[["path", "y_true", "y_pred"]])
        row["b / c"], row["p-value"] = f"{b} / {c}", round(p, 3)
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(a.out, "ablation_summary.csv"), index=False)
pd.set_option("display.width", 200)
print(tab.to_string(index=False))
