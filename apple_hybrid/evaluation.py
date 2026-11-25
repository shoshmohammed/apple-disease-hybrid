"""Cross-validation driver, summary statistics (mean ± SD, t-based 95% CI, bootstrap) and McNemar test."""
import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from . import config as C
from .data import class_names, load_split
from .engine import plot_cm, train_and_evaluate
from .model import MODELS

KEYS = ["accuracy", "precision", "recall", "f1"]


def one_image_per_group(df):
    """Duplicate-aware evaluation: every duplicate group is tested once (one representative image)."""
    te = df[df.split == "test"].sort_values("path").drop_duplicates("group_id")
    return pd.concat([df[df.split != "test"], te], ignore_index=True)


def run_folds(fold_csvs, out_root, model="proposed", data_root=None, seed=C.SEED):
    """Train/evaluate one model on each fold; finished folds (metrics.json present) are skipped."""
    results = []
    for k, csv in enumerate(fold_csvs, 1):
        out = os.path.join(out_root, model, f"fold{k}")
        if os.path.exists(f"{out}/metrics.json"):
            print(f"[{model} fold{k}] already done -> {out}")
            results.append(json.load(open(f"{out}/metrics.json"))); continue
        df = one_image_per_group(load_split(csv, data_root))
        results.append(train_and_evaluate(df, out, class_names(df), seed=seed, model_fn=MODELS[model],
                                          title=f"{model} fold{k}"))
    return results


def summarize_folds(results, names, out_dir, title=""):
    """Per-fold table, mean ± SD with t-based 95% CI, pooled per-class report, bootstrap CI."""
    os.makedirs(out_dir, exist_ok=True)
    K = len(results)
    tab = pd.DataFrame([{"fold": k + 1, "n_test": r["test"]["n"], "best_epoch": r["best_epoch"],
                         **{m: r["test"][m] for m in KEYS}} for k, r in enumerate(results)])
    mean, sd = tab[KEYS].mean(), tab[KEYS].std(ddof=1)
    half = st.t.ppf(0.975, K - 1) * sd / np.sqrt(K)
    summ = pd.DataFrame({"mean": mean, "sd": sd, "ci95_low": mean - half,
                         "ci95_high": np.minimum(mean + half, 100)}).round(2)
    tab.to_csv(f"{out_dir}/per_fold.csv", index=False); summ.to_csv(f"{out_dir}/summary_mean_sd_ci.csv")
    print(tab.round(2).to_string(index=False)); print(summ.to_string())
    return tab, summ


def pooled_predictions(out_root, model, K=5):
    return pd.concat([pd.read_csv(os.path.join(out_root, model, f"fold{k}", "test_predictions.csv")).assign(fold=k)
                      for k in range(1, K + 1)], ignore_index=True)


def pooled_report(pp, names, out_dir, title="", B=2000, seed=C.SEED):
    y, p = pp.y_true.values, pp.y_pred.values
    plot_cm(y, p, names, f"{out_dir}/confusion_matrix_pooled.png", title)
    rep = pd.DataFrame(classification_report(y, p, labels=range(len(names)), target_names=names,
                                             output_dict=True, zero_division=0)).T.drop(index="accuracy")
    rep[["precision", "recall", "f1-score"]] *= 100
    rep.round(2).to_csv(f"{out_dir}/per_class_pooled.csv")
    rng = np.random.default_rng(seed); boot = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        boot.append([100 * accuracy_score(y[i], p[i]),
                     100 * precision_recall_fscore_support(y[i], p[i], average="macro", zero_division=0)[2]])
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    acc = 100 * accuracy_score(y, p); f1 = 100 * precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]
    print(rep.round(2).to_string())
    print(f"Pooled accuracy {acc:.2f}% (bootstrap 95% CI {lo[0]:.2f}-{hi[0]:.2f}); macro-F1 {f1:.2f}% ({lo[1]:.2f}-{hi[1]:.2f})")
    return rep


def mcnemar(ref, other):
    """Exact McNemar test on paired predictions (dataframes with path, y_true, y_pred).
    b = ref correct & other wrong, c = ref wrong & other correct."""
    m = ref.merge(other, on=["path", "y_true"], suffixes=("_ref", "_oth"))
    b = int(((m.y_pred_ref == m.y_true) & (m.y_pred_oth != m.y_true)).sum())
    c = int(((m.y_pred_ref != m.y_true) & (m.y_pred_oth == m.y_true)).sum())
    p = st.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p
