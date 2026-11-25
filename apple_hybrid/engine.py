"""Training and evaluation: validation-based checkpoint selection, test set evaluated once."""
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from . import config as C
from .data import CsvImageDataset, eval_tf, train_tf
from .model import FruitDiseaseModel, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    ys, ps, probs = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * y.size(0); n += y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            ys += y.tolist(); ps += pred.tolist(); probs += out.softmax(1).tolist()
    return tot_loss / n, correct / n, np.array(ys), np.array(ps), np.array(probs)


def metrics(y, p):
    """Accuracy and macro-averaged precision, recall and F1 (Eqs. 5-8), in %."""
    P, R, F, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"n": int(len(y)), "correct": int((y == p).sum()), "accuracy": 100 * accuracy_score(y, p),
            "precision": 100 * P, "recall": 100 * R, "f1": 100 * F}


def plot_cm(y, p, names, path, title):
    cm = confusion_matrix(y, p, labels=range(len(names)))
    plt.figure(figsize=(1.1 * len(names) + 3, 0.9 * len(names) + 2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    e = h["epoch"]
    ax[0].plot(e, h["train_loss"], "o-", label="Train"); ax[0].plot(e, h["val_loss"], "s-", label="Validation")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy loss (mean per image)"); ax[0].legend()
    ax[1].plot(e, h["train_acc"], "o-", label="Train"); ax[1].plot(e, h["val_acc"], "s-", label="Validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def train_and_evaluate(df, out_dir, class_names, seed=C.SEED, model_fn=None, title=""):
    """df: rows with path, label, split in {train, val, test} [, leaked].
    Trains for NUM_EPOCHS, keeps the checkpoint with the best validation accuracy
    (ties -> lower validation loss) and evaluates the test split once with it."""
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    def dl(split, tf, shuffle):
        return DataLoader(CsvImageDataset(df[df.split == split], tf), batch_size=C.BATCH_SIZE,
                          shuffle=shuffle, num_workers=C.NUM_WORKERS, generator=g)

    tr_dl, va_dl, te_dl = dl("train", train_tf, True), dl("val", eval_tf, False), dl("test", eval_tf, False)
    print(f"[{title}] train={len(tr_dl.dataset)} val={len(va_dl.dataset)} test={len(te_dl.dataset)}")

    model = (model_fn or FruitDiseaseModel)(len(class_names)).to(device)
    crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=C.LR)
    hist, best, best_ep, t0 = [], (-1, 1e9), 0, time.time()
    for ep in range(1, C.NUM_EPOCHS + 1):
        tl, ta, *_ = run_epoch(model, tr_dl, crit, opt)
        vl, va, *_ = run_epoch(model, va_dl, crit)
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        flag = ""
        if (va, -vl) > (best[0], -best[1]):
            best, best_ep = (va, vl), ep
            torch.save(model.state_dict(), f"{out_dir}/best.pt"); flag = "  <- best"
        print(f"Epoch {ep:2d}/{C.NUM_EPOCHS}  train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}{flag}")
    train_time = time.time() - t0
    h = pd.DataFrame(hist); h.to_csv(f"{out_dir}/history.csv", index=False); plot_curves(h, f"{out_dir}/curves.png")

    model.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=device))
    _, _, y, p, prob = run_epoch(model, te_dl, crit)        # the single test evaluation
    test_df = df[df.split == "test"].reset_index(drop=True).assign(y_true=y, y_pred=p)
    test_df.to_csv(f"{out_dir}/test_predictions.csv", index=False)

    res = {"tag": title, "seed": seed, "best_epoch": best_ep, "params_M": round(count_parameters(model), 2),
           "train_time_s": round(train_time, 1), "test": metrics(y, p)}
    if "leaked" in test_df and test_df.leaked.astype(bool).any():
        clean = ~test_df.leaked.astype(bool).values
        res["test_clean"] = metrics(y[clean], p[clean])     # test images without a duplicate in training
    res["per_class"] = classification_report(y, p, labels=range(len(class_names)), target_names=class_names,
                                             output_dict=True, zero_division=0)
    json.dump(res, open(f"{out_dir}/metrics.json", "w"), indent=2)
    plot_cm(y, p, class_names, f"{out_dir}/confusion_matrix.png", title)
    print(classification_report(y, p, labels=range(len(class_names)), target_names=class_names,
                                digits=4, zero_division=0))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2))
    return res
