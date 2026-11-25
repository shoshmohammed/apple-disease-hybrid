"""Grad-CAM of correct and incorrect predictions (Section 4.7, Figure 16).

Every image is explained by the model of the cross-validation fold in which it was a test image, so all maps are
out-of-sample. Target layer: final convolutional layer of the CNN branch (conv_head of EfficientNet-Lite).

  python scripts/gradcam.py --runs runs/adec_cv5/proposed --data-root data --out results/gradcam
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_hybrid import config as C  # noqa: E402
from apple_hybrid.data import class_names, eval_tf  # noqa: E402
from apple_hybrid.engine import device  # noqa: E402
from apple_hybrid.model import FruitDiseaseModel  # noqa: E402


def target_layer(model):
    if hasattr(model.cnn, "conv_head"):
        return model.cnn.conv_head
    return [m for m in model.cnn.modules() if isinstance(m, nn.Conv2d)][-1]


def grad_cam(model, x, cls):
    """Eq. (9): alpha_k = spatial mean of d y^c / d A^k;  Eq. (10): ReLU(sum_k alpha_k A^k)."""
    store = {}

    def fwd(m, i, o):                   # keep the activations and their gradient (tensor hook)
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))
    h = target_layer(model).register_forward_hook(fwd)
    model.zero_grad(); out = model(x); out[0, cls].backward(); h.remove()
    A, G = store["a"].detach(), store["g"].detach()
    cam = F.relu((G.mean(dim=(2, 3), keepdim=True) * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), out.softmax(1)[0].detach().cpu().numpy()


ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/adec_cv5/proposed", help="folder with fold1..fold5 of the proposed model")
ap.add_argument("--data-root", default=None)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", default="results/gradcam")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

pred = pd.concat([pd.read_csv(f"{a.runs}/fold{k}/test_predictions.csv").assign(fold=k) for k in range(1, a.k + 1)],
                 ignore_index=True)
if a.data_root:
    pred["path"] = pred["path"].map(lambda p: p if os.path.exists(p) else os.path.join(a.data_root, p))
classes = class_names(pred)
pred["correct"] = pred.y_true == pred.y_pred
print(f"{len(pred)} out-of-fold test images, {int((~pred.correct).sum())} misclassified")

rng = np.random.default_rng(C.SEED)
correct = pd.concat([g.iloc[[rng.integers(len(g))]] for _, g in pred[pred.correct].groupby("y_true")])
wrong = pred[~pred.correct].sort_values(["y_true", "path"])
wrong = pd.concat([g.head(1) for _, g in wrong.groupby("y_true")] + [wrong]).drop_duplicates("path").head(len(classes))

cache = {}


def fold_model(k):
    if k not in cache:
        m = FruitDiseaseModel(len(classes)).to(device)
        m.load_state_dict(torch.load(f"{a.runs}/fold{k}/best.pt", map_location=device))
        cache[k] = m.eval()
    return cache[k]


def panel(rows, name, title):
    rows = rows.reset_index(drop=True); n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.2)); ax = np.array(ax).reshape(2, n); rec = []
    for j, r in rows.iterrows():
        img = Image.open(r.path).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
        cam, prob = grad_cam(fold_model(int(r.fold)), eval_tf(img).unsqueeze(0).to(device), int(r.y_pred))
        ax[0, j].imshow(img); ax[1, j].imshow(img); ax[1, j].imshow(cam, cmap="jet", alpha=0.45)
        t, p = classes[int(r.y_true)], classes[int(r.y_pred)]
        ax[0, j].set_title(f"True: {t}\nPred: {p} ({100 * prob[int(r.y_pred)]:.0f}%)", fontsize=8,
                           color="black" if t == p else "red")
        for x in ax[:, j]:
            x.axis("off")
        rec.append({"path": r.path, "fold": int(r.fold), "true": t, "pred": p, "confidence": round(float(prob[int(r.y_pred)]), 4)})
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(f"{a.out}/{name}.png", dpi=300); plt.close()
    pd.DataFrame(rec).to_csv(f"{a.out}/{name}.csv", index=False)


panel(correct, "gradcam_correct", "Grad-CAM, correctly classified ADEC test images (one per class)")
panel(wrong, "gradcam_incorrect", "Grad-CAM, misclassified ADEC test images")
print("Saved to", a.out)
