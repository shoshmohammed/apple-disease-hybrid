"""
check_duplicates.py  --  Reviewer 1, Comment 1 (leakage check)

Finds exact and near-duplicate images inside a dataset and, most importantly,
ACROSS the train / validation / test partitions.

Two detectors are combined:
  1. Exact duplicates  : MD5 of the file bytes.
  2. Near duplicates   : 64-bit perceptual hash (pHash) computed for all 8
                         flip/rotation variants of each image, so rotated or
                         mirrored copies (typical of pre-augmented Kaggle sets)
                         are caught. Pairs with Hamming distance <= --phash-thresh.
  3. (optional) --emb  : cosine similarity of DINOv2 embeddings (timm) to catch
                         re-crops / re-scaled / re-compressed copies that pHash misses.

Supported folder layouts
  --layout split : ROOT/<split>/<class>/<image>   (e.g. ROOT/train/Apple_scab/x.jpg)
  --layout flat  : ROOT/<class>/<image>           (whole dataset, no split yet)

Outputs (in --out):
  images.csv        one row per image (path, split, class, md5, group_id)
  pairs.csv         every duplicate pair found, with distance/similarity
  summary.txt       counts to report in the paper / response letter
  cross_split.png   montage of cross-partition pairs for visual verification
`group_id` links all images connected by a duplicate edge; use it in make_splits.py.

Usage
  pip install imagehash pillow numpy pandas
  python check_duplicates.py --root /data/ADEC --layout split --out dup_ADEC
  python check_duplicates.py --root /data/ADEC --layout split --out dup_ADEC --emb
"""
import argparse, hashlib, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- collection
def collect(root, layout):
    rows = []
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        rel = p.relative_to(root).parts
        if layout == "split":
            if len(rel) < 3:
                continue
            split, cls = rel[0], rel[1]
        else:
            if len(rel) < 2:
                continue
            split, cls = "all", rel[0]
        rows.append({"path": str(p), "split": split, "class": cls})
    if not rows:
        sys.exit(f"No images found under {root} with layout={layout}")
    return pd.DataFrame(rows)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dihedral(img):
    """8 flip/rotation variants of a PIL image."""
    out = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        out += [r, ImageOps.mirror(r)]
    return out


def hash_to_int(h):
    return int(str(h), 16)


# ---------------------------------------------------------------- pHash search
def phash_pairs(df, thresh, chunk=128):
    n = len(df)
    variants = np.zeros((n, 8), dtype=np.uint64)
    for i, p in enumerate(df["path"]):
        img = Image.open(p).convert("RGB")
        img.thumbnail((256, 256))
        variants[i] = [hash_to_int(imagehash.phash(v)) for v in dihedral(img)]
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{n}")
    canon = variants[:, 0]                      # original orientation of every image
    pairs = []
    for s in range(0, n, chunk):
        q = variants[s:s + chunk]               # (c, 8)
        x = q[:, :, None] ^ canon[None, None, :]  # (c, 8, n)
        d = POPCOUNT[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)
        dmin = d.min(1)                         # min over the 8 variants -> (c, n)
        ii, jj = np.nonzero(dmin <= thresh)
        for a, j in zip(ii, jj):
            i = s + a
            if j > i:
                pairs.append((i, int(j), int(dmin[a, j])))
    return pairs


# ---------------------------------------------------------------- embeddings
def emb_pairs(df, thresh, batch=64):
    import torch, timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=224).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg)
    feats = []
    with torch.no_grad():
        for s in range(0, len(df), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB"))
                             for p in df["path"].iloc[s:s + batch]]).to(dev)
            feats.append(torch.nn.functional.normalize(model(x), dim=1).cpu())
    f = torch.cat(feats)
    pairs = []
    for s in range(0, len(f), 512):
        sim = f[s:s + 512] @ f.T
        ii, jj = torch.nonzero(sim >= thresh, as_tuple=True)
        for a, j in zip(ii.tolist(), jj.tolist()):
            i = s + a
            if j > i:
                pairs.append((i, j, float(sim[a, j])))
    return pairs


# ---------------------------------------------------------------- grouping
def union_find_groups(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [remap[r] for r in roots]


def montage(df, pairs, out_png, max_pairs=40, size=160):
    pairs = pairs[:max_pairs]
    if not pairs:
        return
    W, H = size * 2 + 10, size + 28
    canvas = Image.new("RGB", (W * 4, H * ((len(pairs) + 3) // 4)), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(pairs):
        x0, y0 = (k % 4) * W, (k // 4) * H
        for t, idx in enumerate((r["i"], r["j"])):
            im = Image.open(df.loc[idx, "path"]).convert("RGB")
            im.thumbnail((size, size))
            canvas.paste(im, (x0 + t * (size + 5), y0 + 26))
        d.text((x0 + 2, y0 + 2),
               f"{df.loc[r['i'], 'split']}/{df.loc[r['j'], 'split']}  "
               f"{r['method']}={r['score']}", fill="red")
    canvas.save(out_png)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layout", choices=["split", "flat"], default="split")
    ap.add_argument("--out", default="dup_report")
    ap.add_argument("--phash-thresh", type=int, default=6,
                    help="max Hamming distance (of 64 bits) to call a near-duplicate")
    ap.add_argument("--emb", action="store_true", help="also use DINOv2 embeddings")
    ap.add_argument("--emb-thresh", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = collect(a.root, a.layout)
    print(f"{len(df)} images | splits: {df['split'].value_counts().to_dict()}")

    print("1/3 exact duplicates (MD5) ...")
    df["md5"] = [md5(p) for p in df["path"]]
    rows = []
    for _, g in df.groupby("md5"):
        idx = list(g.index)
        rows += [(i, j, "md5", 0) for k, i in enumerate(idx) for j in idx[k + 1:]]

    print(f"2/3 near duplicates (pHash x 8 orientations, thresh={a.phash_thresh}) ...")
    rows += [(i, j, "phash", d) for i, j, d in phash_pairs(df, a.phash_thresh)]

    if a.emb:
        print(f"3/3 embedding similarity (DINOv2, cos >= {a.emb_thresh}) ...")
        rows += [(i, j, "dinov2_cos", round(c, 4)) for i, j, c in emb_pairs(df, a.emb_thresh)]

    pairs = (pd.DataFrame(rows, columns=["i", "j", "method", "score"])
             .drop_duplicates(["i", "j", "method"]))
    for c in ["path", "split", "class"]:
        pairs[f"{c}_i"] = df.loc[pairs["i"], c].values
        pairs[f"{c}_j"] = df.loc[pairs["j"], c].values
    pairs["cross_split"] = pairs["split_i"] != pairs["split_j"]
    pairs["label_conflict"] = pairs["class_i"] != pairs["class_j"]
    pairs.to_csv(f"{a.out}/pairs.csv", index=False)

    df["group_id"] = union_find_groups(len(df), list(zip(pairs["i"], pairs["j"])))
    df.to_csv(f"{a.out}/images.csv", index=False)

    uniq = pairs.drop_duplicates(["i", "j"])
    cross = uniq[uniq["cross_split"]]
    test_like = [s for s in df["split"].unique() if s.lower() in {"test", "testing", "val", "valid", "validation"}]
    contaminated = set()
    for _, r in cross.iterrows():
        for side in ("i", "j"):
            if r[f"split_{side}"] in test_like:
                contaminated.add(r[side])
    n_eval = int(df["split"].isin(test_like).sum())

    lines = [
        f"Dataset root           : {a.root}",
        f"Images                 : {len(df)}",
        f"Per split              : {df['split'].value_counts().to_dict()}",
        f"Duplicate pairs (all)  : {len(uniq)}  "
        f"(md5={int((pairs.method == 'md5').sum())}, phash={int((pairs.method == 'phash').sum())}"
        + (f", dinov2={int((pairs.method == 'dinov2_cos').sum())}" if a.emb else "") + ")",
        f"Cross-split pairs      : {len(cross)}",
        f"Pairs with label clash : {int(uniq['label_conflict'].sum())}",
        f"Eval/test images that have a near-duplicate in another split: "
        f"{len(contaminated)} / {n_eval}"
        + (f" ({100 * len(contaminated) / n_eval:.2f}%)" if n_eval else ""),
        f"Duplicate groups (size>1): {int((df['group_id'].value_counts() > 1).sum())}",
        f"Unique images after merging duplicates: {df['group_id'].nunique()}",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    open(f"{a.out}/summary.txt", "w").write(txt + "\n")

    show = cross.sort_values("score").to_dict("records") if len(cross) else \
        uniq.sort_values("score").to_dict("records")
    montage(df, show, f"{a.out}/cross_split.png")
    print(f"\nSaved: {a.out}/pairs.csv, images.csv, summary.txt, cross_split.png")


if __name__ == "__main__":
    main()
