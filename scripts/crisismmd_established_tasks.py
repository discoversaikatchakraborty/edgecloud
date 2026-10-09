"""Established CrisisMMD tasks on this paper's representation, against the published baselines.

Reviewer R2.3(c) asked for an established CrisisMMD task and baseline. Ofli, Alam and
Imran (ISCRAM 2020, arXiv:2004.11838) define two: informative vs not-informative and
five humanitarian categories, on text-image pairs whose two labels agree, with fixed
train/dev/test splits (crisismmd_datasplit_agreed_label.zip). They report test accuracy
80.8 / 83.3 / 84.4 (informative) and 70.4 / 76.8 / 78.4 (humanitarian) for text, image
and text+image. This script scores the same splits with the representation this paper
uses everywhere else (CLIP ViT-B/32 embeddings, prepare_crisismmd.py's encoder) and a
logistic-regression head chosen on the dev split, in two forms: the full 512-d embedding
and the reduced 8-d text / 10-d image PCA form the budgeting pipeline consumes. It does
not reproduce their CNN/VGG16 models; it measures the gap between this paper's
representation and their published numbers on their task.

Usage: python crisismmd_established_tasks.py --src <CrisisMMD_v2.0 dir> --splits <agreed_label dir>
Outputs: results/established_tasks/{embeddings cache in scratch}, established_tasks.csv,
         established_tasks.json, run_log.txt (each task appended as it completes).
"""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "established_tasks"
PUBLISHED = {"informative": {"text": 80.8, "image": 83.3, "text+image": 84.4},
             "humanitarian": {"text": 70.4, "image": 76.8, "text+image": 78.4}}
TEXT_DIMS, IMAGE_DIMS = 8, 10
CLIP_MODEL = "openai/clip-vit-base-patch32"


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a") as fh: fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def load_split(splits, task, part):
    f = splits / f"task_{task}_text_img_agreed_lab_{part}.tsv"
    d = pd.read_csv(f, sep="\t", dtype=str)
    d["part"] = part
    return d


def encode_all(src, cache, texts, images):
    """CLIP features for every unique text and image, cached to disk."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval(); proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    emb = lambda o: getattr(o, "pooler_output", o)
    T = {}
    if (cache / "text.npz").exists():
        z = np.load(cache / "text.npz", allow_pickle=True); T = dict(zip(z["keys"], z["vals"]))
    todo = [t for t in texts if t not in T]
    log(f"text: {len(texts)} unique, {len(todo)} to encode")
    with torch.no_grad():
        for i in range(0, len(todo), 64):
            b = [str(x)[:300] for x in todo[i:i + 64]]
            inp = proc(text=b, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            v = emb(model.get_text_features(**inp)).cpu().numpy().astype("float64")
            for k, row in zip(todo[i:i + 64], v): T[k] = row
    np.savez(cache / "text.npz", keys=np.array(list(T.keys()), dtype=object), vals=np.stack(list(T.values())))
    I = {}
    if (cache / "image.npz").exists():
        z = np.load(cache / "image.npz", allow_pickle=True); I = dict(zip(z["keys"], z["vals"]))
    todo = [p for p in images if p not in I]
    log(f"image: {len(images)} unique, {len(todo)} to encode")
    batch, keys = [], []
    with torch.no_grad():
        for j, rel in enumerate(todo):
            try:
                batch.append(Image.open(src / rel).convert("RGB")); keys.append(rel)
            except Exception as e:
                log(f"  unreadable image {rel}: {e}"); I[rel] = np.full(512, np.nan)
            if len(batch) == 64 or (j == len(todo) - 1 and batch):
                inp = proc(images=batch, return_tensors="pt").to(dev)
                v = emb(model.get_image_features(**inp)).cpu().numpy().astype("float64")
                for k, row in zip(keys, v): I[k] = row
                batch, keys = [], []
                if (len(I) % 1280) < 64: 
                    np.savez(cache / "image.npz", keys=np.array(list(I.keys()), dtype=object), vals=np.stack(list(I.values())))
                    log(f"  images encoded: {len(I)}")
    np.savez(cache / "image.npz", keys=np.array(list(I.keys()), dtype=object), vals=np.stack(list(I.values())))
    return T, I


def fit_eval(Xtr, ytr, Xdv, ydv, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    sc = StandardScaler().fit(Xtr); Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        a = accuracy_score(ydv, m.predict(Xdv))
        if best is None or a > best[0]: best = (a, C, m)
    p = best[2].predict(Xte)
    return {"C": best[1], "dev_accuracy": round(100 * best[0], 1), "accuracy": round(100 * accuracy_score(yte, p), 1),
            "precision_w": round(100 * precision_score(yte, p, average="weighted", zero_division=0), 1),
            "recall_w": round(100 * recall_score(yte, p, average="weighted", zero_division=0), 1),
            "f1_w": round(100 * f1_score(yte, p, average="weighted"), 1),
            "f1_macro": round(100 * f1_score(yte, p, average="macro"), 1), "n_test": int(len(yte))}


def main(src, splits, cache):
    from sklearn.decomposition import PCA
    tasks = {}
    for task in ("informative", "humanitarian"):
        tasks[task] = pd.concat([load_split(splits, task, p) for p in ("train", "dev", "test")], ignore_index=True)
    texts = sorted({t for d in tasks.values() for t in d.tweet_text.astype(str)})
    images = sorted({i for d in tasks.values() for i in d.image.astype(str)})
    T, I = encode_all(src, cache, texts, images)
    rows = []
    for task, d in tasks.items():
        XT = np.stack([T[t] for t in d.tweet_text.astype(str)]); XI = np.stack([I[i] for i in d.image.astype(str)])
        ok = ~np.isnan(XI).any(1)
        d = d[ok].reset_index(drop=True); XT, XI = XT[ok], XI[ok]
        y = d.label.astype(str).values; classes = sorted(set(y)); yi = np.array([classes.index(v) for v in y])
        tr, dv, te = (d.part == "train").values, (d.part == "dev").values, (d.part == "test").values
        log(f"{task}: classes {classes}; train {tr.sum()} dev {dv.sum()} test {te.sum()} (images unreadable: {int((~ok).sum())})")
        pT = PCA(TEXT_DIMS, random_state=0).fit(XT[tr]); pI = PCA(IMAGE_DIMS, random_state=0).fit(XI[tr])
        RT, RI = pT.transform(XT), pI.transform(XI)
        feats = {"full": {"text": XT, "image": XI, "text+image": np.hstack([XT, XI])},
                 "reduced_8_10": {"text": RT, "image": RI, "text+image": np.hstack([RT, RI])}}
        for form, fs in feats.items():
            for mod, X in fs.items():
                r = fit_eval(X[tr], yi[tr], X[dv], yi[dv], X[te], yi[te])
                r.update({"task": task, "representation": form, "modality": mod, "published_accuracy": PUBLISHED[task][mod],
                          "accuracy_minus_published": round(r["accuracy"] - PUBLISHED[task][mod], 1), "n_classes": len(classes)})
                rows.append(r); log(f"  {form:12s} {mod:10s} acc {r['accuracy']:5.1f} (published {PUBLISHED[task][mod]}) f1w {r['f1_w']} f1m {r['f1_macro']} C={r['C']}")
                pd.DataFrame(rows).to_csv(OUT / "established_tasks.csv", index=False)
    import torch, transformers, sklearn
    json.dump({"rows": rows, "published_source": "Ofli, Alam, Imran, ISCRAM 2020, arXiv:2004.11838, Tables 2 and 3",
               "splits": "crisismmd_datasplit_agreed_label.zip (published split; test sizes match Table 1)",
               "encoder": CLIP_MODEL, "head": "logistic regression on standardised features, C chosen on dev accuracy from {0.01,0.1,1,10}",
               "torch": torch.__version__, "transformers": transformers.__version__, "sklearn": sklearn.__version__,
               "date": time.strftime("%Y-%m-%d")}, open(OUT / "established_tasks.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--cache", required=True)
    a = ap.parse_args(); Path(a.cache).mkdir(parents=True, exist_ok=True)
    main(Path(a.src), Path(a.splits), Path(a.cache))
