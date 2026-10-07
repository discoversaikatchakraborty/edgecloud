"""Build a real-data workload from CrisisMMD v2.0.

The benchmark elsewhere in this package is synthetic, and every conclusion drawn
from it is conditional on that. This script replaces the generator with real
multimodal disaster data so the same policies can be measured on real content,
real event identity, real hazard types and real arrival times.

Source: CrisisMMD v2.0, Alam, Ofli and Imran, ICWSM 2018. Downloaded from
https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz. The archive is not
redistributed here; this script reads an extracted copy and writes derived records.

What is real: the text of each report, the event it belongs to, the hazard type,
the arrival time (recovered from the tweet identifier), the three-class damage
label, and the annotator agreement used as an observable quality signal.

What is not available and is therefore not modelled: there is no sensor modality
in CrisisMMD, so the workload has two modalities rather than three; there is no
geolocation in the annotations, so spatial admissibility degrades to event
identity; and there is no severity field independent of the damage label, so
severity is held constant rather than invented, which makes severity-dependent
behaviour inert on this workload.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "crisismmd"

CLASSES = {"severe_damage": 0, "mild_damage": 1, "little_or_no_damage": 2}
HAZARD = {"california_wildfires": "fire", "hurricane_harvey": "hurricane",
          "hurricane_irma": "hurricane", "hurricane_maria": "hurricane",
          "iraq_iran_earthquake": "earthquake", "mexico_earthquake": "earthquake",
          "srilanka_floods": "flood"}
# Held out by event, as in the synthetic protocol: no test event contributes to any fitted quantity.
# Two test events are required, not optional: the misaligned-evidence condition
# replaces a report with one from a different event, so a single-event test set
# cannot express it. The two chosen span different hazards, which makes the
# misalignment cross-hazard rather than merely cross-incident.
# Every split needs at least two events, not just the test split: the
# misaligned-evidence condition draws a foreign report from within the split it is
# applied to, and calibration receives a scenario view as well.
SPLIT = {"train": ["hurricane_harvey", "hurricane_maria", "california_wildfires"],
         "calibration": ["mexico_earthquake", "srilanka_floods"],
         "test": ["hurricane_irma", "iraq_iran_earthquake"]}
TEXT_DIMS = 8           # matches FULL_DIMS["text"] in core.py
IMAGE_DIMS = 10         # matches FULL_DIMS["image"]
SNOWFLAKE_EPOCH_MS = 1288834974657


def load_annotations(src: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((src / "annotations").glob("*_final_data.tsv")):
        event = f.name.replace("_final_data.tsv", "")
        t = pd.read_csv(f, sep="\t", dtype=str)
        t["event"] = event
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.image_damage.isin(CLASSES)].copy()
    d["y"] = d.image_damage.map(CLASSES).astype(int)
    d["hazard"] = d.event.map(HAZARD)
    ids = np.asarray([int(x) for x in d.tweet_id], dtype=np.int64)
    d["obs_time_s"] = (np.right_shift(ids, 22) + SNOWFLAKE_EPOCH_MS) / 1000.0
    # Annotator agreement is an observable quality signal. It is not the label.
    for src_col, dst in (("image_damage_conf", "image_quality"), ("text_info_conf", "text_quality")):
        d[dst] = pd.to_numeric(d[src_col], errors="coerce").fillna(0.6).clip(0.15, 1.0)
    d = d.sort_values(["event", "obs_time_s"]).reset_index(drop=True)
    return d


CLIP_MODEL = "openai/clip-vit-base-patch32"


def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, proc, dev, torch


def _emb(out):
    """transformers 5.x returns an output object; earlier versions return a tensor."""
    return getattr(out, "pooler_output", out)


def encode(texts, image_paths, src: Path):
    """CLIP embeddings for both modalities.

    A pretrained encoder is used rather than hand-built descriptors. The first
    attempt at this workload used TF-IDF for text and pixel statistics for images,
    and both were too weak to carry damage severity: the classifiers predicted the
    majority class even on training data. That failure is reported in the
    manuscript, because it is the reason this representation was chosen.
    """
    model, proc, dev, torch = _clip()
    T, I, sizes, ok = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = [str(x)[:300] for x in texts[i:i + 64]]
            inp = proc(text=b, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            T.append(_emb(model.get_text_features(**inp)).cpu().numpy())
        from PIL import Image
        batch, idx = [], []
        for j, rel in enumerate(image_paths):
            f = src / rel
            try:
                sizes.append(f.stat().st_size)
                batch.append(Image.open(f).convert("RGB")); idx.append(j); ok.append(True)
            except Exception:
                sizes.append(0); ok.append(False)
            if len(batch) == 64 or (j == len(image_paths) - 1 and batch):
                inp = proc(images=batch, return_tensors="pt").to(dev)
                I.append(_emb(model.get_image_features(**inp)).cpu().numpy())
                batch = []
    T = np.concatenate(T)
    Ifeat = np.concatenate(I) if I else np.zeros((0, 512))
    ok = np.asarray(ok)
    full = np.zeros((len(image_paths), Ifeat.shape[1]))
    full[ok] = Ifeat
    return T.astype("float64"), full.astype("float64"), np.asarray(sizes, dtype="int64"), ok


def reduce_to(X, mask, dims, seed=0):
    p = PCA(n_components=dims, random_state=seed)
    p.fit(X[mask])
    return p.transform(X), float(p.explained_variance_ratio_.sum())


def main(src: Path):
    d = load_annotations(src)
    split_of = {e: k for k, v in SPLIT.items() for e in v}
    d["split"] = d.event.map(split_of)
    train_mask = (d.split == "train").to_numpy()
    TX, IX, img_bytes, img_ok = encode(d.tweet_text.values, d.image_path.values, src)
    T, tvar = reduce_to(TX, train_mask, TEXT_DIMS)
    I, ivar = reduce_to(IX, train_mask & img_ok, IMAGE_DIMS)
    text_bytes = np.asarray([len(str(t).encode("utf-8")) for t in d.tweet_text], dtype="int64")
    print(f"CLIP embeddings reduced by PCA fitted on training events: "
          f"text {TEXT_DIMS} dims retain {tvar:.1%} variance, image {IMAGE_DIMS} dims retain {ivar:.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame({
        "window_id": d.tweet_id.values, "event_id": d.event.values, "split": d.split.values,
        "y": d.y.values, "hazard": d.hazard.values, "obs_time_s": d.obs_time_s.values,
        "image_id": d.image_id.values, "image_path": d.image_path.values,
        "text_quality": d.text_quality.values, "image_quality": d.image_quality.values,
        "text_raw_bytes": text_bytes, "image_raw_bytes": img_bytes, "image_available": img_ok,
    })
    for i in range(TEXT_DIMS):
        rec[f"text_x{i}"] = T[:, i]
    for i in range(IMAGE_DIMS):
        rec[f"image_x{i}"] = I[:, i]
    rec.to_csv(OUT / "records.csv.gz", index=False, float_format="%.6g", compression="gzip")

    prov = {
        "dataset": "CrisisMMD v2.0",
        "cite": "F. Alam, F. Ofli, M. Imran, CrisisMMD: multimodal Twitter datasets from natural "
                "disasters, in: Proc. 12th Int. AAAI Conf. Web and Social Media, 2018.",
        "source_url": "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz",
        "redistributed": False,
        "label": "image_damage, three classes: severe_damage, mild_damage, little_or_no_damage",
        "rows": int(len(rec)), "events": int(rec.event_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in rec.y.value_counts().sort_index().items()},
        "per_event": {e: int(n) for e, n in rec.event_id.value_counts().sort_index().items()},
        "split": SPLIT,
        "encoder": CLIP_MODEL,
        "text_representation": f"CLIP text embedding reduced by PCA to {TEXT_DIMS} components, "
                              f"PCA fitted on training events only",
        "arrival_times": "recovered from the Twitter snowflake identifier; real, not synthesised",
        "image_representation": f"CLIP image embedding reduced by PCA to {IMAGE_DIMS} components, "
                               f"PCA fitted on training events only",
        "payload_sizes": "real: image byte sizes are the files on disk, text byte sizes are the "
                         "UTF-8 length of the tweet. Neither is assumed.",
        "images_resolved": int(img_ok.sum()),
        "not_modelled": ["no sensor modality exists in CrisisMMD; the workload has two modalities",
                         "no geolocation in the annotations; spatial admissibility degrades to event identity",
                         "no severity field independent of the label; severity is held constant"],
        "records_sha256": hashlib.sha256((OUT / "records.csv.gz").read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"rows {len(rec)}  events {rec.event_id.nunique()}  classes {prov['class_counts']}")
    for k, v in SPLIT.items():
        print(f"  {k:12} {sum(rec.event_id.isin(v)):5} windows  {v}")
    print("wrote", (OUT / "records.csv.gz").relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted CrisisMMD_v2.0 directory")
    main(Path(ap.parse_args().src))
