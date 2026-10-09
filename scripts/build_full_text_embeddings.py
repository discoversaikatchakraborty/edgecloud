"""Recompute the full-dimension CLIP text embedding for the CrisisMMD workload.

Item A compares the text pathway at 8 PCA components, at 32 and at full dimension.
The derived table shipped with the prototype carries text_x0..text_x7 only, which is
already a projection to eight dimensions and is not invertible, so the full embedding
has to be recomputed rather than recovered.

What this script does and nothing else:
  1. reads upstream/prototype_v2.3/data/crisismmd/records.csv.gz, the 3,526 derived
     records, which carry image_id but not the tweet text;
  2. recovers the tweet text by joining image_id against the CrisisMMD v2.0 annotation
     tables, which carry both;
  3. encodes the text with openai/clip-vit-base-patch32, the encoder named in
     data/crisismmd/provenance.json, through the same call sequence as the text branch
     of prepare_crisismmd.py encode();
  4. checks the recomputation by fitting PCA to eight components on the training events
     and comparing the result against the shipped text_x0..text_x7 component by
     component, up to the sign that PCA leaves free;
  5. writes data/crisismmd_text_full.npz, keyed by image_id, with a digest manifest.

Nothing under upstream/ is written. Run once; later runs reuse the cache.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, "upstream", "prototype_v2.3", "data", "crisismmd", "records.csv.gz")
ANNOTATIONS = "/Users/arlina/Downloads/all_projects/Tier-Ready/12_edgewait/data_real/ex/CrisisMMD_v2.0/annotations"
OUT = os.path.join(ROOT, "data", "crisismmd_text_full.npz")
MANIFEST = os.path.join(ROOT, "data", "crisismmd_text_full_manifest.json")
CLIP_MODEL = "openai/clip-vit-base-patch32"
TRAIN_EVENTS = ("hurricane_harvey", "hurricane_maria", "california_wildfires")


def recover_text() -> pd.DataFrame:
    d = pd.read_csv(RECORDS)
    files = sorted(glob.glob(os.path.join(ANNOTATIONS, "*_final_data.tsv")))
    if not files:
        sys.exit(f"no CrisisMMD annotation tables under {ANNOTATIONS}")
    ann = pd.concat([pd.read_csv(f, sep="\t", dtype=str)[["image_id", "tweet_text"]]
                     for f in files], ignore_index=True).drop_duplicates("image_id")
    m = d.merge(ann, on="image_id", how="left", validate="one_to_one")
    missing = int(m.tweet_text.isna().sum())
    if missing:
        sys.exit(f"{missing} of {len(m)} records have no recoverable text; the join is incomplete")
    print(f"[text] recovered {len(m)} tweet texts for {len(m)} records by image_id")
    return m


def encode_text(texts):
    """The text branch of prepare_crisismmd.py encode(), unchanged."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[clip] {CLIP_MODEL} on {dev}")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = [str(x)[:300] for x in texts[i:i + 64]]
            inp = proc(text=b, return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(dev)
            e = model.get_text_features(**inp)
            out.append(getattr(e, "pooler_output", e).cpu().numpy())
    X = np.concatenate(out).astype("float64")
    print(f"[clip] text embedding {X.shape}")
    return X


def verify_against_shipped(X, d):
    """Fit PCA to eight components on the training events and compare with text_x0..7.

    PCA fixes each component only up to sign, so the comparison is on absolute
    correlation per component. A faithful recomputation gives correlations at 1.0.
    """
    from sklearn.decomposition import PCA
    train = d.event_id.isin(TRAIN_EVENTS).to_numpy()
    p = PCA(n_components=8, random_state=0).fit(X[train])
    Z = p.transform(X)
    shipped = d[[f"text_x{i}" for i in range(8)]].to_numpy(float)
    cors = [abs(float(np.corrcoef(Z[:, i], shipped[:, i])[0, 1])) for i in range(8)]
    print("[verify] |corr| of recomputed PCA-8 against shipped text_x0..7: "
          + " ".join(f"{c:.4f}" for c in cors))
    print(f"[verify] retained variance {p.explained_variance_ratio_.sum():.1%}")
    return cors, float(p.explained_variance_ratio_.sum())


def main():
    if os.path.exists(OUT):
        print(f"[cache] {OUT} already exists; nothing to do")
        return
    d = recover_text()
    X = encode_text(d.tweet_text.values)
    cors, var = verify_against_shipped(X, d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, image_id=d.image_id.to_numpy().astype(str), X=X)
    man = {
        "encoder": CLIP_MODEL,
        "source_records": os.path.relpath(RECORDS, ROOT),
        "source_records_sha256": hashlib.sha256(open(RECORDS, "rb").read()).hexdigest(),
        "annotations_dir": ANNOTATIONS,
        "rows": int(X.shape[0]),
        "dimension": int(X.shape[1]),
        "text_truncation": "str(text)[:300] then CLIP truncation at max_length=77",
        "pca8_check_abs_correlation_with_shipped": cors,
        "pca8_retained_variance_train_fitted": var,
        "output_sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest(),
    }
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
    print(f"[write] {OUT}")
    print(f"[write] {MANIFEST}")


if __name__ == "__main__":
    main()
