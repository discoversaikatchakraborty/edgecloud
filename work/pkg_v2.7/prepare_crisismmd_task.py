"""v2.7: build a budgeting workload from a PUBLISHED CrisisMMD task on its published split.

Same derivation as prepare_crisismmd.py (CLIP ViT-B/32, PCA 8 text / 10 image fitted on the
training split, real byte sizes, annotator agreement as quality, snowflake arrival times),
but the label is the task's own and the split is the agreed-label split of Ofli, Alam and
Imran (2020): train -> train, dev -> calibration, test -> test. Written for the informative
(two-class) and humanitarian (five-class) tasks. Embeddings come from a cache written by
scripts/crisismmd_established_tasks.py; images and texts are never redistributed.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
HAZARD = {"california_wildfires": "fire", "hurricane_harvey": "hurricane", "hurricane_irma": "hurricane",
          "hurricane_maria": "hurricane", "iraq_iran_earthquake": "earthquake", "mexico_earthquake": "earthquake",
          "srilanka_floods": "flood"}
SNOWFLAKE_EPOCH_MS = 1288834974657
TEXT_DIMS, IMAGE_DIMS = 8, 10
CONF = {"informative": ("text_info_conf", "image_info_conf"), "humanitarian": ("text_human_conf", "image_human_conf")}


def main(task, splits, src, cache):
    parts = []
    for part, name in (("train", "train"), ("dev", "calibration"), ("test", "test")):
        d = pd.read_csv(splits / f"task_{task}_text_img_agreed_lab_{part}.tsv", sep="\t", dtype=str); d["split"] = name; parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    ann = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in sorted((src / "annotations").glob("*_final_data.tsv"))])
    ann = ann.drop_duplicates("image_id").set_index("image_id")
    tconf, iconf = CONF[task]
    d["text_quality"] = pd.to_numeric(ann.loc[d.image_id, tconf].values, errors="coerce")
    d["image_quality"] = pd.to_numeric(ann.loc[d.image_id, iconf].values, errors="coerce")
    for c in ("text_quality", "image_quality"): d[c] = d[c].fillna(0.6).clip(0.15, 1.0)
    classes = sorted(d.label.unique()); d["y"] = d.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    d["hazard"] = d.event_name.map(HAZARD)
    ids = np.asarray([int(x) for x in d.tweet_id], dtype=np.int64); d["obs_time_s"] = (np.right_shift(ids, 22) + SNOWFLAKE_EPOCH_MS) / 1000.0
    zt = np.load(cache / "text.npz", allow_pickle=True); T = dict(zip(zt["keys"], zt["vals"]))
    zi = np.load(cache / "image.npz", allow_pickle=True); I = dict(zip(zi["keys"], zi["vals"]))
    XT = np.stack([T[str(t)] for t in d.tweet_text]); XI = np.stack([I[str(p)] for p in d.image])
    tr = (d.split == "train").to_numpy()
    pt = PCA(TEXT_DIMS, random_state=0).fit(XT[tr]); pi = PCA(IMAGE_DIMS, random_state=0).fit(XI[tr])
    RT, RI = pt.transform(XT), pi.transform(XI)
    img_bytes = np.asarray([(src / p).stat().st_size for p in d.image], dtype="int64")
    text_bytes = np.asarray([len(str(t).encode("utf-8")) for t in d.tweet_text], dtype="int64")
    d = d.sort_values(["event_name", "obs_time_s"]).reset_index(drop=True)
    order = d.index.to_numpy()  # already sorted; recompute arrays in the same order
    # (re-derive arrays in sorted order)
    XT = np.stack([T[str(t)] for t in d.tweet_text]); XI = np.stack([I[str(p)] for p in d.image])
    RT, RI = pt.transform(XT), pi.transform(XI)
    img_bytes = np.asarray([(src / p).stat().st_size for p in d.image], dtype="int64")
    text_bytes = np.asarray([len(str(t).encode("utf-8")) for t in d.tweet_text], dtype="int64")
    rec = pd.DataFrame({"window_id": d.image_id.values, "event_id": d.event_name.values, "split": d.split.values, "y": d.y.values,
                        "hazard": d.hazard.values, "obs_time_s": d.obs_time_s.values, "image_id": d.image_id.values,
                        "image_path": d.image.values, "text_quality": d.text_quality.values, "image_quality": d.image_quality.values,
                        "text_raw_bytes": text_bytes, "image_raw_bytes": img_bytes, "image_available": True})
    for i in range(TEXT_DIMS): rec[f"text_x{i}"] = RT[:, i]
    for i in range(IMAGE_DIMS): rec[f"image_x{i}"] = RI[:, i]
    out = ROOT / "data" / f"crisismmd_{task}"; out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out / "records.csv.gz", index=False, float_format="%.6g", compression="gzip")
    prov = {"task": task, "classes": classes, "rows": int(len(rec)), "split_rows": rec.split.value_counts().to_dict(),
            "class_counts_test": rec[rec.split == "test"].y.value_counts().sort_index().to_dict(),
            "split_source": "crisismmd_datasplit_agreed_label.zip (Ofli, Alam, Imran 2020)", "encoder": "openai/clip-vit-base-patch32",
            "pca": f"text {TEXT_DIMS}, image {IMAGE_DIMS}, fitted on the training split; text retains {pt.explained_variance_ratio_.sum():.3f}, image {pi.explained_variance_ratio_.sum():.3f}",
            "records_sha256": hashlib.sha256((out / "records.csv.gz").read_bytes()).hexdigest()}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(task, "rows", len(rec), rec.split.value_counts().to_dict(), "classes", classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--src", required=True); ap.add_argument("--cache", required=True); a = ap.parse_args()
    main(a.task, Path(a.splits), Path(a.src), Path(a.cache))
