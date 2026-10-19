"""Compare the rerun derived records with the package's frozen records.csv.gz."""
import hashlib, json, sys
import numpy as np, pandas as pd
PKG = "/Users/arlina/Downloads/all_projects/25_EDGECLOUD/upstream/prototype_v2.3/data/crisismmd"
NEW = sys.argv[1]
a = pd.read_csv(f"{PKG}/records.csv.gz"); b = pd.read_csv(f"{NEW}/records.csv.gz")
ha = hashlib.sha256(open(f"{PKG}/records.csv.gz","rb").read()).hexdigest(); hb = hashlib.sha256(open(f"{NEW}/records.csv.gz","rb").read()).hexdigest()
print("digest package", ha[:16], "| rerun", hb[:16], "| byte-identical:", ha == hb)
print("shape package", a.shape, "rerun", b.shape, "| same columns:", list(a.columns) == list(b.columns))
key = ["window_id", "image_id"]
m = a.merge(b, on=key, suffixes=("_a", "_b"), how="outer", indicator=True)
print("row alignment:", m["_merge"].value_counts().to_dict())
rep = []
for c in a.columns:
    if c in key: continue
    x, y = m[c + "_a"], m[c + "_b"]
    if np.issubdtype(a[c].dtype, np.number):
        d = (x - y).abs(); rel = d / np.maximum(x.abs(), 1e-12)
        rep.append((c, "numeric", int((d > 0).sum()), float(d.max()), float(rel.max()), int(np.isclose(x, y, rtol=1e-4, atol=1e-6).sum())))
    else:
        rep.append((c, "text", int((x.astype(str) != y.astype(str)).sum()), None, None, None))
print(f"{'column':18s} {'kind':8s} {'n_diff':>7s} {'max_abs':>12s} {'max_rel':>10s} {'n_close(1e-4)':>14s}")
for r in rep: print(f"{r[0]:18s} {r[1]:8s} {r[2]:7d} {('%.3g' % r[3]) if r[3] is not None else '':>12s} {('%.3g' % r[4]) if r[4] is not None else '':>10s} {str(r[5]) if r[5] is not None else '':>14s}")
exact = [c for c in a.columns if c not in key and ((m[c+'_a'].astype(str) == m[c+'_b'].astype(str)).all())]
print("columns byte-identical as text:", len(exact), "of", len(a.columns) - 2)
pa = json.load(open(f"{PKG}/provenance.json")); pb = json.load(open(f"{NEW}/provenance.json"))
for k in ("rows", "events", "class_counts", "per_event", "images_resolved"): print(k, "package", pa[k], "| rerun", pb[k], "| same:", pa[k] == pb[k])
