"""Table III reproduction: frozen package results against the rerun on re-derived records."""
import pandas as pd, numpy as np, sys
S = sys.argv[1]
a = pd.read_csv(f"{S}/results_frozen/real_data_summary.csv"); b = pd.read_csv(f"{S}/results/real_data_summary.csv")
m = a.merge(b, on=["scenario", "method"], suffixes=("_pkg", "_new"))
cols = ["errors_mean", "mean_bytes_mean", "p95_latency_s_mean"]
print(f"{'scenario':12s} {'method':26s} " + " ".join(f"{c[:-5]+'(pkg/new)':>28s}" for c in cols))
worst = 0
for _, r in m.iterrows():
    cells = []
    for c in cols:
        d = abs(r[c+'_pkg'] - r[c+'_new']); worst = max(worst, d / max(abs(r[c+'_pkg']), 1e-9) if c != 'errors_mean' else 0)
        cells.append(f"{r[c+'_pkg']:12.3f}/{r[c+'_new']:<12.3f}" if c != 'errors_mean' else f"{r[c+'_pkg']:12.1f}/{r[c+'_new']:<12.1f}")
    print(f"{r.scenario:12s} {r.method:26s} " + " ".join(f"{x:>28s}" for x in cells))
e = (m.errors_mean_pkg - m.errors_mean_new)
print("\nerrors: rows identical", int((e == 0).sum()), "of", len(m), "| max |diff| %.1f errors" % e.abs().max(), "| out of 968")
# rankings per scenario
for sc, g in m.groupby("scenario"):
    ra = g.sort_values("errors_mean_pkg").method.tolist(); rb = g.sort_values("errors_mean_new").method.tolist()
    print(f"ranking by errors identical under {sc}: {ra == rb}")
# paired differences used in the text (optimizer vs fixed rule)
pa = pd.read_csv(f"{S}/results_frozen/real_data_paired.csv"); pb = pd.read_csv(f"{S}/results/real_data_paired.csv")
print("\npaired file columns:", list(pa.columns)[:8])
key = [c for c in pa.columns if pa[c].dtype == object][:3]
mm = pa.merge(pb, on=key, suffixes=("_pkg", "_new"))
num = [c for c in pa.columns if c not in key and np.issubdtype(pa[c].dtype, np.number)]
print("paired rows", len(mm), "| max |diff| across numeric columns:", {c: round(float((mm[c+'_pkg']-mm[c+'_new']).abs().max()), 3) for c in num[:6]})
