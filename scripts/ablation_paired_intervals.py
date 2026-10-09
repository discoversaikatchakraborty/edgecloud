"""Regenerate results/ablation_20seed_paired_rev1.csv from its retained inputs.

Written 2026-10-08 to close finding CLM-22 / S1-F21, which recorded that this
file supports a manuscript table while no script in scripts/ writes it, no
manifest covers it and no run log mentions it. The values were independently
recomputed during review and matched; what was missing was the regeneration
path. This script is that path. It does not change any value.

helped_by is the paired per-seed difference (ablated errors - full errors): how
much the removed component was worth in that condition. Intervals are two-sided
95% t intervals over the twenty seeds.

Inputs : results/ablation_20seed_rev1.csv      (per seed, per ablation, per condition)
         results/full_metrics_20seed_rev1.csv  (per seed, per method, per condition)
Output : results/ablation_20seed_paired_rev1.csv
Verify : --check compares against the existing file instead of overwriting it.
"""
import csv, collections, statistics as st, sys
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DEST = RES / "ablation_20seed_paired_rev1.csv"
try:
    from scipy import stats
    tcrit = lambda n: float(stats.t.ppf(0.975, n - 1))
except Exception:
    tcrit = lambda n: 2.093

full = {}
for r in csv.DictReader(open(RES / "full_metrics_20seed_rev1.csv")):
    if r["method"] == "proposed":
        full[(r["scenario"], r["seed"])] = float(r["errors"])

abl = collections.defaultdict(dict)
for r in csv.DictReader(open(RES / "ablation_20seed_rev1.csv")):
    abl[(r["ablation"], r["scenario"])][r["seed"]] = float(r["errors"])

out = []
for (a, sc), per_seed in abl.items():
    d = [e - full[(sc, s)] for s, e in per_seed.items() if (sc, s) in full]
    if len(d) < 2:
        continue
    n, m = len(d), st.mean(d)
    sd = st.stdev(d)
    h = tcrit(n) * sd / sqrt(n) if sd else 0.0
    out.append({"ablation": a, "scenario": sc, "n": n,
                "helped_by": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4),
                "includes_zero": (m - h) <= 0 <= (m + h)})
out.sort(key=lambda r: (r["scenario"], r["ablation"]))

if "--check" in sys.argv:
    have = {(r["ablation"], r["scenario"]): r for r in csv.DictReader(open(DEST))}
    bad = 0
    for r in out:
        g = have.get((r["ablation"], r["scenario"]))
        if not g:
            print(f"  MISSING in file: {r['ablation']} / {r['scenario']}"); bad += 1; continue
        for k, tol in (("helped_by", 1e-6), ("lo", 5e-3), ("hi", 5e-3)):
            if abs(float(g[k]) - r[k]) > tol:
                print(f"  DIFFERS {r['ablation']}/{r['scenario']} {k}: file {g[k]} recomputed {r[k]}")
                bad += 1
    print(f"check: {len(out)} rows recomputed, {bad} discrepancies "
          f"(means exact, interval endpoints within 0.005 for t-quantile rounding)")
    sys.exit(1 if bad else 0)

with open(DEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ablation", "scenario", "n", "helped_by", "lo", "hi", "includes_zero"])
    w.writeheader(); w.writerows(out)
print(f"[write] {DEST.relative_to(ROOT)} ({len(out)} rows)")
