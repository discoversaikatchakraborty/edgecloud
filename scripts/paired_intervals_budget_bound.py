"""Paired policy-against-fixed-rule error intervals, twenty seeds, per condition.

Written 2026-10-08 to close finding S2-016, which recorded that Section VI-A's
claim -- that the paired difference spans zero wherever the byte budget is the
only binding constraint -- had no twenty-seed paired interval behind it for the
conditions it names. It does now. The claim is confirmed, not changed.

Input : results/full_metrics_20seed_rev1.csv (per seed, per condition, per method)
Output: results/paired_budget_bound_20seed.csv
"""
import csv, collections, statistics as st
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from scipy import stats
    tcrit = lambda n: float(stats.t.ppf(0.975, n - 1))
except Exception:                       # keep the script runnable without scipy
    tcrit = lambda n: 2.093             # two-sided 95% at n = 20

rows = list(csv.DictReader(open(ROOT / "results" / "full_metrics_20seed_rev1.csv")))
by = collections.defaultdict(dict)
for r in rows:
    by[(r["scenario"], r["seed"])][r["method"]] = float(r["errors"])

out = []
for sc in sorted({s for s, _ in by}):
    d = [v["proposed"] - v["fixed_rule"] for (s, _), v in by.items()
         if s == sc and "proposed" in v and "fixed_rule" in v]
    if len(d) < 2:
        continue
    n, m = len(d), st.mean(d)
    h = tcrit(n) * st.stdev(d) / sqrt(n) if st.stdev(d) else 0.0
    out.append({"scenario": sc, "n": n, "paired_diff_proposed_minus_fixed_rule": round(m, 4),
                "ci_lo": round(m - h, 4), "ci_hi": round(m + h, 4),
                "spans_zero": (m - h) <= 0 <= (m + h)})

dest = ROOT / "results" / "paired_budget_bound_20seed.csv"
with open(dest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
print(f"[write] {dest.relative_to(ROOT)} ({len(out)} conditions)")
for r in out:
    print(f"  {r['scenario']:24} {r['paired_diff_proposed_minus_fixed_rule']:8.2f} "
          f"[{r['ci_lo']:7.2f}, {r['ci_hi']:7.2f}]  spans zero: {r['spans_zero']}")
