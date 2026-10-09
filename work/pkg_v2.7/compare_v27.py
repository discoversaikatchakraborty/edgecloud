"""Reductions of the v2.7 runs for the record and the manuscript.

1. Twenty-seed synthetic regression: run_all(pc off) seed-level errors/bytes/p95 against the paper's
   results/main_20seed_rev1.csv (condition, method, seed, errors, bytes, p95).
2. Twenty-seed synthetic, term on vs off: proposed-policy errors per condition.
3. Rotations: proposed minus fixed rule by rotation and condition, term off vs on; counts of 21.
Outputs: results_v27/summary_*.csv and printed tables.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; R = HERE / "results_v27"; PROJ = HERE.parents[1]
what = sys.argv[1]
if what == "synthetic":
    paper = pd.read_csv(PROJ / "results" / "main_20seed_rev1.csv")
    off = pd.read_csv(R / "synthetic20_pc_off" / "seed_level_results.csv") if (R / "synthetic20_pc_off" / "seed_level_results.csv").exists() else None
    if off is None:
        import glob; print("files:", [Path(f).name for f in glob.glob(str(R / "synthetic20_pc_off" / "*"))]); sys.exit()
    m = paper.merge(off, left_on=["condition", "method", "seed"], right_on=["scenario", "method", "seed"], how="inner")
    print("regression rows matched", len(m), "of", len(paper))
    for a, b in (("errors", "errors"), ("bytes", "mean_bytes"), ("p95", "p95_latency_s")):
        print(f"  {a:7s} max |diff| = {float((m[a+'_x' if a==b else a] - m[b+'_y' if a==b else b]).abs().max()):.6g}")
    on = pd.read_csv(R / "synthetic20_pc_on" / "seed_level_results.csv") if (R / "synthetic20_pc_on" / "seed_level_results.csv").exists() else None
    if on is not None:
        po = off[off.method == "proposed"].groupby("scenario").errors.mean(); pn = on[on.method == "proposed"].groupby("scenario").errors.mean()
        fx = off[off.method == "fixed_rule"].groupby("scenario").errors.mean()
        t = pd.DataFrame({"proposed_off": po, "proposed_on": pn, "fixed_rule": fx}).round(2); t["on_minus_off"] = (t.proposed_on - t.proposed_off).round(2)
        print("\ntwenty-seed synthetic, proposed errors per condition, term off vs on:\n", t.to_string()); t.to_csv(R / "summary_synthetic_pc_on_vs_off.csv")
        # per-modality contribution values are not stored by run_all; recompute from one seed
elif what == "rotations":
    rows = []
    for pc in ("off", "on"):
        f = R / f"real_data_split_robustness_pc_{pc}.csv"
        if not f.exists(): print("missing", f.name); continue
        d = pd.read_csv(f)
        print(f"columns ({pc}):", list(d.columns)[:12])
        key = [c for c in ("rotation", "scenario", "condition") if c in d.columns]
        g = d.groupby(key + ["method"]).errors.mean().unstack()
        g["proposed_minus_fixed"] = g["proposed"] - g["fixed_rule"]; g["proposed_minus_image_only"] = g["proposed"] - g["image_only"]
        g["pc"] = pc; rows.append(g.reset_index())
    t = pd.concat(rows); t.to_csv(R / "summary_rotations_pc_on_vs_off.csv", index=False)
    for pc in ("off", "on"):
        s = t[t.pc == pc]
        print(f"\npc {pc}: optimizer ahead of fixed rule in {int((s.proposed_minus_fixed < 0).sum())} of {len(s)}; ahead of image-only in {int((s.proposed_minus_image_only < 0).sum())} of {len(s)}")
        print(s.pivot_table(index=[c for c in ("rotation",) if c in s], columns=[c for c in ("scenario", "condition") if c in s][0], values="proposed_minus_fixed").round(1).to_string())
