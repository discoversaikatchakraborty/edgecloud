"""Item C: real-data budgeting comparison with item A's class handling applied.

Re-runs the revised manuscript's Table XIII (three conditions, rotation 0) and
Table XIV (proposed minus fixed rule across rotations 0..6) with the class
handling chosen by item A, and writes both side by side with the current
(unweighted) numbers. Nothing is replaced silently.

Requires results/itemA_decision.json from item A.

Outputs:
  results/crisismmd_budgeting_rev2.csv
  results/crisismmd_rotations_rev2.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, mean_ci, run_manifest, write_csv  # noqa: E402


def main():
    proto = load_protocol("itemC_protocol_rev2.json")
    run_manifest(os.path.join(RESULTS, "itemC_manifest_rev2.json"), proto)
    # item A's decision file carries the _rev1 suffix, as every rerun output does
    dec = json.load(open(os.path.join(RESULTS, "itemA_decision_rev1.json")))
    ch = dec["best_cell"]["class_handling"]
    assert ch == proto["class_handling"], (
        f"frozen protocol says {proto['class_handling']!r} but item A says {ch!r}")
    # the handling NAME is passed through, not a scikit-learn class_weight value:
    # resampling is an operation on the training rows and has no class_weight spelling
    cw = ch
    print(f"[item C] class handling from item A: {ch}")

    rows, rot_rows = [], []
    for rot in range(7):
        split = adapters.load_crisismmd_rotation(rot)
        for cond in proto["conditions"]:
            per_seed = {"current": [], "handled": []}
            for seed in proto["seeds"]:
                cur = adapters.run_real_budgeting(split, cond, seed, proto["methods"], class_weight="none")
                new = adapters.run_real_budgeting(split, cond, seed, proto["methods"], class_weight=cw)
                per_seed["current"].append(cur)
                per_seed["handled"].append(new)
                if rot == 0:
                    for variant, res in (("current", cur), ("handled", new)):
                        for m, v in res.items():
                            rows.append({"condition": cond, "variant": variant, "method": m, "seed": seed, **v})
            for variant in ("current", "handled"):
                diffs = [r["Proposed (exact knapsack)"]["errors"] - r["Fixed rule"]["errors"]
                         for r in per_seed[variant]]
                mean, lo, hi = mean_ci(diffs)
                rot_rows.append({"rotation": rot, "condition": cond, "variant": variant,
                                 "n_test": int(split.mask("test").sum()),
                                 "proposed_minus_fixed": mean, "lo": lo, "hi": hi})
    write_csv(os.path.join(RESULTS, "crisismmd_budgeting_rev2.csv"), rows)
    write_csv(os.path.join(RESULTS, "crisismmd_rotations_rev2.csv"), rot_rows)

    # reproduction check against the paper's rotation-0 current numbers
    ref = proto["reference_rotation0_current"]
    cur0 = [r for r in rows if r["variant"] == "current" and r["condition"] == "normal"]
    for m, expected in ref.items():
        got = np.mean([r["errors"] for r in cur0 if r["method"] == m])
        flag = "OK" if abs(got - expected) < 0.05 else "MISMATCH"
        print(f"[repro] {m:28s} expected {expected:7.1f} got {got:7.1f} {flag}")
    wins = sum(1 for r in rot_rows if r["variant"] == "handled" and r["proposed_minus_fixed"] < 0)
    print(f"[item C] handled variant: optimizer ahead of fixed rule in {wins} of 21 comparisons "
          f"(paper, current: 15 of 21)")
    print("MANUSCRIPT: Tables XIII and XIV, Section VIII-C first two paragraphs, abstract '15 of 21'.")


if __name__ == "__main__":
    main()
