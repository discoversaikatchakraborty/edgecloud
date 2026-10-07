"""Item E: 20 seeds on the five main synthetic conditions (Reviewer 2.4).

Step 1 reproduces the published five-seed means for the Proposed and Fixed
rule methods from the revised manuscript's Table XV (seeds 0..4) and stops on a
mismatch. Step 2 runs seeds 0..19 and recomputes every paired interval that
Sections VI-A to VI-D report.

Outputs:
  results/main_20seed_rev1.csv     per method, condition, seed
  results/paired_20seed_rev1.csv   paired differences with 95% t intervals
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, mean_ci, paired_t, run_manifest, write_csv  # noqa: E402


def main():
    proto = load_protocol("itemE_protocol.json")
    run_manifest(os.path.join(RESULTS, "itemE_manifest.json"), proto)
    rows, paired = [], []
    for cond, spec in proto["conditions"].items():
        res = adapters.run_synthetic_condition(cond, proto["seeds"], spec["methods"])
        # reproduction gate on the first five seeds
        for m, expected in spec["reference_5seed_errors"].items():
            got = np.mean([res[m][s]["errors"] for s in proto["seeds"][:5]])
            if abs(got - expected) > 0.05:
                sys.exit(f"REPRODUCTION FAILED {cond}/{m}: expected {expected} got {got:.2f}. "
                         "The runner is not the code path that produced the paper.")
            print(f"[repro] {cond:16s} {m:24s} {got:6.1f} OK")
        for m in spec["methods"]:
            for s in proto["seeds"]:
                rows.append({"condition": cond, "method": m, "seed": s, **res[m][s]})
            mean, lo, hi = mean_ci([res[m][s]["errors"] for s in proto["seeds"]])
            print(f"  {cond:16s} {m:24s} errors {mean:6.1f} [{lo:.1f}, {hi:.1f}] n={len(proto['seeds'])}")
        for a, b in spec["paired"]:
            for metric in ("errors", "bytes", "p95"):
                pt = paired_t([res[a][s][metric] for s in proto["seeds"]],
                              [res[b][s][metric] for s in proto["seeds"]])
                paired.append({"condition": cond, "a": a, "b": b, "metric": metric,
                               "diff": pt.mean, "lo": pt.lo, "hi": pt.hi, "p": pt.p, "n": pt.n,
                               "includes_zero": bool(pt.lo <= 0 <= pt.hi)})
    write_csv(os.path.join(RESULTS, "main_20seed_rev1.csv"), rows)
    write_csv(os.path.join(RESULTS, "paired_20seed_rev1.csv"), paired)
    print("MANUSCRIPT: every interval in Sections VI-A to VI-D, Tables V, VI and XVI, "
          "Section V-C 'with five seeds several include zero', Section X.")


if __name__ == "__main__":
    main()
