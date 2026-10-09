"""Scalar quantities the manuscript states in prose, derived and written to results/.

This package claims that every number in the manuscript is read from results/ at
build time. Several were not: they were correct, but typed into the prose by hand,
which is how the two errors corrected in version 2.6.5 survived as long as they did.
The ones that no existing result file exposed are derived here, as one tidy table of
named scalars, so the builder can read them like any other result.

Outputs: results/prose_diagnostics.csv with columns name, value, source.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
WAIT_THRESHOLD_S = 300.0


def main(log=print):
    rows = []

    # How often each policy waited longer than the threshold quoted in Section 6.2.
    rec = pd.read_csv(RES / "record_level_results.csv.gz")
    base = rec[rec.ablation.isna()] if rec.ablation.isna().any() else rec
    di = base[base.scenario == "delayed_imagery"]
    for m in ("proposed", "derived_threshold"):
        g = di[di.method == m]
        if len(g):
            rows.append({"name": f"wait_over_{int(WAIT_THRESHOLD_S)}s_rate_{m}",
                         "value": float(g.waited_for_image_s.gt(WAIT_THRESHOLD_S).mean()),
                         "source": "record_level_results.csv.gz"})

    # The severity estimator's training fit, reported as evidence that it is weak.
    man = json.loads((RES / "experiment_manifest.json").read_text())
    r2 = [s["severity_estimator_train_r2"] for s in man["splits"].values()]
    rows.append({"name": "severity_train_r2_mean", "value": float(np.mean(r2)),
                 "source": "experiment_manifest.json"})

    # The fifth-percentile throughput of the trace Section 6.9 singles out.
    prov = json.loads((ROOT / "data/traces/provenance.json").read_text())
    rows.append({"name": "verizon_driving_p5_mbps",
                 "value": float(prov["traces"]["Verizon-LTE-driving.up"]["per_second_mbps_p5"]),
                 "source": "data/traces/provenance.json"})

    # The calibration-event error rates that fixed the contested-budget difficulty.
    # These live in the frozen, digest-checked protocol rather than in a results file.
    proto = (ROOT / "PROTOCOL_contested_budget.md").read_text()
    m = re.search(r"^\|\s*1\.00\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%", proto, re.M)
    if m:
        rows.append({"name": "calibration_all_summary_error_rate", "value": float(m.group(1)) / 100,
                     "source": "PROTOCOL_contested_budget.md"})
        rows.append({"name": "calibration_fixed_rule_error_rate", "value": float(m.group(2)) / 100,
                     "source": "PROTOCOL_contested_budget.md"})

    out = pd.DataFrame(rows)
    out.to_csv(RES / "prose_diagnostics.csv", index=False)
    log(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
