"""Does the real-data reversal survive a different split of the seven events?

Section 6.12 reports that the optimizer beats the fixed rule under two of three
conditions on CrisisMMD. That rests on one assignment of seven events to
train/calibration/test, and the test set is dominated by a single hurricane, so the
obvious referee question is whether the sign is a property of the data or of the
split. This runs the same comparison over a pre-specified set of rotations and
reports the sign of proposed minus fixed rule in each.

The rotations are fixed here before running and are not selected afterwards. Each
keeps the 3/2/2 shape of the original, because the misaligned condition needs at
least two events in calibration and in test.

Outputs: results/real_data_split_robustness.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import argparse, os
import core
from core import CONFIG, Scenario, fit_models, attach_probabilities, scenario_view, _run
from run_real_data import build_frame, scenario, METHODS

ROOT = Path(__file__).resolve().parent
SEEDS = CONFIG["seeds"]
CONDITIONS = ("normal", "low_bandwidth", "misaligned_evidence")

E = ["hurricane_harvey", "hurricane_maria", "california_wildfires",
     "mexico_earthquake", "srilanka_floods", "hurricane_irma", "iraq_iran_earthquake"]

# Pre-specified rotations. Rotation 0 is the split Section 6.12 reports; the rest
# rotate the event list so that every event serves in the test set at least once.
ROTATIONS = [tuple(E[(i + k) % 7] for i in range(7)) for k in range(7)]


def main(log=print):
    base = build_frame()
    rows = []
    for k, order in enumerate(ROTATIONS):
        assign = {e: ("train" if i < 3 else "calibration" if i < 5 else "test")
                  for i, e in enumerate(order)}
        d = base.copy()
        d["split"] = d.event_id.map(assign)
        train = d[d.split == "train"].drop(columns="split").reset_index(drop=True)
        cal = d[d.split == "calibration"].drop(columns="split").reset_index(drop=True)
        test = d[d.split == "test"].drop(columns="split").reset_index(drop=True)
        if len(cal) < 20 or len(test) < 20:
            log(f"rotation {k}: skipped, calibration {len(cal)} test {len(test)}")
            continue
        log(f"rotation {k}: train {len(train)} calibration {len(cal)} test {len(test)} "
            f"[test = {sorted(set(test.event_id))}]")
        for seed in SEEDS:
            models = fit_models(train, cal, seed)
            c, t = attach_probabilities(cal, models), attach_probabilities(test, models)
            for cond in CONDITIONS:
                sc = scenario(cond)
                tv, cv = scenario_view(t, sc, seed, models), scenario_view(c, sc, seed + 7, models)
                for m in METHODS:
                    res, met, _ = _run(tv, cv, m, sc, seed, models)
                    rows.append({"rotation": k, "seed": seed, "scenario": cond, "method": m,
                                 "n_test": len(test), "errors": met["errors"],
                                 "test_events": "|".join(sorted(set(test.event_id)))})
    out = pd.DataFrame(rows)
    tag = "pc_on" if core.USE_PRED_CONTRIB else "pc_off"
    (ROOT / "results_v27").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results_v27" / f"real_data_split_robustness_{tag}.csv", index=False)

    g = (out.groupby(["rotation", "scenario", "method"]).errors.mean().reset_index()
            .pivot_table(index=["rotation", "scenario"], columns="method", values="errors"))
    g["proposed_minus_fixed_rule"] = g["proposed"] - g["fixed_rule"]
    n = out.groupby("rotation").n_test.first()
    log("")
    log("proposed minus fixed rule, by rotation and condition (negative = optimizer wins)")
    for (k, cond), r in g.iterrows():
        log(f"  rotation {k} n={n[k]:4d}  {cond:20s} {r['proposed_minus_fixed_rule']:+7.1f}")
    return out


if __name__ == "__main__":
    core.USE_PRED_CONTRIB = os.environ.get("PC", "off") == "on"
    main()
