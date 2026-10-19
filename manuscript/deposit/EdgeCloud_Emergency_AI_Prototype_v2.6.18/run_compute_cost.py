"""Measured edge-side decision cost (exploratory).

The complexity of the slot allocator is stated analytically in the method section
but was never measured. This script times the real thing on real inputs: the
allocation step for each method over the scenario views the benchmark actually
produces, and the exact knapsack alone as the number of windows sharing a slot
grows. Inputs are the genuine utility, byte and feasibility matrices from the
contested-budget condition; only how many of them are handed to the solver varies.

This measures wall-clock time and Python allocation, on one machine, in one
interpreter. It is not an energy measurement and no power figure is inferred from
it: energy requires instrumented hardware this study does not have.

Outputs: results/compute_cost_methods.csv, results/compute_cost_scaling.csv,
results/compute_cost_manifest.json.
"""
from pathlib import Path
import json
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

import core
from core import (CONFIG, Scenario, generate_events, split_events, fit_models,
                  attach_probabilities, scenario_view, select_actions, proposed_options,
                  solve_slot)

ROOT = Path(__file__).resolve().parent
CB = CONFIG["contested_budget"]
METHODS = ("fixed_rule", "fixed_rule_window", "greedy_uncertainty", "greedy_value", "proposed")
GROUP_SIZES = (1, 2, 3, 4, 6, 8, 12)
REPEATS = 5


def timed(fn, repeats=REPEATS):
    """Best-of-n wall clock, which is the standard way to time against a noisy machine."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rows, scale = [], []
    for seed in CB["seeds"][:5]:
        train, cal, test = split_events(generate_events(seed, noise_scale=CB["noise_scale"]))
        models = fit_models(train, cal, seed)
        test = attach_probabilities(test, models)
        sc = Scenario(name="compute", bandwidth_bps=CB["bandwidth_bps"],
                      base_latency_s=CB["base_latency_s"], budget_bytes=1200)
        tv = scenario_view(test, sc, seed, models)
        n_slots = tv.slot_id.nunique()
        for method in METHODS:
            tracemalloc.start()
            t = timed(lambda: select_actions(tv, method, sc, seed))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"seed": seed, "method": method, "windows": len(tv), "slots": n_slots,
                         "total_s": t, "per_slot_ms": t / n_slots * 1e3,
                         "per_window_ms": t / len(tv) * 1e3, "peak_kib": peak / 1024})
        # Exact knapsack alone, as the number of windows sharing one slot grows.
        util, byts, feas = proposed_options(tv, sc, None, CONFIG["policy"]["latency_penalty_per_second"])
        for k in GROUP_SIZES:
            idx = np.arange(k)
            t = timed(lambda: solve_slot(util[idx], byts[idx], feas[idx], sc.budget_bytes * k))
            scale.append({"seed": seed, "windows_per_slot": k, "solve_ms": t * 1e3})
        print(f"  seed {seed} complete")

    out = ROOT / "results"
    m = pd.DataFrame(rows)
    s = pd.DataFrame(scale)
    agg = m.groupby("method").agg(per_slot_ms_mean=("per_slot_ms", "mean"),
                                  per_slot_ms_min=("per_slot_ms", "min"),
                                  per_window_ms_mean=("per_window_ms", "mean"),
                                  peak_kib_mean=("peak_kib", "mean")).reset_index()
    sagg = s.groupby("windows_per_slot").agg(solve_ms_mean=("solve_ms", "mean"),
                                             solve_ms_min=("solve_ms", "min")).reset_index()
    agg.to_csv(out / "compute_cost_methods.csv", index=False)
    sagg.to_csv(out / "compute_cost_scaling.csv", index=False)
    import importlib.metadata as md
    (out / "compute_cost_manifest.json").write_text(json.dumps({
        "experiment": "measured edge-side decision cost",
        "preregistered": False, "exploratory": True,
        "measures": "wall-clock time (best of %d) and peak Python allocation" % REPEATS,
        "does_not_measure": "energy; no power figure is inferred from these timings",
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "python": platform.python_version()},
        "budget_bytes": 1200, "seeds": CB["seeds"][:5], "group_sizes": list(GROUP_SIZES),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "runtime": {p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")},
    }, indent=2) + "\n", encoding="utf-8")
    print()
    print(agg.round(4).to_string(index=False))
    print()
    print(sagg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
