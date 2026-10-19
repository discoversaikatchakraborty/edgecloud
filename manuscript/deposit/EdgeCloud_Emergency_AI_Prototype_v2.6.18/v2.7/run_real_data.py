"""The policy comparison on real disaster data (CrisisMMD).

Every result elsewhere in this package rests on a synthetic generator, and the
manuscript says so repeatedly. This runs the same policies, unchanged, on a
workload built from CrisisMMD v2.0 by prepare_crisismmd.py: real text, real images,
real event identity, real hazard types, real arrival times, real class imbalance
and real payload sizes.

Differences from the synthetic workload, all forced by what the data contains:
there is no sensor modality, so that modality is marked permanently unavailable;
there is no geolocation, so spatial admissibility degrades to event identity; and
there is no severity field independent of the damage label, so severity is held
constant and severity-dependent behaviour is inert.

Exploratory: not pre-registered.

Outputs: results/real_data_{seed_level,summary,paired}.csv, real_data_manifest.json.
"""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats

import core
from core import (CONFIG, MODALITIES, Scenario, fit_models, attach_probabilities,
                  scenario_view, _run, FULL_DIMS)

ROOT = Path(__file__).resolve().parent
REC = ROOT / "data" / "crisismmd" / "records.csv.gz"
SEEDS = CONFIG["seeds"]
# greedy_value solves the same multiple-choice knapsack as `proposed` with a greedy
# heuristic instead of the exact dynamic program, so the pair isolates the solver from
# the utility it consumes. Without it the claim that the solver is not where the value
# lies rests on the synthetic benchmark alone.
METHODS = ("image_only", "text_only", "cloud_transmit_all", "fixed_confidence_gate",
           "fixed_compression", "fixed_rule", "random_within_budget",
           "greedy_uncertainty", "greedy_value", "proposed")
REFERENCE = {"cloud_transmit_all"}
METRICS = ("errors", "mean_bytes", "p95_latency_s", "median_latency_s", "review_rate",
           "macro_f1", "conformal_coverage")


def build_frame() -> pd.DataFrame:
    r = pd.read_csv(REC)
    n = len(r)
    d = pd.DataFrame({
        "event_id": r.event_id, "window_id": r.window_id.astype(str), "y": r.y.astype(int),
        # No severity field exists that is independent of the label. Holding it
        # constant keeps severity out of the decision rather than inventing it.
        "severity_true": np.full(n, 0.5),
        "hazard": r.hazard,
        "decision_time_s": r.obs_time_s - r.groupby(r.event_id).obs_time_s.transform("min"),
    })
    # No geolocation in CrisisMMD: the incident cell is the event identity, so the
    # spatial admissibility term reduces to "does this report belong to this event".
    codes = {e: i for i, e in enumerate(sorted(r.event_id.unique()))}
    d["cell_x"] = r.event_id.map(codes).astype(int) * 50
    d["cell_y"] = r.event_id.map(codes).astype(int) * 50
    for m in MODALITIES:
        if m == "sensor":
            # CrisisMMD has no sensor modality. A fixed-seed placeholder is supplied
            # only so the shared three-modality code path runs; every scenario sets
            # missing_sensor = 1.0, and run() asserts the modality is never available,
            # so it cannot enter any decision, fusion step or byte count.
            rng = np.random.default_rng(0)
            d[f"{m}_x"] = list(rng.standard_normal((n, FULL_DIMS[m])))
            d[f"{m}_quality"] = 0.5
            d[f"{m}_reliability"] = 0.5
            d[f"{m}_raw_bytes"] = 1
        else:
            cols = [f"{m}_x{i}" for i in range(FULL_DIMS[m])]
            d[f"{m}_x"] = list(r[cols].to_numpy(dtype="float64"))
            d[f"{m}_quality"] = r[f"{m}_quality"].clip(0.15, 1.0).to_numpy()
            # Annotator agreement on the other modality is an observable proxy for
            # how much this source can be trusted; it never uses the label.
            other = "text" if m == "image" else "image"
            d[f"{m}_reliability"] = r[f"{other}_quality"].clip(0.30, 0.99).to_numpy()
            d[f"{m}_raw_bytes"] = r[f"{m}_raw_bytes"].astype(int).to_numpy()
    d["split"] = r.split
    return d


def scenario(name, **over):
    base = next(s for s in CONFIG["scenarios"] if s["name"] == name)
    spec = dict(base)
    spec["missing_sensor"] = 1.0        # CrisisMMD has no sensor modality
    spec.update(over)
    return Scenario(**spec)


def run(log=print):
    d = build_frame()
    train = d[d.split == "train"].drop(columns="split").reset_index(drop=True)
    cal = d[d.split == "calibration"].drop(columns="split").reset_index(drop=True)
    test = d[d.split == "test"].drop(columns="split").reset_index(drop=True)
    log(f"  train {len(train)}  calibration {len(cal)}  test {len(test)} windows")
    rows = []
    for seed in SEEDS:
        models = fit_models(train, cal, seed)
        c, t = attach_probabilities(cal, models), attach_probabilities(test, models)
        for cond in ("normal", "low_bandwidth", "misaligned_evidence"):
            sc = scenario(cond)
            tv, cv = scenario_view(t, sc, seed, models), scenario_view(c, sc, seed + 7, models)
            # The placeholder sensor modality must never be available anywhere.
            assert not tv.sensor_available.any() and not cv.sensor_available.any(), cond
            for method in METHODS:
                res, met, _ = _run(tv, cv, method, sc, seed, models)
                if method not in REFERENCE:
                    assert met["slot_budget_violation_rate"] == 0.0, (seed, cond, method)
                err = (res.prediction != res.y).to_numpy()
                rows.append({"seed": seed, "scenario": cond, "method": method,
                             "windows": len(res), "errors": int(err.sum()),
                             **{k: met.get(k, float("nan")) for k in METRICS if k != "errors"}})
        log(f"  seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df):
    n = df.seed.nunique()
    tc = stats.t.ppf(0.975, n - 1)
    out = []
    for (s_, m), g in df.groupby(["scenario", "method"], sort=False):
        row = {"scenario": s_, "method": m, "windows": int(g.windows.iloc[0])}
        for k in METRICS:
            mu, sd = g[k].mean(), g[k].std(ddof=1)
            h = tc * sd / np.sqrt(n) if np.isfinite(sd) else float("nan")
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        out.append(row)
    paired = []
    for s_ in df.scenario.unique():
        base = df[(df.scenario == s_) & (df.method == "proposed")].set_index("seed")
        for m in METHODS:
            if m == "proposed":
                continue
            o = df[(df.scenario == s_) & (df.method == m)].set_index("seed")
            for k in ("errors", "mean_bytes", "p95_latency_s"):
                dd = (base[k] - o[k].loc[base.index]).to_numpy(dtype=float)
                mu, sd = dd.mean(), dd.std(ddof=1)
                h = tc * sd / np.sqrt(n)
                paired.append({"scenario": s_, "comparison": f"proposed_minus_{m}", "metric": k,
                               "mean_difference": mu, "ci_low": mu - h, "ci_high": mu + h})
    return pd.DataFrame(out), pd.DataFrame(paired)


if __name__ == "__main__":
    out = ROOT / "results"
    df = run()
    summ, paired = summarize(df)
    df.to_csv(out / "real_data_seed_level.csv", index=False)
    summ.to_csv(out / "real_data_summary.csv", index=False)
    paired.to_csv(out / "real_data_paired.csv", index=False)
    import importlib.metadata as md
    prov = json.loads((ROOT / "data/crisismmd/provenance.json").read_text())
    (out / "real_data_manifest.json").write_text(json.dumps({
        "experiment": "policy comparison on CrisisMMD real disaster data",
        "preregistered": False, "exploratory": True,
        "dataset_provenance": prov, "methods": list(METHODS), "seeds": SEEDS,
        "conditions": ["normal", "low_bandwidth", "misaligned_evidence"],
        "sensor_modality": "absent in CrisisMMD; marked permanently unavailable",
        "severity": "held constant; no field independent of the label exists",
        "spatial_admissibility": "degrades to event identity; no geolocation in the annotations",
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "runtime": {"python": platform.python_version(),
                    **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}},
    }, indent=2) + "\n", encoding="utf-8")
    print()
    print(summ.pivot(index="method", columns="scenario", values="errors_mean").round(1).to_string())
    print()
    print(summ.pivot(index="method", columns="scenario", values="mean_bytes_mean").round(0).to_string())
