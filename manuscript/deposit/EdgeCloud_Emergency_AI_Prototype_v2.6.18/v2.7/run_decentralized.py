"""What does coordination between controllers buy? (exploratory)

Every other experiment assumes one controller owns all K windows in a slot and
allocates the whole shared budget. That is the most favourable possible assumption
for a shared-budget formulation, and it is not how field deployments are built: an
incident area is more likely to have several edge nodes that cannot coordinate
per-slot over the same degraded link they are trying to conserve.

Here the slot's windows are dealt round-robin to C independent controllers, each
allocating only its own windows against a proportional share of the byte budget.
The total budget is unchanged and the physical link is still shared by all of
them; only coordination is removed. C = 1 is the centralised design and reproduces
the earlier results exactly.

The contested-budget condition is used because it is the only regime in which the
shared budget binds, and therefore the only one in which coordination can matter.

Exploratory: designed after the version 2.3 results, not pre-registered.

Outputs: results/decentralized_{seed_level,summary,paired}.csv,
results/decentralized_manifest.json.
"""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats

import core
from core import (CONFIG, Scenario, generate_events, split_events, fit_models,
                  attach_probabilities, scenario_view, _run)

ROOT = Path(__file__).resolve().parent
CB = CONFIG["contested_budget"]
CONTROLLERS = (1, 2, 3, 6)          # 6 = one controller per window, no sharing at all
METHODS = ("proposed", "greedy_uncertainty", "fixed_rule_window", "all_summary")
COORDINATED = {"proposed", "greedy_uncertainty"}   # the two that use the shared budget
METRICS = ("errors", "mean_bytes", "p95_latency_s", "review_rate", "macro_f1")


def scenario(b, c):
    return Scenario(name=f"dec_B{b}_C{c}", rng_name=f"contested_B{b}",
                    bandwidth_bps=CB["bandwidth_bps"], base_latency_s=CB["base_latency_s"],
                    budget_bytes=int(b), controllers=int(c))


def run(log=print):
    rows = []
    for seed in CB["seeds"]:
        train, cal, test = split_events(generate_events(seed, noise_scale=CB["noise_scale"]))
        models = fit_models(train, cal, seed)
        cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
        for b in CB["budgets_bytes_per_window"]:
            for c in CONTROLLERS:
                sc = scenario(b, c)
                tv, cv = scenario_view(test, sc, seed, models), scenario_view(cal, sc, seed + 7, models)
                for method in METHODS:
                    res, met, _ = _run(tv, cv, method, sc, seed, models)
                    assert met["slot_budget_violation_rate"] == 0.0, (seed, b, c, method)
                    err = (res.prediction != res.y).to_numpy()
                    rows.append({"seed": seed, "budget_bytes": b, "controllers": c,
                                 "method": method, "errors": int(err.sum()),
                                 **{k: met[k] for k in METRICS if k != "errors"}})
        log(f"  seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df):
    n = df.seed.nunique()
    t = stats.t.ppf(0.975, n - 1)
    out = []
    for (b, c, m), g in df.groupby(["budget_bytes", "controllers", "method"], sort=False):
        row = {"budget_bytes": b, "controllers": c, "method": m}
        for k in METRICS:
            mu, sd = g[k].mean(), g[k].std(ddof=1)
            h = t * sd / np.sqrt(n)
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        out.append(row)
    paired = []
    for m in METHODS:
        for b in CB["budgets_bytes_per_window"]:
            base = df[(df.method == m) & (df.budget_bytes == b) & (df.controllers == 1)].set_index("seed").errors
            for c in CONTROLLERS[1:]:
                cur = df[(df.method == m) & (df.budget_bytes == b) & (df.controllers == c)].set_index("seed").errors
                d = (cur - base.loc[cur.index]).to_numpy(dtype=float)
                mu, sd = d.mean(), d.std(ddof=1)
                h = t * sd / np.sqrt(n)
                pv = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (1.0 if mu == 0 else 0.0)
                paired.append({"method": m, "budget_bytes": b, "controllers": c,
                               "cost_of_decentralization": mu, "ci_low": mu - h, "ci_high": mu + h,
                               "p_value": pv, "n_seeds": n})
    return pd.DataFrame(out), pd.DataFrame(paired)


if __name__ == "__main__":
    out = ROOT / "results"
    df = run()
    summ, paired = summarize(df)
    df.to_csv(out / "decentralized_seed_level.csv", index=False)
    summ.to_csv(out / "decentralized_summary.csv", index=False)
    paired.to_csv(out / "decentralized_paired.csv", index=False)
    import importlib.metadata as md
    (out / "decentralized_manifest.json").write_text(json.dumps({
        "experiment": "cost of removing coordination between controllers",
        "preregistered": False, "exploratory": True,
        "contested_budget": CB, "controllers": list(CONTROLLERS), "methods": list(METHODS),
        "partition": "round-robin within a slot; budget share proportional to windows owned",
        "config_sha256": core.sha256_file(core.CONFIG_PATH),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "runtime": {"python": platform.python_version(),
                    **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}},
    }, indent=2) + "\n", encoding="utf-8")
    print(summ.pivot_table(index="method", columns=["controllers", "budget_bytes"],
                           values="errors_mean").round(1).to_string())
