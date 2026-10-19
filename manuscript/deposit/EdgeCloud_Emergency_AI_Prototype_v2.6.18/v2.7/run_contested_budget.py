"""Contested shared budget (option B). Follows PROTOCOL_contested_budget.md exactly.

Outputs in results/: contested_budget_seed_level.csv, contested_budget_summary.csv,
contested_budget_paired.csv, contested_budget_manifest.json.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats

import core
from core import CONFIG, Scenario, generate_events, split_events, fit_models, attach_probabilities, scenario_view, _run

ROOT = Path(__file__).resolve().parent
CB = CONFIG["contested_budget"]
PROTOCOL = ROOT / CB["protocol_file"]
assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == CB["protocol_sha256"], "protocol changed after freezing"
REFERENCE = {"fixed_rule"}  # ignores the budget
HIGH_SEV = CONFIG["policy"]["high_severity_threshold"]


def scenario(b: int) -> Scenario:
    return Scenario(name=f"contested_B{b}", bandwidth_bps=CB["bandwidth_bps"], base_latency_s=CB["base_latency_s"],
                    budget_bytes=int(b))


def run(log=print):
    rows = []
    for seed in CB["seeds"]:
        train, cal, test = split_events(generate_events(seed, noise_scale=CB["noise_scale"]))
        models = fit_models(train, cal, seed)
        cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
        for b in CB["budgets_bytes_per_window"]:
            sc = scenario(b)
            tv, cv = scenario_view(test, sc, seed, models), scenario_view(cal, sc, seed + 7, models)
            for method in CB["methods"]:
                res, met, _ = _run(tv, cv, method, sc, seed, models)
                err = (res.prediction != res.y).to_numpy()
                sev = res.severity_true.to_numpy()
                if method not in REFERENCE:
                    assert met["slot_budget_violation_rate"] == 0.0, (seed, b, method)
                rows.append({"seed": seed, "budget_bytes": b, "method": method, "errors": int(err.sum()),
                             "severity_weighted_errors": float(sev[err].sum()),
                             "high_severity_errors": int((err & (sev >= HIGH_SEV)).sum()),
                             "mean_bytes": met["mean_bytes"], "p95_latency_s": met["p95_latency_s"],
                             "review_rate": met["review_rate"], "macro_f1": met["macro_f1"],
                             "budget_compliant": method not in REFERENCE})
        log(f"seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    n = df.seed.nunique()
    tcrit = stats.t.ppf(0.975, n - 1)
    metrics = ["errors", "severity_weighted_errors", "high_severity_errors", "mean_bytes", "p95_latency_s", "review_rate"]
    summ = []
    for (b, m), g in df.groupby(["budget_bytes", "method"], sort=False):
        row = {"budget_bytes": b, "method": m}
        for k in metrics:
            mu, h = g[k].mean(), tcrit * g[k].std(ddof=1) / np.sqrt(n)
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        summ.append(row)
    paired = []
    for comp in ("greedy_uncertainty", "greedy_value", "fixed_rule_window", "all_summary"):
        for k in ("errors", "severity_weighted_errors", "high_severity_errors", "mean_bytes"):
            for b in CB["budgets_bytes_per_window"]:
                p = df[(df.budget_bytes == b) & (df.method == "proposed")].set_index("seed")[k]
                c = df[(df.budget_bytes == b) & (df.method == comp)].set_index("seed")[k]
                d = (p - c.loc[p.index]).to_numpy(dtype=float)
                mu = d.mean()
                sd = d.std(ddof=1)
                h = tcrit * sd / np.sqrt(n)
                pv = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (1.0 if mu == 0 else 0.0)
                paired.append({"comparison": f"proposed_minus_{comp}", "metric": k, "budget_bytes": b,
                               "mean_difference": mu, "ci_low": mu - h, "ci_high": mu + h, "p_value": pv, "n_seeds": n})
    paired = pd.DataFrame(paired)
    # Holm adjustment across the five budgets, per comparison and metric (protocol decision rule).
    paired["p_holm"] = np.nan
    for (c, k), g in paired.groupby(["comparison", "metric"]):
        order = g.p_value.sort_values().index
        m = len(order)
        adj, run_max = {}, 0.0
        for r, ix in enumerate(order):
            run_max = max(run_max, min(1.0, (m - r) * g.p_value[ix]))
            adj[ix] = run_max
        paired.loc[list(adj), "p_holm"] = pd.Series(adj)
    paired["optimizer_wins"] = (paired.mean_difference < 0) & (paired.p_holm < CB["alpha_holm"])
    return pd.DataFrame(summ), paired


if __name__ == "__main__":
    out = ROOT / "results"
    df = run()
    summ, paired = summarize(df)
    df.to_csv(out / "contested_budget_seed_level.csv", index=False)
    summ.to_csv(out / "contested_budget_summary.csv", index=False)
    paired.to_csv(out / "contested_budget_paired.csv", index=False)
    import importlib.metadata as md
    manifest = {"protocol_sha256": CB["protocol_sha256"], "config_sha256": core.sha256_file(core.CONFIG_PATH),
                "core_sha256": core.sha256_file(Path(core.__file__)),
                "runner_sha256": core.sha256_file(Path(__file__)), "contested_budget": CB,
                "runtime": {"python": platform.python_version(),
                            **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}}}
    (out / "contested_budget_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prim = paired[(paired.comparison == "proposed_minus_greedy_uncertainty") & (paired.metric == "errors")]
    print(summ.pivot(index="method", columns="budget_bytes", values="errors_mean").round(1).to_string())
    print(prim.round(4).to_string())
