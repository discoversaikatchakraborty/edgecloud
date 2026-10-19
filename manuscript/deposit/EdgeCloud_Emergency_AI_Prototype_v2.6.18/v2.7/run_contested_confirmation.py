"""Confirmatory run on new seeds. Follows PROTOCOL_contested_budget_confirmation.md."""
from pathlib import Path
import hashlib
import numpy as np, pandas as pd
from scipy import stats
import run_contested_budget as R
from core import CONFIG, generate_events, split_events, fit_models, attach_probabilities, scenario_view, _run

ROOT = Path(__file__).resolve().parent
CF = CONFIG["contested_budget"]["confirmation"]
assert hashlib.sha256((ROOT / CF["protocol_file"]).read_bytes()).hexdigest() == CF["protocol_sha256"]
rows = []
for seed in CF["seeds"]:
    train, cal, test = split_events(generate_events(seed, noise_scale=R.CB["noise_scale"]))
    models = fit_models(train, cal, seed)
    cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
    for b in R.CB["budgets_bytes_per_window"]:
        sc = R.scenario(b)
        tv, cv = scenario_view(test, sc, seed, models), scenario_view(cal, sc, seed + 7, models)
        for m in CF["methods"]:
            res, met, _ = _run(tv, cv, m, sc, seed, models)
            assert met["slot_budget_violation_rate"] == 0.0
            rows.append({"seed": seed, "budget_bytes": b, "method": m, "errors": int((res.prediction != res.y).sum()),
                         "mean_bytes": met["mean_bytes"], "p95_latency_s": met["p95_latency_s"]})
    print(f"seed {seed} complete")
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "contested_budget_confirmation_seed_level.csv", index=False)
out = []
for comp in ("greedy_uncertainty", "greedy_value_summary_floor", "proposed", "greedy_value"):
    part = []
    for b in R.CB["budgets_bytes_per_window"]:
        p = df[(df.budget_bytes == b) & (df.method == "proposed_summary_floor")].set_index("seed").errors
        c = df[(df.budget_bytes == b) & (df.method == comp)].set_index("seed").errors
        d = (p - c.loc[p.index]).to_numpy(float)
        h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        pv = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else (1.0 if d.mean() == 0 else 0.0)
        part.append({"comparison": f"proposed_summary_floor_minus_{comp}", "budget_bytes": b,
                     "errors_floor": p.mean(), "errors_comparator": c.mean(), "mean_difference": d.mean(),
                     "ci_low": d.mean() - h, "ci_high": d.mean() + h, "p_value": pv})
    part = pd.DataFrame(part).sort_values("p_value")
    m, run_max, adj = len(part), 0.0, []
    for r, pv in enumerate(part.p_value):
        run_max = max(run_max, min(1.0, (m - r) * pv)); adj.append(run_max)
    part["p_holm"] = adj
    part["confirmed"] = (part.mean_difference < 0) & (part.p_holm < 0.05)
    out.append(part.sort_values("budget_bytes"))
o = pd.concat(out, ignore_index=True)
o.to_csv(ROOT / "results" / "contested_budget_confirmation.csv", index=False)
print(o.round(4).to_string())
print(df.pivot_table(index="method", columns="budget_bytes", values="errors").round(1).to_string())
