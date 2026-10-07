"""EXPLORATORY, not pre-specified: added after the confirmatory contested-budget results showed the
optimizer holding whole modalities under the tightest budget. Tests one variant, the proposed
optimizer with a summary floor, against greedy_uncertainty. Does not change the protocol verdict."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import run_contested_budget as R
from core import generate_events, split_events, fit_models, attach_probabilities, scenario_view, _run

rows = []
for seed in R.CB["seeds"]:
    train, cal, test = split_events(generate_events(seed, noise_scale=R.CB["noise_scale"]))
    models = fit_models(train, cal, seed)
    cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
    for b in R.CB["budgets_bytes_per_window"]:
        sc = R.scenario(b)
        tv, cv = scenario_view(test, sc, seed, models), scenario_view(cal, sc, seed + 7, models)
        res, met, _ = _run(tv, cv, "proposed_summary_floor", sc, seed, models)
        assert met["slot_budget_violation_rate"] == 0.0
        rows.append({"seed": seed, "budget_bytes": b, "method": "proposed_summary_floor",
                     "errors": int((res.prediction != res.y).sum()), "mean_bytes": met["mean_bytes"]})
ex = pd.DataFrame(rows)
base = pd.read_csv(Path(__file__).parent / "results" / "contested_budget_seed_level.csv")
out = []
for b in R.CB["budgets_bytes_per_window"]:
    e = ex[ex.budget_bytes == b].set_index("seed")
    for comp in ("greedy_uncertainty", "proposed"):
        c = base[(base.budget_bytes == b) & (base.method == comp)].set_index("seed")
        d = (e.errors - c.errors.loc[e.index]).to_numpy(float)
        h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        out.append({"budget_bytes": b, "comparison": f"summary_floor_minus_{comp}", "errors_floor": e.errors.mean(),
                    "mean_difference": d.mean(), "ci_low": d.mean() - h, "ci_high": d.mean() + h,
                    "p_value": float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0,
                    "bytes_floor": e.mean_bytes.mean()})
o = pd.DataFrame(out)
o.to_csv(Path(__file__).parent / "results" / "contested_budget_exploratory.csv", index=False)
print(o.round(3).to_string())
