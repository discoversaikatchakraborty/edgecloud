"""Diagnostic for Section VI-G: share of windows in which each method holds (does not send) an
available modality, per budget, over the ten pre-specified seeds."""
from pathlib import Path
import pandas as pd
import run_contested_budget as R
from core import generate_events, split_events, fit_models, attach_probabilities, scenario_view, select_actions, MODALITIES

rows = []
for seed in R.CB["seeds"]:
    train, cal, test = split_events(generate_events(seed, noise_scale=R.CB["noise_scale"]))
    models = fit_models(train, cal, seed)
    test = attach_probabilities(test, models)
    for b in R.CB["budgets_bytes_per_window"]:
        sc = R.scenario(b)
        v = scenario_view(test, sc, seed, models)
        for m in ("proposed", "greedy_value", "greedy_uncertainty", "proposed_summary_floor"):
            a = select_actions(v, m, sc, seed)
            row = {"seed": seed, "budget_bytes": b, "method": m}
            for k, mod in enumerate(MODALITIES):
                row[f"{mod}_held_rate"] = float((a[:, k] == 0).mean())
            row["any_held_rate"] = float((a == 0).any(axis=1).mean())
            rows.append(row)
out = pd.DataFrame(rows).groupby(["budget_bytes", "method"], as_index=False).mean(numeric_only=True).drop(columns="seed")
out.to_csv(Path(__file__).parent / "results" / "contested_budget_hold_rates.csv", index=False)
print(out.round(3).to_string())
