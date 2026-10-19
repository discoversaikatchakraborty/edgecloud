"""Adversarial metadata spoofing against the admissibility filter (exploratory).

Section VI-D reports that the admissibility rule rejects misaligned reports whose
metadata belongs to another event, and states without measuring it that spoofed or
wrong metadata defeats the rule. This sweep measures that. It varies the fraction
of misaligned reports whose location, time and hazard tag are set to the target
incident's own values, which is what an adversary who can forge metadata supplies,
and holds everything else at the misaligned-evidence condition.

This experiment was designed after the version 2.2 results and is exploratory. It
was not pre-registered, and it is reported as a robustness curve rather than a
hypothesis test. Constants live here rather than in config.json so that the frozen
configuration digest recorded by every other experiment is unchanged.

Outputs in results/: spoof_sweep_seed_level.csv, spoof_sweep_summary.csv,
spoof_sweep_paired.csv, spoof_sweep_manifest.json.
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
BASE = next(s for s in CONFIG["scenarios"] if s["name"] == "misaligned_evidence")
SPOOF_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("proposed", "fixed_rule_admissible", "fixed_rule")
UNFILTERED = {"fixed_rule"}  # no admissibility rule; the control arm
METRICS = ("errors", "misaligned_text_fused_rate", "aligned_text_rejected_rate",
           "review_rate", "macro_f1", "mean_bytes")


def scenario(rate: float) -> Scenario:
    spec = dict(BASE)
    spec["name"] = f"spoof_{rate:g}"
    spec["rng_name"] = BASE["name"]
    spec["spoofed_metadata"] = float(rate)
    return Scenario(**spec)


def run(log=print) -> pd.DataFrame:
    rows = []
    for seed in CONFIG["seeds"]:
        train, cal, test = split_events(generate_events(seed))
        models = fit_models(train, cal, seed)
        cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
        for rate in SPOOF_RATES:
            sc = scenario(rate)
            tv = scenario_view(test, sc, seed, models)
            cv = scenario_view(cal, sc, seed + 7, models)
            for method in METHODS:
                res, met, _ = _run(tv, cv, method, sc, seed, models)
                err = (res.prediction != res.y).to_numpy()
                row = {"seed": seed, "spoof_rate": rate, "method": method,
                       "errors": int(err.sum()), "filters_metadata": method not in UNFILTERED}
                for k in METRICS:
                    if k != "errors":
                        row[k] = met.get(k, float("nan"))
                rows.append(row)
        log(f"seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    n = df.seed.nunique()
    tcrit = stats.t.ppf(0.975, n - 1)
    summ = []
    for (rate, m), g in df.groupby(["spoof_rate", "method"], sort=False):
        row = {"spoof_rate": rate, "method": m}
        for k in METRICS:
            mu = g[k].mean()
            sd = g[k].std(ddof=1)
            h = tcrit * sd / np.sqrt(n) if np.isfinite(sd) else float("nan")
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        summ.append(row)
    paired = []
    for a, b in (("proposed", "fixed_rule"), ("fixed_rule_admissible", "fixed_rule"),
                 ("proposed", "fixed_rule_admissible")):
        for k in ("errors", "misaligned_text_fused_rate"):
            for rate in SPOOF_RATES:
                x = df[(df.spoof_rate == rate) & (df.method == a)].set_index("seed")[k]
                y = df[(df.spoof_rate == rate) & (df.method == b)].set_index("seed")[k]
                d = (x - y.loc[x.index]).to_numpy(dtype=float)
                mu, sd = d.mean(), d.std(ddof=1)
                h = tcrit * sd / np.sqrt(n)
                pv = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (1.0 if mu == 0 else 0.0)
                paired.append({"comparison": f"{a}_minus_{b}", "metric": k, "spoof_rate": rate,
                               "mean_difference": mu, "ci_low": mu - h, "ci_high": mu + h,
                               "p_value": pv, "n_seeds": n})
    return pd.DataFrame(summ), pd.DataFrame(paired)


if __name__ == "__main__":
    out = ROOT / "results"
    df = run()
    summ, paired = summarize(df)
    df.to_csv(out / "spoof_sweep_seed_level.csv", index=False)
    summ.to_csv(out / "spoof_sweep_summary.csv", index=False)
    paired.to_csv(out / "spoof_sweep_paired.csv", index=False)
    import importlib.metadata as md
    manifest = {"experiment": "adversarial metadata spoofing against admissibility",
                "preregistered": False, "exploratory": True,
                "base_scenario": BASE, "spoof_rates": list(SPOOF_RATES), "methods": list(METHODS),
                "seeds": CONFIG["seeds"],
                "config_sha256": core.sha256_file(core.CONFIG_PATH),
                "core_sha256": core.sha256_file(Path(core.__file__)),
                "runner_sha256": core.sha256_file(Path(__file__)),
                "runtime": {"python": platform.python_version(),
                            **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}}}
    (out / "spoof_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summ.pivot(index="method", columns="spoof_rate", values="errors_mean").round(1).to_string())
    print()
    print(summ.pivot(index="method", columns="spoof_rate", values="misaligned_text_fused_rate_mean").round(3).to_string())
