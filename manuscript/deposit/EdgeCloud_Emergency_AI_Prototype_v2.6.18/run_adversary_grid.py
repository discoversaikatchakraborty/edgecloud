"""Interaction of two adversary capabilities (exploratory).

The spoofing sweep varies one capability: forging item metadata so that a foreign
report passes the admissibility check. A real adversary may also control the
report's content. This grid crosses the two so that their interaction is measured
rather than assumed, holding the misaligned-evidence condition fixed otherwise.

Exploratory: designed after the version 2.3 results, not pre-registered.

Outputs: results/adversary_grid_{seed_level,summary}.csv, adversary_grid_manifest.json.
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
SPOOF = (0.0, 0.25, 0.5, 1.0)          # 0.25 is the reference condition, and acts as a control
HOSTILE = (0.0, 0.35)            # ability to control content toward a wrong class
METHODS = ("fixed_rule", "fixed_rule_admissible", "proposed")
METRICS = ("errors", "misaligned_text_fused_rate", "review_rate", "macro_f1", "accepted_errors")


def scenario(spoof, hostile):
    spec = dict(BASE)
    spec["name"] = f"adv_s{spoof:g}_h{hostile:g}"
    spec["rng_name"] = BASE["name"]
    spec["spoofed_metadata"] = float(spoof)
    spec["contradictory_text"] = float(hostile)
    return Scenario(**spec)


def run(log=print):
    rows = []
    for seed in CONFIG["seeds"]:
        train, cal, test = split_events(generate_events(seed))
        models = fit_models(train, cal, seed)
        cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
        for spoof in SPOOF:
            for hostile in HOSTILE:
                sc = scenario(spoof, hostile)
                tv = scenario_view(test, sc, seed, models)
                cv = scenario_view(cal, sc, seed + 7, models)
                for method in METHODS:
                    res, met, _ = _run(tv, cv, method, sc, seed, models)
                    err = (res.prediction != res.y).to_numpy()
                    rows.append({"seed": seed, "spoof_rate": spoof, "hostile_rate": hostile,
                                 "method": method, "errors": int(err.sum()),
                                 **{k: met.get(k, float("nan")) for k in METRICS if k != "errors"}})
        log(f"  seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df):
    n = df.seed.nunique()
    t = stats.t.ppf(0.975, n - 1)
    out = []
    for (s_, h_, m), g in df.groupby(["spoof_rate", "hostile_rate", "method"], sort=False):
        row = {"spoof_rate": s_, "hostile_rate": h_, "method": m}
        for k in METRICS:
            mu, sd = g[k].mean(), g[k].std(ddof=1)
            h = t * sd / np.sqrt(n) if np.isfinite(sd) else float("nan")
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    out = ROOT / "results"
    df = run()
    summ = summarize(df)
    df.to_csv(out / "adversary_grid_seed_level.csv", index=False)
    summ.to_csv(out / "adversary_grid_summary.csv", index=False)
    import importlib.metadata as md
    (out / "adversary_grid_manifest.json").write_text(json.dumps({
        "experiment": "metadata forgery crossed with content control",
        "preregistered": False, "exploratory": True,
        "base_scenario": BASE, "spoof_rates": list(SPOOF), "hostile_rates": list(HOSTILE),
        "methods": list(METHODS), "seeds": CONFIG["seeds"],
        "config_sha256": core.sha256_file(core.CONFIG_PATH),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "runtime": {"python": platform.python_version(),
                    **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}},
    }, indent=2) + "\n", encoding="utf-8")
    piv = summ.pivot_table(index="method", columns=["hostile_rate", "spoof_rate"], values="errors_mean")
    print(piv.round(1).to_string())
