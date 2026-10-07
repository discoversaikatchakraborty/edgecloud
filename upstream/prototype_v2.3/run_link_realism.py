"""Packet loss and measured-capacity replay (exploratory).

Two questions the constant-rate link model cannot answer. First, does loss change
the trade-off between sending a summary and sending a raw record? Loss inflates
bytes on the wire and costs one timeout per lost packet, so the penalty grows with
payload and should fall hardest on the methods that send most. Second, does the
paper's main conclusion survive a real, variable link instead of a constant one?

Both experiments hold everything except the link fixed, and both were designed
after the version 2.3 results; they are exploratory and reported as such. Setting
packet_loss = 0 with no trace reproduces the earlier results exactly.

Outputs: results/link_loss_{seed_level,summary}.csv,
results/link_trace_{seed_level,summary}.csv, results/link_realism_manifest.json.
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
_PROV = json.loads((ROOT / "data/traces/provenance.json").read_text())
TRACE_STATS = _PROV["traces"]
BASE = next(s for s in CONFIG["scenarios"] if s["name"] == "low_bandwidth")
NORMAL = next(s for s in CONFIG["scenarios"] if s["name"] == "normal")
LOSS_RATES = (0.0, 0.01, 0.02, 0.05, 0.10)
TRACES = ("Verizon-LTE-short.up", "Verizon-LTE-driving.up", "TMobile-LTE-driving.up")
METHODS = ("cloud_transmit_all", "fixed_confidence_gate", "fixed_compression",
           "fixed_rule", "proposed")
REFERENCE = {"cloud_transmit_all"}
METRICS = ("errors", "mean_bytes", "p95_latency_s", "median_latency_s", "review_rate", "macro_f1")


def make(name, base=None, **over):
    spec = dict(base if base is not None else BASE)
    spec["name"] = name
    spec["rng_name"] = (base if base is not None else BASE)["name"]
    spec.update(over)
    return Scenario(**spec)


def sweep(variants, label, log=print):
    rows = []
    for seed in CONFIG["seeds"]:
        train, cal, test = split_events(generate_events(seed))
        models = fit_models(train, cal, seed)
        cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
        for key, sc in variants:
            tv = scenario_view(test, sc, seed, models)
            cv = scenario_view(cal, sc, seed + 7, models)
            for method in METHODS:
                res, met, _ = _run(tv, cv, method, sc, seed, models)
                if method not in REFERENCE:
                    assert met["slot_budget_violation_rate"] == 0.0, (seed, key, method)
                err = (res.prediction != res.y).to_numpy()
                rows.append({"seed": seed, label: key, "method": method,
                             "errors": int(err.sum()),
                             **{k: met[k] for k in METRICS if k != "errors"}})
        log(f"  seed {seed} complete")
    return pd.DataFrame(rows)


def summarize(df, label):
    n = df.seed.nunique()
    t = stats.t.ppf(0.975, n - 1)
    out = []
    for (key, m), grp in df.groupby([label, "method"], sort=False):
        row = {label: key, "method": m}
        for k in METRICS:
            mu, sd = grp[k].mean(), grp[k].std(ddof=1)
            h = t * sd / np.sqrt(n)
            row.update({f"{k}_mean": mu, f"{k}_ci_low": mu - h, f"{k}_ci_high": mu + h})
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    out = ROOT / "results"
    print("packet loss sweep")
    loss = sweep([(p, make(f"loss_{p:g}", packet_loss=p)) for p in LOSS_RATES], "packet_loss")
    ls = summarize(loss, "packet_loss")
    loss.to_csv(out / "link_loss_seed_level.csv", index=False)
    ls.to_csv(out / "link_loss_summary.csv", index=False)

    print("packet loss sweep, normal budget")
    lossn = sweep([(p, make(f"lossn_{p:g}", base=NORMAL, packet_loss=p)) for p in LOSS_RATES],
                  "packet_loss")
    lns = summarize(lossn, "packet_loss")
    lossn.to_csv(out / "link_loss_normal_seed_level.csv", index=False)
    lns.to_csv(out / "link_loss_normal_summary.csv", index=False)

    # Each trace is compared with a constant link at the SAME mean capacity, so any
    # difference is attributable to variability rather than to how fast the link is.
    print("measured capacity replay, mean-matched constants")
    variants = []
    for t in TRACES:
        bps = TRACE_STATS[t]["mean_mbps"] * 1e6
        variants.append((f"{t} (replayed)", make(f"trace_{t}", link_trace=t, bandwidth_bps=bps)))
        variants.append((f"{t} (constant at same mean)", make(f"const_{t}", bandwidth_bps=bps)))
    tr = sweep(variants, "link")
    ts = summarize(tr, "link")
    tr.to_csv(out / "link_trace_seed_level.csv", index=False)
    ts.to_csv(out / "link_trace_summary.csv", index=False)

    import importlib.metadata as md
    prov = json.loads((ROOT / "data/traces/provenance.json").read_text())
    (out / "link_realism_manifest.json").write_text(json.dumps({
        "experiment": "packet loss and measured-capacity replay",
        "preregistered": False, "exploratory": True,
        # Three sweeps run here and they do not share a base. Recording a single
        # base_scenario made the manifest disagree with the loss table the paper
        # reports, which is drawn from the normal-condition sweep.
        "sweeps": {
            "packet_loss_low_bandwidth": {"base_scenario": BASE,
                                          "outputs": ["link_loss_seed_level.csv",
                                                      "link_loss_summary.csv"]},
            "packet_loss_normal": {"base_scenario": NORMAL,
                                   "outputs": ["link_loss_normal_seed_level.csv",
                                               "link_loss_normal_summary.csv"]},
            "measured_capacity_replay": {"base_scenario": BASE,
                                         "outputs": ["link_trace_seed_level.csv",
                                                     "link_trace_summary.csv"]},
        },
        "loss_rates": list(LOSS_RATES), "traces": list(TRACES),
        "methods": list(METHODS), "seeds": CONFIG["seeds"],
        "mtu_bytes": core.MTU_BYTES, "retransmit_timeout_s": core.RETRANSMIT_TIMEOUT_S,
        "trace_provenance": prov,
        "config_sha256": core.sha256_file(core.CONFIG_PATH),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "runtime": {"python": platform.python_version(),
                    **{p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}},
    }, indent=2) + "\n", encoding="utf-8")

    print()
    print(ls.pivot(index="method", columns="packet_loss", values="errors_mean").round(1).to_string())
    print()
    print(ls.pivot(index="method", columns="packet_loss", values="p95_latency_s_mean").round(2).to_string())
    print()
    print(lns.pivot(index="method", columns="packet_loss", values="p95_latency_s_mean").round(2).to_string())
    print()
    print(ts.pivot(index="method", columns="link", values="p95_latency_s_mean").round(2).to_string())
