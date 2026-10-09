"""Reduce emulator_summary.csv to the quantities the manuscript quotes (item F)."""
import json, os
import pandas as pd
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "emulator")
s = pd.read_csv(os.path.join(OUT, "emulator_summary.csv"))
small = s[s.payload_bytes <= 2048]; raw = s[s.payload_bytes == 420000]
k = {}
# additive gap on loss-free constant links, small payloads: measured - modelled, by RTT
for cell, rtt in (("low_bandwidth_loss0", 0.18), ("delayed_imagery_loss0", 0.10), ("normal_loss0", 0.04)):
    g = small[small.cell == cell]
    k[f"small_gap_s[{cell}]"] = {"rtt_s": rtt, "gap_mean_s": round(float((g.measured_mean_s - g.modelled_mean_s).mean()), 3),
                                 "gap_min_s": round(float((g.measured_mean_s - g.modelled_mean_s).min()), 3),
                                 "gap_max_s": round(float((g.measured_mean_s - g.modelled_mean_s).max()), 3),
                                 "ratio_min": round(float(g.ratio_mean.min()), 2), "ratio_max": round(float(g.ratio_mean.max()), 2)}
# raw record on constant links, loss 0
for cell in ("low_bandwidth_loss0", "delayed_imagery_loss0", "normal_loss0",
             "constmatched_Verizon-LTE-short.up", "constmatched_Verizon-LTE-driving.up", "constmatched_TMobile-LTE-driving.up"):
    r = raw[raw.cell == cell].iloc[0]
    k[f"raw[{cell}]"] = {"modelled_s": round(float(r.modelled_mean_s), 3), "measured_s": round(float(r.measured_mean_s), 3),
                         "ratio": round(float(r.ratio_mean), 2)}
# loss sweep, raw record and small-payload p95
for scn in ("low_bandwidth", "normal"):
    for p in (0.0, 0.01, 0.02, 0.05, 0.1):
        cell = f"{scn}_loss{p:g}"
        r = raw[raw.cell == cell].iloc[0]; g = small[small.cell == cell]
        k[f"loss[{cell}]"] = {"raw_modelled_s": round(float(r.modelled_mean_s), 2), "raw_measured_s": round(float(r.measured_mean_s), 2),
                              "raw_ratio": round(float(r.ratio_mean), 2), "raw_p95_s": round(float(r.measured_p95_s), 2),
                              "small_p95_max_s": round(float(g.measured_p95_s.max()), 2)}
# trace vs mean-matched constant, measured raw record
for t in ("Verizon-LTE-short.up", "Verizon-LTE-driving.up", "TMobile-LTE-driving.up"):
    a = raw[raw.cell == f"trace_{t}"].iloc[0]; b = raw[raw.cell == f"constmatched_{t}"].iloc[0]
    k[f"trace_vs_const[{t}]"] = {"trace_mean_s": round(float(a.measured_mean_s), 2), "const_mean_s": round(float(b.measured_mean_s), 2),
                                 "mean_ratio": round(float(a.measured_mean_s / b.measured_mean_s), 2),
                                 "trace_p95_s": round(float(a.measured_p95_s), 2), "const_p95_s": round(float(b.measured_p95_s), 2),
                                 "p95_ratio": round(float(a.measured_p95_s / b.measured_p95_s), 2)}
k["totals"] = {"transfers": int(s.n_ok.sum() + s.n_failed.sum()), "failed": int(s.n_failed.sum()), "cells": int(s.cell.nunique())}
json.dump(k, open(os.path.join(OUT, "emulator_key_numbers.json"), "w"), indent=1)
print(json.dumps(k, indent=1))
