"""Measured Mahimahi transfer latency against the package's modelled latency (item F).

For every transfer in results/emulator/transfers.csv the modelled latency is the
package's own expression for one record on an otherwise idle link:

    serialization_time(bytes, scenario, offset) + FIXED_OVERHEAD_S
        + base_latency_s + retransmission_stall(bytes, scenario)

evaluated with core.py unchanged, at the trace offset the client recorded. The
measured latency is the client's connect-to-acknowledgement time. Both are
written per transfer, then summarised per cell and payload with the ratio
measured / modelled and a 95% t-interval over repeats.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "upstream", "prototype_v2.3")
sys.path.insert(0, PKG)
import core  # noqa: E402
from core import Scenario, serialization_time, retransmission_stall, FIXED_OVERHEAD_S  # noqa: E402

OUT = os.path.join(ROOT, "results", "emulator")
df = pd.read_csv(os.path.join(OUT, "transfers.csv"))
df = df[df.payload_bytes > 0].copy()
base = {s.name: s for s in core.SCENARIOS}


def scenario_for(row):
    kind, val = row.link.split(":", 1)
    rtt = row.delay_ms * 2 / 1000.0
    if kind == "trace":
        return Scenario(name="emu", bandwidth_bps=1.0, base_latency_s=rtt, budget_bytes=1,
                        packet_loss=row.loss, link_trace=val)
    return Scenario(name="emu", bandwidth_bps=float(val), base_latency_s=rtt, budget_bytes=1,
                    packet_loss=row.loss)


model = []
for _, r in df.iterrows():
    sc = scenario_for(r)
    b = np.array([float(r.payload_bytes)])
    ser = serialization_time(b, sc, int(round(r.offset_s * 1000)))[0]
    stall = retransmission_stall(b, sc)[0]
    model.append(ser + FIXED_OVERHEAD_S + sc.base_latency_s + stall)
df["modelled_s"] = model
df["measured_s"] = df.ack_s
df["ratio"] = df.measured_s / df.modelled_s
df.to_csv(os.path.join(OUT, "transfers_with_model.csv"), index=False)

rows = []
for (cell, n), g in df.groupby(["cell", "payload_bytes"], sort=False):
    ok = g[g.ok == 1]
    k = len(ok)
    t = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan
    def ci(x):
        mu, sd = x.mean(), x.std(ddof=1) if k > 1 else np.nan
        h = t * sd / np.sqrt(k) if k > 1 else np.nan
        return mu, mu - h, mu + h
    mm = ci(ok.measured_s); mr = ci(ok.ratio)
    rows.append({"cell": cell, "link": g.link.iloc[0], "delay_ms": g.delay_ms.iloc[0], "loss": g.loss.iloc[0],
                 "payload_bytes": n, "n_ok": k, "n_failed": int((g.ok == 0).sum()),
                 "modelled_mean_s": ok.modelled_s.mean(), "measured_mean_s": mm[0],
                 "measured_ci_low": mm[1], "measured_ci_high": mm[2],
                 "measured_p95_s": ok.measured_s.quantile(0.95),
                 "ratio_mean": mr[0], "ratio_ci_low": mr[1], "ratio_ci_high": mr[2]})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "emulator_summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(summ[["cell", "payload_bytes", "n_ok", "n_failed", "modelled_mean_s", "measured_mean_s",
            "measured_p95_s", "ratio_mean", "ratio_ci_low", "ratio_ci_high"]].round(4).to_string(index=False))
