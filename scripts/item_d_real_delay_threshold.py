"""Item D: the derived constant wait threshold on real data (Reviewer 2.1).

CrisisMMD has no delayed imagery, so an arrival delay is constructed from the
same log-normal family as the synthetic condition and frozen in the protocol:
  70% of test images are late; delay ~ LogNormal(median 900 s, sigma_delay);
  forecast = delay * LogNormal(0, sigma_forecast=0.4).
Delays are drawn once per rotation from a protocol seed and written to
results/ so the exact draw is auditable. T* is derived on calibration windows
only, through the adapter.

Outputs:
  results/real_delay_rev1.csv
  results/real_delay_draws_rev1.csv
  results/itemD_decision.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, paired_t, rng, run_manifest, write_csv  # noqa: E402

METHODS = ["Fixed rule", "Wait if uncertain", "Derived threshold", "Proposed"]


def draw_delays(n: int, proto: dict, seed: int):
    """The package's own delay family, so the real arm matches the synthetic condition.

    A late item waits the median times a lognormal of sigma_delay. An item that is not late
    still waits a little, drawn uniformly, rather than arriving instantly. The forecast is the
    actual delay times a lognormal of sigma_forecast, for every item.
    """
    r = rng(seed)
    late = r.random(n) < proto["late_fraction"]
    lo, hi = proto.get("not_late_delay_uniform_s", [0.0, 0.0])
    delay = np.where(late,
                     proto["median_delay_s"] * r.lognormal(0.0, proto["sigma_delay"], n),
                     r.uniform(lo, hi, n))
    forecast = delay * r.lognormal(0.0, proto["sigma_forecast"], n)
    return delay, forecast


def main():
    proto = load_protocol("itemD_protocol_rev1.json")
    run_manifest(os.path.join(RESULTS, "itemD_manifest.json"), proto)
    rows, draws, decision = [], [], {}
    for rot in range(7):
        split = adapters.load_crisismmd_rotation(rot)
        n_all = len(split.y)
        delay, forecast = draw_delays(n_all, proto, proto["draw_seed_base"] + rot)
        for i in range(n_all):
            draws.append({"rotation": rot, "index": i, "role": split.role[i],
                          "delay_s": delay[i], "forecast_s": forecast[i]})
        tstar = adapters.derive_wait_threshold(split, forecast)
        per_seed = []
        for seed in proto["seeds"]:
            res = adapters.run_real_budgeting(split, "delayed_imagery", seed, METHODS,
                                              delay_seconds=delay, forecast_seconds=forecast)
            per_seed.append(res)
            for m, v in res.items():
                rows.append({"rotation": rot, "seed": seed, "T_star": tstar, "method": m, **v})
        # decision rule components for this rotation
        p95_thr = np.mean([r["Derived threshold"]["p95"] for r in per_seed])
        p95_wiu = np.mean([r["Wait if uncertain"]["p95"] for r in per_seed])
        pt = paired_t([r["Derived threshold"]["errors"] for r in per_seed],
                      [r["Proposed"]["errors"] for r in per_seed])
        decision[int(rot)] = {"T_star": float(tstar), "latency_ok": bool(p95_thr < p95_wiu),
                              "errors_within_interval": bool(pt.lo <= 0 <= pt.hi),
                              "thr_minus_proposed": float(pt.mean), "lo": float(pt.lo), "hi": float(pt.hi)}
        print(f"rot{rot} T*={tstar:.1f}s p95 thr={p95_thr:.1f} wiu={p95_wiu:.1f} "
              f"errors thr-prop={pt.mean:+.1f} [{pt.lo:+.1f},{pt.hi:+.1f}]")
    write_csv(os.path.join(RESULTS, "real_delay_rev1.csv"), rows)
    write_csv(os.path.join(RESULTS, "real_delay_draws_rev1.csv"), draws)
    lat = sum(d["latency_ok"] for d in decision.values())
    err = sum(d["errors_within_interval"] for d in decision.values())
    carries = lat == 7 and err >= proto["decision_rule"]["min_rotations_errors_within"]
    out = {"latency_rotations_passing": int(lat), "error_rotations_within_interval": int(err),
           "threshold_carries_over": bool(carries), "per_rotation": decision}
    json.dump(out, open(os.path.join(RESULTS, "itemD_decision.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("MANUSCRIPT: Section VIII-B 'not tested on real data', Section I-A item 1, "
          "Section IX check 1, Conclusion.")


if __name__ == "__main__":
    main()
