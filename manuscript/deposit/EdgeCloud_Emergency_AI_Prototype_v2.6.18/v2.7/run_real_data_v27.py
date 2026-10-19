"""v2.7: the real-data comparison with the predictive-contribution term switchable.

--pc off reproduces run_real_data.py on the N-class-generalised core (regression test: the
summary must equal the frozen results/real_data_summary.csv); --pc on adds the held-out
predictive-contribution factor to the evidence value. Outputs under results_v27/.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import run_real_data as R

ap = argparse.ArgumentParser(); ap.add_argument("--pc", choices=["on", "off"], required=True); a = ap.parse_args()
core.USE_PRED_CONTRIB = (a.pc == "on")
out = Path(__file__).resolve().parent / "results_v27"; out.mkdir(exist_ok=True)
tag = f"pc_{a.pc}"
# capture the contribution values per seed
contrib = []
_fit = core.fit_models
def fit_logged(train, cal, seed):
    m = _fit(train, cal, seed); contrib.append({"seed": seed, **{f"raw_{k}": v for k, v in m["pred_contrib_raw"].items()},
                                                 **{f"rel_{k}": v for k, v in m["pred_contrib"].items()}}); return m
R.fit_models = fit_logged
df = R.run(log=print)
summ, paired = R.summarize(df)
df.to_csv(out / f"real_data_{tag}_seed_level.csv", index=False); summ.to_csv(out / f"real_data_{tag}_summary.csv", index=False)
paired.to_csv(out / f"real_data_{tag}_paired.csv", index=False); pd.DataFrame(contrib).to_csv(out / f"real_data_{tag}_contrib.csv", index=False)
print(pd.DataFrame(contrib).round(3).to_string(index=False))
print(summ[["scenario", "method", "errors_mean", "mean_bytes_mean", "p95_latency_s_mean"]].round(2).to_string(index=False))
