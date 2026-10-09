"""v2.7: the unchanged budgeting comparison on a PUBLISHED CrisisMMD task and split.

Reads data/crisismmd_{task}/records.csv.gz from prepare_crisismmd_task.py, sets the class
list from its provenance and runs run_real_data.run() unchanged (same methods, same three
conditions, same five seeds). The published baselines of Ofli, Alam and Imran (2020) are
test accuracies of text-only, image-only and text+image classifiers on the same test set,
so the anchors are: text_only, image_only and cloud_transmit_all, as 1 - errors / n_test.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import core, run_real_data as R
PUBLISHED = {"informative": {"text_only": 80.8, "image_only": 83.3, "cloud_transmit_all": 84.4},
             "humanitarian": {"text_only": 70.4, "image_only": 76.8, "cloud_transmit_all": 78.4}}
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--pc", choices=["on", "off"], default="off"); a = ap.parse_args()
prov = json.loads((HERE / "data" / f"crisismmd_{a.task}" / "provenance.json").read_text())
core.set_classes(prov["classes"]); core.USE_PRED_CONTRIB = (a.pc == "on")
R.REC = HERE / "data" / f"crisismmd_{a.task}" / "records.csv.gz"
contrib = []; _fit = core.fit_models
def fit_logged(train, cal, seed):
    m = _fit(train, cal, seed); contrib.append({"seed": seed, **{f"rel_{k}": v for k, v in m["pred_contrib"].items()}, **{f"raw_{k}": v for k, v in m["pred_contrib_raw"].items()}}); return m
R.fit_models = fit_logged
df = R.run(log=print); summ, paired = R.summarize(df)
out = HERE / "results_v27"; out.mkdir(exist_ok=True); tag = f"task_{a.task}_pc_{a.pc}"
df.to_csv(out / f"{tag}_seed_level.csv", index=False); summ.to_csv(out / f"{tag}_summary.csv", index=False); paired.to_csv(out / f"{tag}_paired.csv", index=False)
pd.DataFrame(contrib).to_csv(out / f"{tag}_contrib.csv", index=False)
n = int(summ.windows.iloc[0]); summ["accuracy_pct"] = 100 * (1 - summ.errors_mean / n)
summ["published_pct"] = summ.method.map(PUBLISHED[a.task])
print(f"\n{a.task} (K={core.K()}, n_test={n}, pc={a.pc})")
print(summ[["scenario", "method", "errors_mean", "accuracy_pct", "published_pct", "mean_bytes_mean", "p95_latency_s_mean"]].round(2).to_string(index=False))
print(pd.DataFrame(contrib).round(3).to_string(index=False))
