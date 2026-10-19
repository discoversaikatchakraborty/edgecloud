"""Per-modality classifier accuracy on the real workload, written to results/.

Section 6.12 states how well each real modality predicts the label. Those figures
were prose for one draft, which is exactly the condition this package exists to
prevent: a number in the manuscript that no file produces. This script derives them
from the same records, split and fitting procedure the policy comparison uses, and
writes results/real_data_representation.csv for the builder to read.

The quantity is the raw per-record accuracy of each modality's own classifier on the
held-out events, which is not the same as that modality's error count in the policy
comparison: the policy decides per window under a budget, aggregates through fusion
and conformal sets, and can decline to transmit. Both are reported so neither is
mistaken for the other.

Outputs: results/real_data_representation.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import core
from core import CONFIG, MODALITIES, fit_models
from run_real_data import build_frame

ROOT = Path(__file__).resolve().parent
SEEDS = CONFIG["seeds"]


def main(log=print):
    d = build_frame()
    train = d[d.split == "train"].drop(columns="split").reset_index(drop=True)
    cal = d[d.split == "calibration"].drop(columns="split").reset_index(drop=True)
    test = d[d.split == "test"].drop(columns="split").reset_index(drop=True)
    y = test.y.to_numpy()
    n = len(y)
    majority = float(np.bincount(y).max()) / n

    rows = []
    for seed in SEEDS:
        models = fit_models(train, cal, seed)
        for m in MODALITIES:
            key = f"{m}_local_p"
            if key not in models and not hasattr(models.get(m, None), "predict"):
                pass
            p = core.stack(core.attach_probabilities(test, models), key)
            rows.append({"seed": seed, "modality": m,
                         "accuracy": float((p.argmax(axis=1) == y).mean()),
                         "majority_baseline": majority, "n_test": n})
    df = pd.DataFrame(rows)
    out = (df.groupby("modality")
             .agg(accuracy_mean=("accuracy", "mean"),
                  accuracy_min=("accuracy", "min"),
                  accuracy_max=("accuracy", "max"),
                  majority_baseline=("majority_baseline", "first"),
                  n_test=("n_test", "first"),
                  n_seeds=("seed", "size"))
             .reset_index())
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "real_data_representation.csv", index=False)
    log(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
