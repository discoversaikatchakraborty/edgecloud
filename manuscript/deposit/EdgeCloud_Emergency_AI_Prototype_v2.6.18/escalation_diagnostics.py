"""How often the severity escalation rule actually fires, written to results/.

Section 6.6 reports that the severity-based escalation rule is nearly inert. That
sentence carried a hand-typed 0.4%, which matches no quantity this package
produces: the rule fires on 0.07% of windows, and even its first clause alone,
ignoring the conjunction, reaches only 0.22%. This script derives both the firing
rate and the rate the generating severity would have produced, so the sentence is
read from a file like every other number in the manuscript.

Outputs: results/escalation_rates.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SEVERITY_ESCALATION_THRESHOLD = 0.70


def main(log=print):
    rec = pd.read_csv(RES / "record_level_results.csv.gz")
    base = rec[rec.ablation.isna()] if rec.ablation.isna().any() else rec
    rows = []
    for method, g in base.groupby("method"):
        n = len(g)
        rows.append({
            "method": method,
            "n_windows": n,
            # R1 is the escalation rule; status_rule records which rule decided the window.
            "escalation_fired_rate": float(g.status_rule.str.startswith("R1").mean()),
            # Its first clause on its own, to show the conjunction is not what makes it rare.
            "severity_est_above_threshold_rate": float(
                g.severity_est.ge(SEVERITY_ESCALATION_THRESHOLD).mean()),
            # What the rule would have fired on given the generating severity.
            "severity_true_above_threshold_rate": float(
                g.severity_true.ge(SEVERITY_ESCALATION_THRESHOLD).mean()),
            "threshold": SEVERITY_ESCALATION_THRESHOLD,
        })
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    out.to_csv(RES / "escalation_rates.csv", index=False)
    log(out[out.method.isin(["proposed", "fixed_rule"])].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
