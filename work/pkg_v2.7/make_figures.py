"""Recreate every figure used in the manuscript from results/ (no manual values)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

root = Path(__file__).resolve().parent
out = root / "figures"
out.mkdir(exist_ok=True)
ci = pd.read_csv(root / "results" / "confidence_intervals.csv")
sweep = pd.read_csv(root / "results" / "latency_penalty_sweep.csv")
tsweep = pd.read_csv(root / "results" / "wait_threshold_sweep.csv")

LABEL = {"cloud_transmit_all": "Transmit all (unconstrained)", "fixed_confidence_gate": "Confidence gate",
         "fixed_compression": "Fixed compression", "fixed_rule": "Fixed rule", "random_within_budget": "Random within budget",
         "proposed": "Proposed", "wait_if_uncertain": "Wait if uncertain", "image_only": "Image only", "text_only": "Text only",
         "derived_threshold": "Derived threshold (calibration)", "fixed_rule_admissible": "Fixed rule + admissibility"}
COLOR = {"cloud_transmit_all": "#6B7280", "fixed_confidence_gate": "#9CA3AF", "fixed_compression": "#5B9BD5",
         "fixed_rule": "#C0504D", "random_within_budget": "#D6B656", "proposed": "#17365D", "wait_if_uncertain": "#8064A2",
         "derived_threshold": "#E36C09", "fixed_rule_admissible": "#9BBB59"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def boxes(fname, rows, caption_note=None, figsize=(8.6, 2.3)):
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    pos = {}
    for key, (x, y, w, h, text, hl) in rows["nodes"].items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc="#BDD7EE" if hl else "#EAF1FB", ec="#2F5597", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3, color="#1F2937")
        pos[key] = (x, y, w, h)
    for a, b in rows["edges"]:
        xa, ya, wa, ha = pos[a]; xb, yb, wb, hb = pos[b]
        if abs(ya - yb) < 1:
            start, end = (xa + wa + 0.8, ya + ha / 2), (xb - 0.8, yb + hb / 2)
            if xb < xa:
                start, end = (xa - 0.8, ya + ha / 2), (xb + wb + 0.8, yb + hb / 2)
        else:
            start, end = (xa + wa / 2, ya - 0.8), (xb + wb / 2, yb + hb + 0.8)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#2F5597", lw=1.1))
    if caption_note:
        ax.text(50, 2, caption_note, ha="center", va="bottom", fontsize=6.8, color="#374151")
    fig.savefig(out / fname, bbox_inches="tight", facecolor="white"); plt.close(fig)


boxes("fig_architecture.png", {
    "nodes": {"a": (1, 35, 16, 40, "Edge evidence\nimage | text | sensor", False),
              "b": (21, 35, 16, 40, "Local models\ncalibration, quality,\nseverity estimate", False),
              "c": (41, 35, 18, 40, "Admissibility check\nand shared-budget\naction selection", True),
              "d": (63, 35, 16, 40, "Cloud fusion\nreliability, quality,\nrecency weights", False),
              "e": (83, 35, 16, 40, "Conformal set,\nstatus, review,\nprovenance envelope", False)},
    "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]},
    "Envelopes record observed time, grid cell, hazard tag, source digest, versions, action, bytes, and status rule.")

boxes("fig_decision_flow.png", {
    "nodes": {"a": (1, 58, 17, 30, "New evidence\nfor K concurrent\nincident windows", False),
              "b": (22, 58, 17, 30, "Admissibility:\ncell, time,\nhazard tag", False),
              "c": (43, 58, 17, 30, "Value, byte cost,\ntransfer and\nforecast delay", False),
              "d": (64, 58, 17, 30, "Knapsack over\nthe shared slot\nbudget", True),
              "e": (64, 12, 17, 30, "Transmit selected;\nhold or wait\nfor the rest", False),
              "f": (43, 12, 17, 30, "Fuse admissible\nevidence", False),
              "g": (22, 12, 17, 30, "Conformal set\nand disagreement", False),
              "h": (1, 12, 17, 30, "Assign status;\nroute to review", False)},
    "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")]},
    None, figsize=(8.6, 3.0))

boxes("fig_evaluation_workflow.png", {
    "nodes": {"a": (1, 35, 16, 40, "Synthetic generator\n12 events, 3 modalities,\nmetadata", False),
              "b": (21, 35, 16, 40, "Event-held-out split\ntrain 6, calibrate 2,\ntest 4", False),
              "c": (41, 35, 16, 40, "Nine modeled\nconditions\n(no packet traces)", False),
              "d": (61, 35, 16, 40, "Eight methods + three\ncondition baselines,\nablations, sweeps", False),
              "e": (81, 35, 18, 40, "Errors, bytes, latency,\ncoverage, review load,\nprovenance checks", False)},
    "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]},
    "Conformal thresholds are calibrated per method and condition on calibration events only.")

# Low bandwidth: errors vs bytes with 95% CI across seeds.
low = ci[(ci.scenario == "low_bandwidth") & ci.method.isin(COLOR)]
fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)
for _, r in low.iterrows():
    ax.errorbar(r.mean_bytes_mean, r.errors_mean, yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                fmt="o" if r.method != "cloud_transmit_all" else "s", mfc=COLOR[r.method] if r.method != "cloud_transmit_all" else "white",
                color=COLOR[r.method], ms=7, capsize=3, lw=1)
    off = {"fixed_rule": (4, 22), "fixed_compression": (8, 10), "proposed": (8, -22), "fixed_confidence_gate": (-30, -26),
           "random_within_budget": (8, 4), "cloud_transmit_all": (-60, 14)}[r.method]
    ax.annotate(f"{LABEL[r.method]}, p95 {r.p95_latency_s_mean:.2f} s", (r.mean_bytes_mean, r.errors_mean), xytext=off,
                textcoords="offset points", fontsize=6.6, arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.6))
ax.set_xscale("log"); ax.set_ylim(bottom=0)
ax.set_xlabel("Mean bytes per incident window (log scale)")
ax.set_ylabel("Misclassified windows out of 720 (mean, 95% CI)")
ax.set_title("Low bandwidth: errors against bytes; hollow marker ignores the budget")
ax.grid(True, alpha=.22); fig.tight_layout(); fig.savefig(out / "low_bandwidth_tradeoff.png", facecolor="white"); plt.close(fig)

# Delayed imagery: accuracy-latency frontier traced by the latency penalty, with baselines.
dl = ci[(ci.scenario == "delayed_imagery") & ci.method.isin(["cloud_transmit_all", "fixed_rule", "fixed_compression", "wait_if_uncertain", "derived_threshold"])]
fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)
s = sweep.sort_values("latency_penalty_per_second")
ax.plot(s.p95_latency_s_mean, s.errors_mean, "-o", color=COLOR["proposed"], ms=5, label="Proposed, penalty sweep")
for _, r in s.iterrows():
    if r.latency_penalty_per_second in (0.0, 0.0002, 0.0004, 0.0008, 0.01):
        ax.annotate(f"λ={r.latency_penalty_per_second:g}", (r.p95_latency_s_mean, r.errors_mean), xytext=(-6, -13),
                    textcoords="offset points", fontsize=6.3, color=COLOR["proposed"])
ts = tsweep[tsweep.recency_decay & (tsweep.wait_threshold_s > 0)].sort_values("wait_threshold_s")
ax.plot(ts.p95_latency_s_mean, ts.errors_mean, "--^", color=COLOR["derived_threshold"], ms=4, lw=1, alpha=.8,
        label="Constant threshold, sweep of T")
for _, r in ts.iterrows():
    if r.wait_threshold_s in (300, 900):
        ax.annotate(f"T={r.wait_threshold_s:g} s", (r.p95_latency_s_mean, r.errors_mean), xytext=(4, 5),
                    textcoords="offset points", fontsize=6.0, color=COLOR["derived_threshold"])
jit = {"cloud_transmit_all": 0.93, "fixed_compression": 1.0, "fixed_rule": 1.07, "wait_if_uncertain": 1.0, "derived_threshold": 1.0}
for _, r in dl.iterrows():
    ax.errorbar(r.p95_latency_s_mean * jit[r.method], r.errors_mean, yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                fmt="s", color=COLOR[r.method], ms=6, capsize=3, label=LABEL[r.method])
ax.set_xscale("log"); ax.set_ylim(0, 25)
ax.set_xlabel("95th-percentile modeled latency, s (log scale)")
ax.set_ylabel("Misclassified windows out of 720")
ax.set_title("Delayed imagery: wait-or-act frontier")
ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.grid(True, alpha=.22)
fig.tight_layout(); fig.savefig(out / "delayed_imagery_frontier.png", facecolor="white"); plt.close(fig)

# Tail latency across conditions.
scen = ["normal", "low_bandwidth", "delayed_imagery", "intermittent_outage", "misaligned_evidence"]
meths = ["cloud_transmit_all", "fixed_compression", "fixed_rule", "proposed"]
piv = ci[ci.scenario.isin(scen) & ci.method.isin(meths)].pivot(index="scenario", columns="method", values="p95_latency_s_mean").loc[scen]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=220)
w = .2
for i, m in enumerate(meths):
    ax.bar([x + (i - 1.5) * w for x in range(len(scen))], piv[m], w, label=LABEL[m], color=COLOR[m])
ax.set_yscale("log"); ax.set_ylabel("p95 modeled latency, s (log scale)")
ax.set_xticks(range(len(scen)), ["Normal", "Low\nbandwidth", "Delayed\nimagery", "Intermittent\noutage", "Misaligned\nevidence"])
ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(out / "latency_comparison.png", facecolor="white"); plt.close(fig)
print(out)

# Contested shared budget (option B): errors against per-window budget, pre-specified run.
cb_path = root / "results" / "contested_budget_summary.csv"
if cb_path.exists():
    cb = pd.read_csv(cb_path)
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)
    style = {"all_summary": ("Summaries only", "#9CA3AF", "o"), "fixed_rule_window": ("Fixed rule, per window", "#C0504D", "s"),
             "greedy_uncertainty": ("Greedy by uncertainty (shared)", "#E36C09", "^"),
             "greedy_value": ("Greedy, proposed value (shared)", "#5B9BD5", "v"), "proposed": ("Proposed (exact knapsack)", "#17365D", "D")}
    for m, (lab, col, mk) in style.items():
        g = cb[cb.method == m].sort_values("budget_bytes")
        ax.errorbar(g.budget_bytes, g.errors_mean, yerr=[g.errors_mean - g.errors_ci_low, g.errors_ci_high - g.errors_mean],
                    fmt="-" + mk, color=col, ms=5, capsize=2.5, lw=1.2, label=lab)
    ref = cb[cb.method == "fixed_rule"].errors_mean.mean()
    ax.axhline(ref, color="#6B7280", ls="--", lw=1, label="Fixed rule, ignores budget")
    ax.set_xlabel("Per-window byte budget B (slot budget 6B)")
    ax.set_ylabel("Misclassified windows out of 720 (mean, 95% CI)")
    ax.set_title("Contested shared budget (noise scale 1.75, 10 seeds)")
    ax.legend(frameon=False, fontsize=7); ax.grid(True, alpha=.22)
    fig.tight_layout(); fig.savefig(out / "contested_budget.png", facecolor="white"); plt.close(fig)
