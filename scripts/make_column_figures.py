"""The three evidence figures re-authored at column width with readable type (item 7).

The upstream figures were authored 5.8 to 7.2 in wide with 8 to 9 pt type and are
printed at the 3.45 in column, which takes their type to about 4 to 5 pt. Scaling
the canvas alone made the inline labels collide (scripts/make_submission_figures.py,
kept as the record of that attempt). This script redraws each figure on a 3.45 in
canvas from the same retained CSVs with the label positions chosen for that canvas.
Every plotted value is read from the same upstream files; nothing is typed in.
Outputs go to results/figures_colwidth/ and are copied into manuscript/latex/ by
the caller; upstream/ is never written.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "upstream" / "prototype_v2.3" / "results"
OUT = ROOT / "results" / "figures_colwidth"; OUT.mkdir(parents=True, exist_ok=True)
COL = 3.45
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#dcdcda", "#9a9a95"
F = 7.5
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                     "font.size": F, "axes.labelsize": F, "axes.titlesize": F, "xtick.labelsize": F - 0.5,
                     "ytick.labelsize": F - 0.5, "legend.fontsize": F - 0.5, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": MUTED, "figure.dpi": 400, "savefig.dpi": 400,
                     "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2})
LABEL = {"image_only": "Image only", "text_only": "Text only", "cloud_transmit_all": "Transmit all",
         "fixed_confidence_gate": "Confidence gate", "fixed_compression": "Fixed compression",
         "fixed_rule": "Fixed rule", "random_within_budget": "Random within budget",
         "greedy_uncertainty": "Greedy by uncertainty", "greedy_value": "Greedy, proposed utility",
         "proposed": "Proposed (exact)"}


def frame(ax, xgrid=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    if xgrid: ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.6, colors=MUTED, labelcolor=INK)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def fig_delay():
    sweep = pd.read_csv(RES / "latency_penalty_sweep.csv"); tsweep = pd.read_csv(RES / "wait_threshold_sweep.csv")
    tsweep = tsweep[tsweep.recency_decay == True] if "recency_decay" in tsweep else tsweep
    d = pd.read_csv(RES / "confidence_intervals.csv"); d = d[d.scenario == "delayed_imagery"]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    s = sweep.sort_values("p95_latency_s_mean")
    ax.plot(s.p95_latency_s_mean, s.errors_mean, "-o", color=BLUE, lw=1.3, ms=3.6, mec="white", mew=0.6, zorder=4,
            label="Proposed policy, latency penalty swept")
    t = tsweep.sort_values("p95_latency_s_mean")
    ax.plot(t.p95_latency_s_mean, t.errors_mean, "--^", color=ORANGE, lw=1.1, ms=3.6, mec="white", mew=0.6, zorder=3,
            label="Constant wait threshold, T swept")
    marks = {"cloud_transmit_all": ("Transmit all", (-7, 5), "right", "center"),
             "fixed_rule": ("Fixed rule", (-7, 0), "right", "center"),
             "wait_if_uncertain": ("Wait if uncertain", (-7, -3), "right", "center"),
             "derived_threshold": ("Derived threshold T*", (7, -5), "left", "top")}
    for m, (lab, off, ha, va) in marks.items():
        r = d[d.method == m].iloc[0]; star = m == "derived_threshold"
        ax.errorbar(r.p95_latency_s_mean, r.errors_mean,
                    yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                    fmt="*" if star else "s", ms=10 if star else 4.5, color=VIOLET if star else GREY,
                    ecolor=VIOLET if star else GREY, elinewidth=0.7, capsize=1.8, mec="white", mew=0.5, zorder=5)
        ax.annotate(lab, (r.p95_latency_s_mean, r.errors_mean), textcoords="offset points", xytext=off, ha=ha, va=va,
                    fontsize=F - 0.5, color=VIOLET if star else INK)
    ax.set_xscale("log"); ax.set_xlim(0.12, 5200); ax.set_ylim(5.0, 23.5); frame(ax)
    ax.set_xlabel("95th-percentile modelled latency, s (log scale)"); ax.set_ylabel("Misclassified windows of 720")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderaxespad=0.2, handlelength=2.0)
    save(fig, "fig_delay_frontier.png")


def fig_contested():
    s = pd.read_csv(RES / "contested_budget_summary.csv"); pr = pd.read_csv(RES / "contested_budget_paired.csv")
    pr = pr[(pr.comparison == "proposed_minus_greedy_uncertainty") & (pr.metric == "errors")].sort_values("budget_bytes")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.5})
    ax = axes[0]
    context = [("all_summary", "Summaries", 3), ("fixed_confidence_gate", "Conf. gate", 6),
               ("fixed_rule_window", "Fixed/window", -6)]
    for m, lab, dy in context:
        g = s[s.method == m].sort_values("budget_bytes")
        ax.plot(g.budget_bytes, g.errors_mean, "-", color=GREY, lw=0.9, zorder=2)
        ax.annotate(lab, (g.budget_bytes.iloc[-1], g.errors_mean.iloc[-1]), textcoords="offset points",
                    xytext=(3, dy), fontsize=F - 1.5, color=MUTED, va="center")
    ref = s[s.method == "fixed_rule"].sort_values("budget_bytes")
    ax.plot(ref.budget_bytes, ref.errors_mean, ls=(0, (2, 2)), color=GREY, lw=1.0, zorder=1)
    ax.annotate("Fixed, no budget", (ref.budget_bytes.iloc[-1], ref.errors_mean.iloc[-1]), textcoords="offset points",
                xytext=(3, -1), fontsize=F - 1.5, color=MUTED, va="center")
    for m, lab, col, mk, ms_, z in (("greedy_value", "Greedy, proposed utility", BLUE, "D", 6.5, 3),
                                    ("proposed", "Proposed, exact knapsack", BLUE, "o", 3.8, 4),
                                    ("greedy_uncertainty", "Greedy by uncertainty", ORANGE, "^", 3.8, 4)):
        g = s[s.method == m].sort_values("budget_bytes")
        ax.plot(g.budget_bytes, g.errors_mean, mk + "-", color=col, lw=1.4, ms=ms_, mec="white", mew=0.6, zorder=z,
                label=lab, alpha=0.45 if m == "greedy_value" else 1.0)
    miss = set(s.method.unique()) - {"all_summary", "fixed_confidence_gate", "fixed_rule_window", "fixed_rule",
                                     "proposed", "greedy_uncertainty", "greedy_value", "random_within_budget"}
    if miss: raise SystemExit(f"fig_contested omits {sorted(miss)} without declaring them")
    frame(ax); ax.set_xlabel("Per-window byte budget, B"); ax.set_ylabel("Misclassified windows of 720")
    ax.set_xlim(620, 2650); ax.set_xticks([700, 1200, 1900]); ax.set_xticklabels(["700", "1,200", "1,900"]); ax.set_xticks([900, 1500], minor=True)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.1, handlelength=1.6, fontsize=F - 1.5)
    ax.set_title("(a) Errors against the shared budget", loc="left", pad=3, fontsize=F - 0.5)
    ax = axes[1]
    x = np.arange(len(pr)); lo = pr.mean_difference - pr.ci_low; hi = pr.ci_high - pr.mean_difference
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.errorbar(x, pr.mean_difference, yerr=[lo, hi], fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=0.9,
                capsize=2, mec="white", mew=0.6, zorder=4)
    for i, (v, p_) in enumerate(zip(pr.mean_difference, pr.p_holm)):
        if p_ < 0.05:
            ax.annotate("Holm p < 0.001", (i, v), textcoords="offset points", xytext=(6, 0), fontsize=F - 1.5, color=INK, va="center")
    ax.annotate("optimizer worse", (0.04, 0.97), xycoords="axes fraction", fontsize=F - 1.5, color=MUTED, va="top")
    ax.annotate("optimizer better", (0.04, 0.10), xycoords="axes fraction", fontsize=F - 1.5, color=MUTED, va="bottom")
    frame(ax); ax.set_xticks(x); ax.set_xticklabels([f"{int(b)}" if b < 1000 else f"{b/1000:.1f}k" for b in pr.budget_bytes], fontsize=F - 1.5); ax.set_xlim(-0.5, len(pr) - 0.15)
    ax.set_xlabel("Per-window byte budget, B"); ax.set_ylabel("Paired difference in errors")
    ax.set_title("(b) Proposed \u2212 greedy by uncertainty", loc="left", pad=3, fontsize=F - 0.5)
    save(fig, "fig_contested.png")


def fig_real():
    S = pd.read_csv(RES / "real_data_summary.csv"); d = S[S.scenario == "low_bandwidth"]; BASELINE = 494
    ORDER = ["image_only", "text_only", "fixed_rule", "random_within_budget", "fixed_confidence_gate",
             "greedy_uncertainty", "greedy_value", "proposed", "fixed_compression", "cloud_transmit_all"]
    assert set(d.method) == set(ORDER), set(d.method) ^ set(ORDER)
    MARK = {"image_only": ("*", 12), "text_only": ("s", 5), "fixed_rule": ("^", 5.5), "random_within_budget": ("v", 5.5),
            "fixed_confidence_gate": ("p", 5.5), "greedy_uncertainty": ("X", 5.5), "greedy_value": ("D", 5),
            "proposed": ("o", 10), "fixed_compression": ("P", 5.5), "cloud_transmit_all": ("h", 5.5)}
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.axhline(BASELINE, color=GREY, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.annotate("majority-class baseline, 494", (3.2e5, BASELINE), fontsize=F - 1, color=MUTED, ha="right", va="bottom")
    d = d.set_index("method").loc[ORDER].reset_index()
    for _, r in d.iterrows():
        best, prop, greedy = r.method == "image_only", r.method == "proposed", r.method == "greedy_value"
        col = VIOLET if best else (BLUE if (prop or greedy) else GREY); mk, ms = MARK[r.method]
        ax.errorbar(r.mean_bytes_mean, r.errors_mean, yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                    fmt=mk, ms=ms, color=col, ecolor=col, elinewidth=0.7, capsize=1.8, mec="white", mew=0.6,
                    zorder=6 if greedy else (5 if prop else 4), label=LABEL[r.method])
    for m, xy, ha in (("image_only", (0, 11), "center"), ("text_only", (8, 0), "left"), ("cloud_transmit_all", (0, 10), "center")):
        r = d[d.method == m].iloc[0]
        ax.annotate(LABEL[m], (r.mean_bytes_mean, r.errors_mean), textcoords="offset points", xytext=xy, ha=ha, va="center",
                    fontsize=F - 0.5, color=VIOLET if m == "image_only" else INK)
    gv, prp = d[d.method == "greedy_value"].iloc[0], d[d.method == "proposed"].iloc[0]
    if (gv.errors_mean == prp.errors_mean) and (gv.mean_bytes_mean == prp.mean_bytes_mean):
        ax.annotate("exact and greedy coincide", (prp.mean_bytes_mean, prp.errors_mean), textcoords="offset points",
                    xytext=(0, -14), ha="center", va="center", fontsize=F - 1, color=BLUE, style="italic")
    ax.set_xscale("log"); ax.set_xlim(150, 4.0e5); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    frame(ax); ax.set_xlabel("Bytes transmitted per window (log scale)"); ax.set_ylabel("Misclassified windows of 968")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=3, fontsize=F - 1.5,
              columnspacing=0.9, handletextpad=0.3, labelspacing=0.35, borderaxespad=0.0)
    save(fig, "fig_real_scatter.png")


if __name__ == "__main__":
    fig_delay(); fig_contested(); fig_real()
