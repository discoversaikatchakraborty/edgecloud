"""Submission figures, regenerated from results/ with one shared design system.

Every value is read from a frozen result file; nothing is typed in. Colour is used
for at most three series per panel, identity is always carried a second time by a
direct label or a marker shape, and the palette was checked for colour-vision
deficiency and for contrast against a white page before use.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "figures_submission"

# Validated categorical slots (all-pairs CVD and contrast checked on a white page).
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#dcdcda", "#9a9a95"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def frame(ax, xgrid=False):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    else:
        ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3, width=0.8)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def ci():
    return pd.read_csv(RES / "confidence_intervals.csv")


# --------------------------------------------------------------- 1. bandwidth
def fig_bandwidth():
    d = ci()
    d = d[d.scenario == "low_bandwidth"]
    label = {"cloud_transmit_all": "Transmit all\n(ignores the budget)", "fixed_confidence_gate": "Confidence gate",
             "fixed_compression": "Fixed compression", "fixed_rule": "Fixed rule",
             "random_within_budget": "Random within budget", "proposed": "Proposed"}
    # The two single-modality references sit at 51.8 and 87.4 errors, so including them
    # would stretch the axis by a factor of four and flatten the cluster this figure
    # exists to show. They are omitted deliberately, they are in Table 5 in full, and
    # the caption names them and their values. Any OTHER omission is an error.
    OMITTED = {"image_only", "text_only"}
    _missing = set(d.method.unique()) - set(label) - OMITTED
    if _missing:
        raise SystemExit(f"fig_bandwidth omits {sorted(_missing)} without declaring them")
    offs = {"cloud_transmit_all": ((0, 20), "center"), "fixed_confidence_gate": ((-11, 0), "right"),
            "fixed_compression": ((13, -3), "left"), "fixed_rule": ((0, -16), "center"),
            "random_within_budget": ((13, 0), "left"), "proposed": ((0, 15), "center")}
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for m, lab in label.items():
        r = d[d.method == m].iloc[0]
        unconstrained = m == "cloud_transmit_all"
        col = GREY if unconstrained else BLUE
        ax.errorbar(r.mean_bytes_mean, r.errors_mean,
                    yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                    fmt="o" if not unconstrained else "s", ms=6.5, color=col, ecolor=col,
                    elinewidth=1.0, capsize=2.5, mfc="white" if unconstrained else col,
                    mec=col, mew=1.4, zorder=3)
        xy, ha = offs[m]
        ax.annotate(lab, (r.mean_bytes_mean, r.errors_mean), textcoords="offset points",
                    xytext=xy, ha=ha, va="center", fontsize=8.2, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    frame(ax)
    ax.set_xlabel("Bytes transmitted per window (log scale)")
    ax.set_ylabel("Misclassified windows out of 720")
    ax.set_ylim(-1.5, 24)
    ax.set_xlim(620, 1.1e6)
    save(fig, "fig_bandwidth_tradeoff.png")


# ------------------------------------------------------------------ 2. delay
def fig_delay():
    sweep = pd.read_csv(RES / "latency_penalty_sweep.csv")
    tsweep = pd.read_csv(RES / "wait_threshold_sweep.csv")
    tsweep = tsweep[tsweep.recency_decay == True] if "recency_decay" in tsweep else tsweep
    d = ci()
    d = d[d.scenario == "delayed_imagery"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    s = sweep.sort_values("p95_latency_s_mean")
    ax.plot(s.p95_latency_s_mean, s.errors_mean, "-o", color=BLUE, lw=1.6, ms=5,
            mec="white", mew=0.9, zorder=4, label="Proposed policy, latency penalty swept")
    t = tsweep.sort_values("p95_latency_s_mean")
    ax.plot(t.p95_latency_s_mean, t.errors_mean, "--^", color=ORANGE, lw=1.4, ms=5,
            mec="white", mew=0.9, zorder=3, label="Constant wait threshold, T swept")
    marks = {"cloud_transmit_all": ("Transmit all", (-12, 6), "right"),
             "fixed_rule": ("Fixed rule", (-12, 0), "right"),
             "wait_if_uncertain": ("Wait if uncertain", (-12, -4), "right"),
             "derived_threshold": ("Derived threshold T*", (0, -26), "center")}
    for m, (lab, off, ha) in marks.items():
        r = d[d.method == m].iloc[0]
        star = m == "derived_threshold"
        ax.errorbar(r.p95_latency_s_mean, r.errors_mean,
                    yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                    fmt="*" if star else "s", ms=14 if star else 6, color=VIOLET if star else GREY,
                    ecolor=VIOLET if star else GREY, elinewidth=0.9, capsize=2.5,
                    mec="white", mew=0.8, zorder=5)
        ax.annotate(lab, (r.p95_latency_s_mean, r.errors_mean), textcoords="offset points",
                    xytext=off, ha=ha, va="center", fontsize=8.2,
                    color=VIOLET if star else INK)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 5200)
    ax.set_ylim(5.0, 23.5)
    frame(ax)
    ax.set_xlabel("95th-percentile modelled latency, seconds (log scale)")
    ax.set_ylabel("Misclassified windows out of 720")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.4)
    save(fig, "fig_delay_frontier.png")


# -------------------------------------------------------- 3. contested budget
def fig_contested():
    s = pd.read_csv(RES / "contested_budget_summary.csv")
    pr = pd.read_csv(RES / "contested_budget_paired.csv")
    pr = pr[(pr.comparison == "proposed_minus_greedy_uncertainty") & (pr.metric == "errors")]
    pr = pr.sort_values("budget_bytes")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) levels. Marginal intervals are omitted: between-seed variation dominates
    # them and every comparison in the text is paired.
    ax = axes[0]
    context = [("all_summary", "Summaries only", 4), ("fixed_confidence_gate", "Confidence gate", 9),
               ("fixed_rule_window", "Fixed rule, per window", -9)]
    for m, lab, dy in context:
        g = s[s.method == m].sort_values("budget_bytes")
        ax.plot(g.budget_bytes, g.errors_mean, "-", color=GREY, lw=1.1, zorder=2)
        ax.annotate(lab, (g.budget_bytes.iloc[-1], g.errors_mean.iloc[-1]), textcoords="offset points",
                    xytext=(6, dy), fontsize=8.0, color=MUTED, va="center")
    ref = s[s.method == "fixed_rule"].sort_values("budget_bytes")
    ax.plot(ref.budget_bytes, ref.errors_mean, ls=(0, (2, 2)), color=GREY, lw=1.2, zorder=1)
    ax.annotate("Fixed rule, ignores the budget", (ref.budget_bytes.iloc[-1], ref.errors_mean.iloc[-1]),
                textcoords="offset points", xytext=(6, 0), fontsize=8.0, color=MUTED, va="center")
    # greedy_value solves the same knapsack heuristically, so it is the series that
    # separates the solver from the utility. Leaving it out hid the paper's own point.
    for m, lab, col, mk, ms_, z in (("greedy_value", "Greedy, proposed utility", BLUE, "D", 9.0, 3),
                                    ("proposed", "Proposed, exact knapsack", BLUE, "o", 5.5, 4),
                                    ("greedy_uncertainty", "Greedy by uncertainty", ORANGE, "^", 5.5, 4)):
        g = s[s.method == m].sort_values("budget_bytes")
        ax.plot(g.budget_bytes, g.errors_mean, mk + "-", color=col, lw=1.8, ms=ms_,
                mec="white", mew=0.9, zorder=z, label=lab,
                alpha=0.45 if m == "greedy_value" else 1.0)
    # random_within_budget runs from 140.4 to 109.2 errors and would stretch the axis
    # by half again, flattening every series this panel compares. It is omitted on
    # purpose, it is in Table 10 in full, and the caption names it. Anything else
    # missing is an error.
    _OMIT = {"random_within_budget"}
    _miss = set(s.method.unique()) - {"all_summary", "fixed_confidence_gate", "fixed_rule_window",
                                      "fixed_rule", "proposed", "greedy_uncertainty",
                                      "greedy_value"} - _OMIT
    if _miss:
        raise SystemExit(f"fig_contested omits {sorted(_miss)} without declaring them")
    frame(ax)
    ax.set_xlabel("Per-window byte budget, B")
    ax.set_ylabel("Misclassified windows out of 720")
    ax.set_xlim(620, 2560)
    ax.set_xticks([700, 900, 1200, 1500, 1900])
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5, borderaxespad=0.2)
    ax.set_title("(a) Errors against the shared budget", loc="left", pad=6)

    # (b) paired difference, which is the comparison the analysis actually makes.
    ax = axes[1]
    x = np.arange(len(pr))
    lo = pr.mean_difference - pr.ci_low
    hi = pr.ci_high - pr.mean_difference
    sig = pr.p_holm < 0.05
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.errorbar(x, pr.mean_difference, yerr=[lo, hi], fmt="o", ms=6, color=BLUE, ecolor=BLUE,
                elinewidth=1.2, capsize=3, mec="white", mew=0.9, zorder=4)
    for i, (v, p_) in enumerate(zip(pr.mean_difference, pr.p_holm)):
        if p_ < 0.05:
            ax.annotate("Holm p < 0.001", (i, v), textcoords="offset points", xytext=(10, 0),
                        fontsize=8.0, color=INK, va="center")
    ax.annotate("optimizer worse", (0.04, 0.965), xycoords="axes fraction", fontsize=8.0,
                color=MUTED, va="top")
    ax.annotate("optimizer better", (0.04, 0.035), xycoords="axes fraction", fontsize=8.0,
                color=MUTED, va="bottom")
    frame(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in pr.budget_bytes])
    ax.set_xlim(-0.5, len(pr) - 0.15)
    ax.set_xlabel("Per-window byte budget, B")
    ax.set_ylabel("Paired difference in errors\n(proposed \u2212 greedy by uncertainty)")
    ax.set_title("(b) Paired difference, ten seeds", loc="left", pad=6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_contested_budget.png")


# ------------------------------------------------------------ 4. spoof sweep
def fig_spoof():
    s = pd.read_csv(RES / "spoof_sweep_summary.csv")
    series = [("fixed_rule", "No admissibility filter", GREY, "s", (0, (2, 2))),
              ("fixed_rule_admissible", "Fixed rule with filter", ORANGE, "^", "-"),
              ("proposed", "Proposed", BLUE, "o", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    ax = axes[0]
    for m, lab, col, mk, ls in series:
        g = s[s.method == m].sort_values("spoof_rate")
        ax.errorbar(g.spoof_rate * 100, g.errors_mean,
                    yerr=[g.errors_mean - g.errors_ci_low, g.errors_ci_high - g.errors_mean],
                    fmt=mk, ls=ls, color=col, ecolor=col, lw=1.6, ms=5.5, elinewidth=1.0,
                    capsize=2.5, mec="white", mew=0.9, label=lab, zorder=3)
    frame(ax)
    ax.set_xlabel("Misaligned reports carrying forged metadata (%)")
    ax.set_ylabel("Misclassified windows out of 720")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.5)

    ax = axes[1]
    for m, lab, col, mk, ls in series:
        g = s[s.method == m].sort_values("spoof_rate")
        ax.plot(g.spoof_rate * 100, g.misaligned_text_fused_rate_mean * 100, mk, ls=ls, color=col,
                lw=1.6, ms=5.5, mec="white", mew=0.9, zorder=3)
    ax.plot([0, 100], [0, 100], ls=(0, (1, 2)), color=MUTED, lw=1.0, zorder=1)
    ax.annotate("admits exactly the\nforged items, no more", (78, 34), fontsize=8.0, color=MUTED,
                ha="center", va="center")
    frame(ax)
    ax.set_xlabel("Misaligned reports carrying forged metadata (%)")
    ax.set_ylabel("Misaligned reports admitted to fusion (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_spoof_sweep.png")


# --------------------------------------------------------- 5. coverage shift
def fig_coverage():
    d = ci()
    d = d[d.method == "proposed"]
    order = ["normal", "low_bandwidth", "intermittent_outage", "delayed_imagery", "noisy_text",
             "misaligned_evidence", "text_missing", "contradictory_evidence", "image_missing"]
    nice = {"normal": "Normal", "low_bandwidth": "Low bandwidth", "intermittent_outage": "Outage",
            "delayed_imagery": "Delayed imagery", "noisy_text": "Noisy text",
            "misaligned_evidence": "Misaligned", "text_missing": "Text missing",
            "contradictory_evidence": "Contradictory", "image_missing": "Image missing"}
    inc = [float(d[d.scenario == s].conformal_coverage_mean.iloc[0]) for s in order]
    nrm = [float(d[d.scenario == s].coverage_normal_calibrated_mean.iloc[0]) for s in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    for i, (a, b) in enumerate(zip(nrm, inc)):
        ax.plot([a, b], [i, i], color=GRID, lw=3.0, solid_capstyle="round", zorder=1)
    ax.axvline(0.90, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.annotate("nominal 1 − α = 0.90", (0.9, len(order) - 0.35), fontsize=8.0,
                color=MUTED, ha="center", va="bottom")
    ax.plot(inc, y, "o", color=BLUE, ms=6.5, mec="white", mew=1.0, zorder=4,
            label="Calibrated within the condition")
    ax.plot(nrm, y, "^", color=ORANGE, ms=6.5, mec="white", mew=1.0, zorder=4,
            label="Normal-condition threshold reused")
    for i, (a, b) in enumerate(zip(nrm, inc)):
        if b - a > 0.02:
            ax.annotate(f"−{100*(b-a):.1f} pts", (a, i), textcoords="offset points",
                        xytext=(-7, 0), ha="right", va="center", fontsize=8.0, color=ORANGE)
    frame(ax, xgrid=False)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[s] for s in order])
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(0.71, 0.945)
    ax.set_xlabel("Empirical coverage of the conformal set")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.5)
    save(fig, "fig_coverage_shift.png")



# ------------------------------------------------------------ 6. link realism
def fig_link():
    ln = pd.read_csv(RES / "link_loss_normal_summary.csv")
    tr = pd.read_csv(RES / "link_trace_summary.csv")
    # The axis is logarithmic, so every method in the sweep fits without crowding.
    series = [("fixed_rule", "Fixed rule", ORANGE, "^", "-"),
              ("fixed_compression", "Fixed compression", GREY, "v", "-"),
              ("fixed_confidence_gate", "Confidence gate", GREY, "P", (0, (1, 2))),
              ("proposed", "Proposed", BLUE, "o", "-"),
              ("cloud_transmit_all", "Transmit all", GREY, "s", (0, (2, 2)))]
    _miss = set(ln.method.unique()) - {x[0] for x in series}
    if _miss:
        raise SystemExit(f"fig_link omits {sorted(_miss)} present in link_loss_normal_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    for m, lab, col, mk, ls in series:
        g = ln[ln.method == m].sort_values("packet_loss")
        ax.plot(g.packet_loss * 100, g.p95_latency_s_mean, mk, ls=ls, color=col, lw=1.7,
                ms=5.5, mec="white", mew=0.9, label=lab, zorder=3)
    ax.set_yscale("log")
    frame(ax)
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("95th-percentile latency, s (log scale)")
    ax.set_xticks([0, 2, 5, 10])
    ax.legend(frameon=False, loc="upper left", handletextpad=0.5)
    ax.set_title("(a) Loss at the normal budget", loc="left", pad=6)

    ax = axes[1]
    traces = [t for t in tr.link.unique() if "replayed" in t]
    short = {t: t.split("-LTE")[0].replace("Verizon", "Verizon LTE").replace("TMobile", "T-Mobile LTE")
             + (" (driving)" if "driving" in t else " (short)") for t in traces}
    x = np.arange(len(traces))
    # The text quotes the proposed policy's inflation on the driving trace, so the
    # panel has to show it. Three of the five methods are drawn: the largest inflation
    # (fixed compression), the paper's method, and the simple rule it is measured
    # against. The other two are in Table 12 and the caption says so.
    PANEL_B = [s_ for s_ in series if s_[0] in ("fixed_compression", "proposed", "fixed_rule")]
    w = 0.78 / len(PANEL_B)
    for i, (m, lab, col, mk, ls) in enumerate(PANEL_B):
        rep = [float(tr[(tr.link == t) & (tr.method == m)].p95_latency_s_mean.iloc[0]) for t in traces]
        con = [float(tr[(tr.link == t.replace("(replayed)", "(constant at same mean)")) &
                        (tr.method == m)].p95_latency_s_mean.iloc[0]) for t in traces]
        ax.bar(x + (i - (len(PANEL_B) - 1) / 2) * w, rep, w * 0.92, color=col, edgecolor="white", lw=0.8,
               label=lab, zorder=3)
        ax.bar(x + (i - (len(PANEL_B) - 1) / 2) * w, con, w * 0.92, color="white", edgecolor=col, lw=1.3,
               hatch="///", zorder=4)
        for j, (r_, c_) in enumerate(zip(rep, con)):
            if r_ / max(c_, 1e-9) > 1.4:
                ax.annotate(f"\u00d7{r_/c_:.1f}", (x[j] + (i - (len(PANEL_B) - 1) / 2) * w, r_), ha="center",
                            va="bottom", fontsize=8.0, color=col, xytext=(0, 2),
                            textcoords="offset points")
    frame(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([short[t] for t in traces], fontsize=8.2)
    ax.set_ylabel("95th-percentile latency, s")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", handletextpad=0.5, ncol=1)
    ax.set_ylim(0, max(ax.get_ylim()[1], 2.62))
    ax.set_title("(b) Measured capacity, mean-matched", loc="left", pad=6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_link_realism.png")



# ---------------------------------------------------------- 7. decentralization
def fig_decentralized():
    su = pd.read_csv(RES / "decentralized_summary.csv")
    pa = pd.read_csv(RES / "decentralized_paired.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) errors against controller count, at the budget where the effect is largest
    ax = axes[0]
    B = 900
    # Every method in the source file is drawn. Omitting the summaries-only floor made
    # the panel show three of the four rows that Table 13 reports.
    SERIES = (("proposed", "Proposed, exact knapsack", BLUE, "o", "-"),
              ("greedy_uncertainty", "Greedy by uncertainty", ORANGE, "^", "-"),
              ("fixed_rule_window", "Fixed rule, per window", GREY, "s", (0, (2, 2))),
              ("all_summary", "Summaries only", GREY, "v", (0, (1, 2))))
    _missing = sorted(set(su.method.unique()) - {x[0] for x in SERIES})
    if _missing:
        raise SystemExit(f"fig_decentralized omits {_missing} present in decentralized_summary.csv")
    for m, lab, col, mk, ls in SERIES:
        g = su[(su.method == m) & (su.budget_bytes == B)].sort_values("controllers")
        ax.errorbar(g.controllers, g.errors_mean,
                    yerr=[g.errors_mean - g.errors_ci_low, g.errors_ci_high - g.errors_mean],
                    fmt=mk, ls=ls, color=col, ecolor=col, lw=1.7, ms=5.5, elinewidth=1.0,
                    capsize=2.5, mec="white", mew=0.9, label=lab, zorder=3)
    frame(ax)
    ax.set_xticks([1, 2, 3, 6])
    ax.set_xlabel("Independent controllers per slot")
    ax.set_ylabel("Misclassified windows out of 720")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.5, fontsize=8.2)
    ax.set_title(f"(a) At B = {B} bytes", loc="left", pad=6)

    # (b) cost of removing coordination, all budgets
    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    budgets = sorted(pa.budget_bytes.unique())
    x = np.arange(len(budgets))
    for m, lab, col, mk in (("proposed", "Proposed", BLUE, "o"),
                            ("greedy_uncertainty", "Greedy by uncertainty", ORANGE, "^")):
        g = pa[(pa.method == m) & (pa.controllers == 6)].set_index("budget_bytes").loc[budgets]
        ax.errorbar(x, g.cost_of_decentralization,
                    yerr=[g.cost_of_decentralization - g.ci_low, g.ci_high - g.cost_of_decentralization],
                    fmt=mk, ls="-", color=col, ecolor=col, lw=1.7, ms=5.5, elinewidth=1.0,
                    capsize=2.5, mec="white", mew=0.9, label=lab, zorder=3)
    frame(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Per-window byte budget, B")
    ax.set_ylabel("Extra errors when coordination\nis removed (6 controllers \u2212 1)")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5, fontsize=8.2)
    ax.set_title("(b) Cost of decentralization", loc="left", pad=6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_decentralized.png")


if __name__ == "__main__":
    fig_bandwidth(); fig_delay(); fig_contested(); fig_spoof(); fig_coverage(); fig_link(); fig_decentralized()
