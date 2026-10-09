"""Figures for the real-data companion paper. Every value read from results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "figures_realdata"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#dcdcda", "#9a9a95"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
                     "figure.dpi": 400, "savefig.dpi": 400})
LABEL = {"image_only": "Image only", "text_only": "Text only", "cloud_transmit_all": "Transmit all",
         "fixed_confidence_gate": "Confidence gate", "fixed_compression": "Fixed compression",
         "fixed_rule": "Fixed rule", "random_within_budget": "Random within budget",
         "greedy_uncertainty": "Greedy by uncertainty", "greedy_value": "Greedy, proposed utility",
         "proposed": "Proposed (exact)"}

# Every method present in the results file must appear in every figure drawn from it.
# A KeyError here used to mean the script died and left the previous figures in place,
# which is indistinguishable from success if stderr is discarded.
def check_complete(frame, mapping, what):
    missing = sorted(set(frame.method.unique()) - set(mapping))
    if missing:
        raise SystemExit(f"{what} omits {missing}, which are present in real_data_summary.csv")


def frame(ax):
    ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.xaxis.grid(False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(0.8); ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3, width=0.8)


def save(fig, name):
    OUT.mkdir(exist_ok=True); fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", (OUT / name).relative_to(ROOT))


S = pd.read_csv(RES / "real_data_summary.csv")
CI = pd.read_csv(RES / "confidence_intervals.csv")
REC = pd.read_csv(ROOT / "data/crisismmd/records.csv.gz")
BASELINE = 494


def fig_tradeoff():
    """Errors against bytes for every method in the results file.

    Ten methods land in a tight cluster between 1,000 and 2,500 bytes, so direct
    labelling collides no matter how the offsets are tuned. A legend carries identity
    and the three outliers keep their labels on the plot.
    """
    d = S[S.scenario == "low_bandwidth"]
    ORDER = ["image_only", "text_only", "fixed_rule", "random_within_budget",
             "fixed_confidence_gate", "greedy_uncertainty", "greedy_value", "proposed",
             "fixed_compression", "cloud_transmit_all"]
    check_complete(d, set(ORDER), "the errors-against-bytes figure")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axhline(BASELINE, color=GREY, lw=1.1, ls=(0, (3, 3)), zorder=1)
    ax.annotate("majority-class baseline, 494", (1.6e5, BASELINE), fontsize=8.2, color=MUTED,
                ha="right", va="bottom")
    # Seven of the ten are contextual and drawn in grey, so grey alone cannot carry
    # identity. Each gets its own marker shape and the legend is then readable.
    MARK = {"image_only": ("*", 17), "text_only": ("s", 7), "fixed_rule": ("^", 8),
            "random_within_budget": ("v", 8), "fixed_confidence_gate": ("p", 8),
            "greedy_uncertainty": ("X", 8), "greedy_value": ("D", 7),
            "proposed": ("o", 14), "fixed_compression": ("P", 8),
            "cloud_transmit_all": ("h", 8)}
    check_complete(d, MARK, "the errors-against-bytes figure")
    d = d.set_index("method").loc[ORDER].reset_index()
    for _, r in d.iterrows():
        best = r.method == "image_only"
        prop = r.method == "proposed"
        greedy = r.method == "greedy_value"
        col = VIOLET if best else (BLUE if (prop or greedy) else GREY)
        mk, ms = MARK[r.method]
        ax.errorbar(r.mean_bytes_mean, r.errors_mean,
                    yerr=[[r.errors_mean - r.errors_ci_low], [r.errors_ci_high - r.errors_mean]],
                    fmt=mk, ms=ms, color=col, ecolor=col, elinewidth=1.0, capsize=2.5,
                    mec="white", mew=0.9, zorder=6 if greedy else (5 if prop else 4), label=LABEL[r.method])
    # Only the three well-separated points keep a label on the plot itself.
    for m, xy, ha in (("image_only", (0, 18), "center"), ("text_only", (12, 0), "left"),
                      ("cloud_transmit_all", (0, 15), "center")):
        r = d[d.method == m].iloc[0]
        ax.annotate(LABEL[m], (r.mean_bytes_mean, r.errors_mean), textcoords="offset points",
                    xytext=xy, ha=ha, va="center", fontsize=8.4,
                    color=VIOLET if m == "image_only" else INK)
    _gv = d[d.method == "greedy_value"].iloc[0]
    _pr = d[d.method == "proposed"].iloc[0]
    if (_gv.errors_mean == _pr.errors_mean) and (_gv.mean_bytes_mean == _pr.mean_bytes_mean):
        ax.annotate("exact and greedy coincide", (_pr.mean_bytes_mean, _pr.errors_mean),
                    textcoords="offset points", xytext=(0, -22), ha="center", va="center",
                    fontsize=8.0, color=BLUE, style="italic")
    ax.set_xscale("log")
    ax.set_xlim(150, 4.0e5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    frame(ax)
    ax.set_xlabel("Bytes transmitted per window (log scale)")
    ax.set_ylabel("Misclassified windows out of 968")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=8.0, labelspacing=0.55, handletextpad=0.5)
    save(fig, "rd_fig1_tradeoff.png")


def fig_fusion_cost():
    fig, ax = plt.subplots(figsize=(5.9, 3.0))
    order = ["image_only", "cloud_transmit_all", "fixed_rule", "fixed_confidence_gate",
             "random_within_budget", "greedy_uncertainty", "greedy_value", "proposed",
             "fixed_compression", "text_only"]
    check_complete(S[S.scenario == "low_bandwidth"], set(order), "the cost-of-fusion figure")
    d = S[S.scenario == "low_bandwidth"].set_index("method").loc[order]
    y = np.arange(len(order))
    base = float(d.loc["image_only"].errors_mean)
    for i, (m, r) in enumerate(d.iterrows()):
        fuses = m not in ("image_only",)
        col = VIOLET if m == "image_only" else (BLUE if m == "proposed" else GREY)
        ax.barh(i, r.errors_mean - 400, left=400, height=0.62, color=col,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{r.errors_mean:.1f}", (r.errors_mean, i), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.4, color=INK)
    ax.axvline(base, color=VIOLET, lw=1.2, ls=(0, (3, 3)), zorder=4)
    ax.axvline(BASELINE, color=GREY, lw=1.1, ls=(0, (2, 2)), zorder=2)
    ax.annotate("image only, 420", (base, len(order) - 0.35), fontsize=8.2, color=VIOLET,
                ha="left", va="center", xytext=(4, 0), textcoords="offset points")
    ax.annotate("majority class, 494", (BASELINE, len(order) - 0.35), fontsize=8.2, color=MUTED,
                ha="right", va="center", xytext=(-4, 0), textcoords="offset points")
    frame(ax); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.yaxis.grid(False)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in order]); ax.invert_yaxis()
    ax.set_xlim(400, 512)
    ax.set_xlabel("Misclassified windows out of 968")
    save(fig, "rd_fig2_fusion_cost.png")


def fig_coverage():
    fig, ax = plt.subplots(figsize=(5.9, 3.0))
    syn = [float(CI[(CI.method == "proposed") & (CI.scenario == s)].iloc[0].conformal_coverage_mean)
           for s in ("normal", "low_bandwidth", "misaligned_evidence")]
    real = [float(S[(S.method == "proposed") & (S.scenario == s)].iloc[0].conformal_coverage_mean)
            for s in ("normal", "low_bandwidth", "misaligned_evidence")]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, syn, w * 0.94, color=GREY, edgecolor="white", lw=0.8, label="Synthetic benchmark", zorder=3)
    ax.bar(x + w / 2, real, w * 0.94, color=ORANGE, edgecolor="white", lw=0.8, label="CrisisMMD", zorder=3)
    for i, (a, b) in enumerate(zip(syn, real)):
        ax.annotate(f"{a:.3f}", (i - w / 2, a), ha="center", va="bottom", fontsize=8.2,
                    color=MUTED, xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{b:.3f}", (i + w / 2, b), ha="center", va="bottom", fontsize=8.2,
                    color=ORANGE, xytext=(0, 2), textcoords="offset points")
    ax.axhline(0.90, color=INK, lw=1.1, ls=(0, (3, 3)), zorder=4)
    ax.annotate("nominal 1 − α = 0.90", (2.45, 0.902), fontsize=8.2, color=INK, ha="right", va="bottom")
    frame(ax)
    ax.set_xticks(x); ax.set_xticklabels(["Normal", "Low bandwidth", "Misaligned evidence"])
    ax.set_ylim(0.6, 0.95); ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "rd_fig3_coverage.png")


def fig_bytes():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (col, assumed, name) in zip(axes, [("image_raw_bytes", (220_000, 620_000), "Image"),
                                               ("text_raw_bytes", (350, 1_800), "Text")]):
        v = REC[REC.image_available][col] if col.startswith("image") else REC[col]
        ax.hist(v, bins=np.logspace(np.log10(max(v.min(), 1)), np.log10(v.max()), 45),
                color=BLUE, edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvspan(*assumed, color=ORANGE, alpha=0.22, zorder=1)
        ax.axvline(v.median(), color=VIOLET, lw=1.4, zorder=4)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        frame(ax)
        ax.set_xlabel(f"{name} payload, bytes (log scale)")
        ax.set_ylabel("Windows")
        ax.annotate(f"assumed\n{assumed[0]:,}–{assumed[1]:,}", (np.sqrt(assumed[0] * assumed[1]),
                    ax.get_ylim()[1] * 0.92), ha="center", va="top", fontsize=8.0, color="#b8471c")
        ax.annotate(f"real median\n{int(v.median()):,}", (v.median(), ax.get_ylim()[1] * 0.62),
                    ha="right", va="center", fontsize=8.2, color=VIOLET,
                    xytext=(-6, 0), textcoords="offset points")
    fig.tight_layout(w_pad=2.0)
    save(fig, "rd_fig4_payload.png")


if __name__ == "__main__":
    fig_tradeoff(); fig_fusion_cost(); fig_coverage(); fig_bytes()
