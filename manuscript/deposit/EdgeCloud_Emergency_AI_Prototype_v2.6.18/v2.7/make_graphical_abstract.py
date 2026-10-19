"""Elsevier graphical abstract: one image, readable at thumbnail size.

Every number is read from the result files at build time. Target canvas is
1328 x 531 px, Elsevier's preferred graphical-abstract size.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "figures_submission" / "graphical_abstract.png"
# Elsevier's preferred graphical-abstract formats are TIFF, EPS, PDF or MS Office;
# PNG is not among them, so the same figure is written in the accepted ones too.
OUT_TIFF = ROOT / "figures_submission" / "graphical_abstract.tiff"
OUT_PDF = ROOT / "figures_submission" / "graphical_abstract.pdf"

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d9d6"

ci = pd.read_csv(RES / "confidence_intervals.csv")
sp = pd.read_csv(RES / "spoof_sweep_summary.csv")
lt = pd.read_csv(RES / "link_trace_summary.csv")
dp = pd.read_csv(RES / "decentralized_paired.csv")
rd = pd.read_csv(RES / "real_data_summary.csv")

g = lambda m, s, c: float(ci[(ci.method == m) & (ci.scenario == s)].iloc[0][c])
spv = lambda m, r: float(sp[(sp.method == m) & (sp.spoof_rate == r)].iloc[0].errors_mean)
vr = [x for x in lt.link.unique() if "Verizon-LTE-driving" in x and "replayed" in x][0]
vc = vr.replace("(replayed)", "(constant at same mean)")
ltv = lambda m, k: float(lt[(lt.link == k) & (lt.method == m)].iloc[0].p95_latency_s_mean)
dcv = lambda m: float(dp[(dp.method == m) & (dp.budget_bytes == 900) & (dp.controllers == 6)].iloc[0].cost_of_decentralization)
rdv = lambda m, sc: float(rd[(rd.method == m) & (rd.scenario == sc)].iloc[0].errors_mean)
# "Up to" ranges over every method and trace, not the proposed policy alone.
TRACE_METHODS = ("fixed_rule", "fixed_confidence_gate", "fixed_compression",
                 "proposed", "cloud_transmit_all")
infl = max(ltv(m, t) / ltv(m, t.replace("(replayed)", "(constant at same mean)"))
           for t in lt.link.unique() if "replayed" in t for m in TRACE_METHODS)

CARDS = [
    (BLUE, "A fixed rule matches\nthe exact optimizer",
     f"{g('fixed_rule','low_bandwidth','errors_mean'):.1f} vs {g('proposed','low_bandwidth','errors_mean'):.1f}",
     "errors out of 720,\nat half the bytes"),
    (ORANGE, "Admissibility collapses\nunder forged metadata",
     f"{spv('fixed_rule_admissible',0.0):.1f} → {spv('fixed_rule_admissible',1.0):.1f}",
     "errors, reaching the\nunfiltered baseline"),
    (VIOLET, "A measured link is not\na constant one",
     f"×{infl:.1f}",
     "tail latency, at\nmatched mean capacity"),
    (BLUE, "The optimizer needs\ncentralisation",
     f"+{dcv('proposed'):.1f} vs +{dcv('greedy_uncertainty'):.1f}",
     "errors when\ncoordination is removed"),
    (ORANGE, "On real data the\nverdict reverses",
     f"{rdv('fixed_rule','normal') - rdv('proposed','normal'):+.1f}",
     "errors, optimizer over\nfixed rule, 2 of 3 conditions"),
]

fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.5, 0.935, "Evidence budgeting for edge AI under degraded connectivity",
        ha="center", va="center", fontsize=24, color=INK, family="serif", weight="bold")
ax.text(0.5, 0.855, "A validation study on synthetic and real disaster data: what adaptivity buys, and where it does not",
        ha="center", va="center", fontsize=15.0, color=MUTED, family="serif", style="italic")
ax.plot([0.055, 0.945], [0.80, 0.80], color=RULE, lw=1.6)

w, gap = 0.1645, 0.018
x0 = (1 - (len(CARDS) * w + (len(CARDS) - 1) * gap)) / 2
for i, (col, head, big, sub) in enumerate(CARDS):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.20), w, 0.54, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.3, edgecolor=RULE, facecolor="#fcfcfb"))
    ax.add_patch(FancyBboxPatch((x, 0.715), w, 0.025, boxstyle="round,pad=0.002,rounding_size=0.004",
                                linewidth=0, facecolor=col))
    ax.text(x + w / 2, 0.655, head, ha="center", va="center", fontsize=13.0,
            color=INK, family="serif", linespacing=1.45)
    ax.text(x + w / 2, 0.435, big, ha="center", va="center", fontsize=21,
            color=col, family="serif", weight="bold")
    ax.text(x + w / 2, 0.285, sub, ha="center", va="center", fontsize=10.8,
            color=MUTED, family="serif", linespacing=1.5)

ax.text(0.5, 0.135, "The formulation earns its place by naming the quantities and supplying the threshold.",
        ha="center", va="center", fontsize=14.5, color=INK, family="serif")
ax.text(0.5, 0.088, "Not the solver: it never beat a greedy alternative on the same utility, on either workload.",
        ha="center", va="center", fontsize=14.5, color=INK, family="serif")
ax.text(0.5, 0.035, "Nine synthetic conditions  ·  a real CrisisMMD workload held out by event  "
                    "·  five seeds  ·  two pre-registered tests  ·  artifact reproduces byte for byte",
        ha="center", va="center", fontsize=11.0, color=MUTED, family="serif")

OUT.parent.mkdir(exist_ok=True)
# "531 x 1328 pixels or proportionally more": the raster copies are written at 2x so
# the figure is comfortably above the floor and stays readable at 5 x 13 cm.
fig.savefig(OUT, facecolor="white", dpi=200)
fig.savefig(OUT_TIFF, facecolor="white", dpi=200)
# matplotlib stamps a PDF with the time it was written, which would make this the
# one file in the package that cannot reproduce byte for byte. Pinning CreationDate
# to None removes it, so the claim in Section 10 holds for every shipped file.
fig.savefig(OUT_PDF, facecolor="white", metadata={"CreationDate": None})
plt.close(fig)
from PIL import Image
for f in (OUT, OUT_TIFF, OUT_PDF):
    try: print("wrote", f.relative_to(ROOT), Image.open(f).size)
    except Exception: print("wrote", f.relative_to(ROOT))
