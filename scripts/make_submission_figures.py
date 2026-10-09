"""Regenerate the submission figures at the width they are actually printed at.

OUTCOME: NOT USED. Kept because the attempt and its result are part of the record.
Regenerating on a column-width canvas does raise the type from about 5 pt to 7 pt,
but the inline annotations are positioned in data coordinates that were chosen for
a 5.8 inch canvas, so on the shorter canvas "Fixed rule" and "Transmit all" stack
on each other and "Derived threshold T*" runs into the x axis. Overlapping labels
are a worse defect than small type, so the manuscript keeps the original figures.
Resolving this properly needs either a thirteenth page, which would let the figures
run two columns wide at their authored size, or re-authoring the annotation offsets,
which changes a validated figure and was not done without authorisation.

The figures in upstream/prototype_v2.3 were authored 5.8 to 7.2 inches wide with
8.5 pt text and are included in the manuscript at the column width, 3.49 inches. The
downscaling takes their text to roughly 4 to 5 pt, below any readable floor. This
script re-runs the upstream plotting functions unchanged, on the same retained
CSVs, with the canvas set to the column width and the font sizes set so that no
downscaling happens at all.

upstream/ is protected and is not edited. The plotting logic and the data are the
upstream ones; only the canvas size and the type sizes differ, so every plotted
value is identical to the published figure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream" / "prototype_v2.3"
sys.path.insert(0, str(UP))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_IN = 3.45          # \columnwidth in the IEEEtran two-column A4 layout
BASE_FONT = 7.0        # type size in the regenerated figure, at 1:1

_real_subplots = plt.subplots
_scale = {"k": 1.0}


def _subplots(*a, **kw):
    """Shrink the canvas to the column width, remembering the factor used."""
    fs = kw.get("figsize")
    if fs:
        k = COL_IN / fs[0]
        _scale["k"] = k
        kw["figsize"] = (COL_IN, fs[1] * k)
    return _real_subplots(*a, **kw)


def _patch_fontsizes():
    """Hardcoded fontsize= kwargs in the upstream code are in the authored scale."""
    from matplotlib.axes import Axes
    for name in ("annotate", "text", "legend", "set_title", "set_xlabel", "set_ylabel"):
        orig = getattr(Axes, name)

        def wrap(self, *a, _o=orig, **kw):
            if "fontsize" in kw and isinstance(kw["fontsize"], (int, float)):
                kw["fontsize"] = BASE_FONT * (kw["fontsize"] / 8.5)
            return _o(self, *a, **kw)
        setattr(Axes, name, wrap)


def main():
    plt.subplots = _subplots
    _patch_fontsizes()
    import make_figures_adhoc as A
    # upstream/ is protected: redirect the module's output away from it before
    # any figure is written. An earlier run of this script wrote into
    # upstream/prototype_v2.3/figures_submission and broke the package digests;
    # the gate caught it and the file was restored from git.
    A.OUT = ROOT / "results" / "figures_colwidth"
    A.ROOT = ROOT  # upstream save() prints OUT relative to its own ROOT
    A.OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": BASE_FONT, "axes.labelsize": BASE_FONT,
        "axes.titlesize": BASE_FONT + 0.5,
        "xtick.labelsize": BASE_FONT - 0.5, "ytick.labelsize": BASE_FONT - 0.5,
        "legend.fontsize": BASE_FONT - 0.5,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    A.plt.rcParams.update(plt.rcParams)
    A.plt.subplots = _subplots
    A.fig_delay()
    import shutil
    dest = ROOT / "manuscript" / "latex"
    src = A.OUT / "fig_delay_frontier.png"
    shutil.copy2(src, dest / "fig_delay_frontier.png")
    print(f"[regenerated at column width] {src.relative_to(ROOT)} -> manuscript/latex/")


if __name__ == "__main__":
    main()
