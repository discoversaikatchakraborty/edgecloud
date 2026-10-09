"""v2.7: the full synthetic benchmark at twenty seeds, with the predictive-contribution term off
(regression against the paper's twenty-seed results; also the twenty-seed sweeps behind Fig. 1)
and on (does the term disturb a workload where every modality carries signal?)."""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import core
TWENTY = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020]
ap = argparse.ArgumentParser(); ap.add_argument("--pc", choices=["on", "off"], required=True); a = ap.parse_args()
core.USE_PRED_CONTRIB = (a.pc == "on")
core.run_all(HERE / "results_v27" / f"synthetic20_pc_{a.pc}", seeds=TWENTY, log=print)
print("done", a.pc)
