"""Every results table must display every row of its source, or say what it omits.

Three defects of one kind were found by inspection in this manuscript: the
real-data section reported one of three conditions, the review-rule table six of
nine, and the measured-capacity table three of five methods -- and in that last
case the omitted method held the largest effect, which the surrounding prose
quoted as a maximum. Fixing the three instances establishes nothing about the
rest, so this test states the property they violated and checks it everywhere.

    A table is complete if and only if, for each key column of its source file,
    the set of values it displays equals the set present in the source.

Where a subset is deliberate the entry carries an `allow` set naming exactly
what may be missing, so the omission is a recorded decision rather than an
oversight. Run against the built manuscript:

    python test_table_completeness.py ../Option3_Merged/EdgeCloud_Merged_Submission.docx
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.table import Table

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

TRACE_METHODS = {"fixed_rule": "Fixed rule", "fixed_confidence_gate": "Confidence gate",
                 "fixed_compression": "Fixed compression", "proposed": "Proposed",
                 "cloud_transmit_all": "Transmit all*"}
RD_METHODS = {"image_only": "Image only", "cloud_transmit_all": "Transmit all*",
              "fixed_rule": "Fixed rule", "fixed_confidence_gate": "Confidence gate",
              "random_within_budget": "Random within budget",
              "greedy_uncertainty": "Greedy by uncertainty",
              "greedy_value": "Greedy, proposed utility",
              "proposed": "Proposed (exact knapsack)",
              "fixed_compression": "Fixed compression", "text_only": "Text only"}
ADV_METHODS = {"fixed_rule": "Fixed rule (no filter)", "fixed_rule_admissible": "Fixed rule + admissibility",
               "proposed": "Proposed"}
CB_METHODS = {"all_summary": "Summaries only", "fixed_rule_window": "Fixed rule, per window",
              "fixed_confidence_gate": "Confidence gate", "random_within_budget": "Random within budget",
              "greedy_uncertainty": "Greedy by uncertainty", "greedy_value": "Greedy, proposed utility",
              "proposed": "Proposed (exact knapsack)", "fixed_rule": "Fixed rule, ignores budget*"}
DC_METHODS = {"all_summary": "Summaries only", "fixed_rule_window": "Fixed rule, per window",
              "greedy_uncertainty": "Greedy by uncertainty", "proposed": "Proposed (exact knapsack)"}
LB_METHODS = {"image_only": "Image only", "text_only": "Text only",
              "cloud_transmit_all": "Transmit all*", "fixed_confidence_gate": "Confidence gate",
              "fixed_compression": "Fixed compression", "fixed_rule": "Fixed rule",
              "random_within_budget": "Random within budget", "proposed": "Proposed"}
DI_METHODS = dict(LB_METHODS, **{"wait_if_uncertain": "Wait if uncertain",
                                 "derived_threshold": "Derived threshold T*",
                                 "proposed": "Proposed, \u03bb = 0.0004"})
RD_CONDITIONS = {"normal": "Normal", "low_bandwidth": "Low bw.",
                 "misaligned_evidence": "Misaligned"}

# Each deliverable numbers its tables differently, so the map is per document.
# table number -> (source csv, key column, {csv value: label as printed}, allowed omissions)
ROW_CHECKS = {
    "paperA": {5: ("confidence_intervals.csv", "method", LB_METHODS, set(),
                   ("scenario", "low_bandwidth")),
               6: ("confidence_intervals.csv", "method", DI_METHODS, set(),
                   ("scenario", "delayed_imagery")),
               7: ("spoof_sweep_summary.csv", "method", ADV_METHODS, set()),
               8: ("adversary_grid_summary.csv", "method", ADV_METHODS, set()),
               9: ("review_rule_sensitivity.csv", "scenario", None, set()),
               10: ("contested_budget_summary.csv", "method", CB_METHODS, set()),
               11: ("link_loss_normal_summary.csv", "method", TRACE_METHODS, set()),
               12: ("link_trace_summary.csv", "method", TRACE_METHODS, set()),
               13: ("decentralized_summary.csv", "method", DC_METHODS, set())},
    "merged": {5: ("confidence_intervals.csv", "method", LB_METHODS, set(),
                   ("scenario", "low_bandwidth")),
               6: ("confidence_intervals.csv", "method", DI_METHODS, set(),
                   ("scenario", "delayed_imagery")),
               7: ("spoof_sweep_summary.csv", "method", ADV_METHODS, set()),
               8: ("adversary_grid_summary.csv", "method", ADV_METHODS, set()),
               9: ("review_rule_sensitivity.csv", "scenario", None, set()),
               10: ("contested_budget_summary.csv", "method", CB_METHODS, set()),
               11: ("link_loss_normal_summary.csv", "method", TRACE_METHODS, set()),
               12: ("link_trace_summary.csv", "method", TRACE_METHODS, set()),
               13: ("decentralized_summary.csv", "method", DC_METHODS, set()),
               15: ("real_data_summary.csv", "method", RD_METHODS, set())},
    "paperB": {3: ("real_data_summary.csv", "method", RD_METHODS, set())},
}
COL_CHECKS = {
    "paperA": {},
    "merged": {15: ("real_data_summary.csv", "scenario", RD_CONDITIONS, set())},
    "paperB": {3: ("real_data_summary.csv", "scenario", RD_CONDITIONS, set())},
}


def which(path):
    """Identify the deliverable from its filename."""
    n = Path(path).name.lower()
    return "merged" if "merged" in n else "paperB" if "companion" in n else "paperA"

LABELS = {  # condition slugs as Table 9 prints them
    "normal": "Normal", "low_bandwidth": "Low bandwidth",
    "delayed_imagery": "Delayed imagery", "image_missing": "Image missing",
    "text_missing": "Text missing", "noisy_text": "Noisy text",
    "intermittent_outage": "Intermittent outage",
    "contradictory_evidence": "Contradictory", "misaligned_evidence": "Misaligned",
}


def tables_by_number(doc):
    """Map a table number to its grid, pairing each table with the caption below it."""
    body = list(doc.element.body.iterchildren())
    found, pending = {}, []
    for ch in body:
        if ch.tag.endswith("}tbl"):
            pending.append([[c.text.strip() for c in r.cells] for r in Table(ch, doc).rows])
        elif ch.tag.endswith("}p") and pending:
            text = "".join(ch.itertext()).strip()
            if text.startswith("Table "):
                num = text.split()[1].rstrip(".")
                if num.isdigit():
                    found[int(num)] = pending[-1]
                pending = []
            elif text:
                pending = []
    return found


def main(path):
    kind = which(path)
    CHECKS, COLUMN_CHECKS = ROW_CHECKS[kind], COL_CHECKS[kind]
    grids = tables_by_number(Document(path))
    failures = []
    print(f"{Path(path).name}  ({kind})")
    for num, spec in CHECKS.items():
        csv, key, labels, allow = spec[:4]
        where = spec[4] if len(spec) > 4 else None
        if num not in grids:
            failures.append(f"Table {num}: not found in the document")
            continue
        src = pd.read_csv(RES / csv, dtype=str)
        # A table scoped to one condition is complete if it shows every method that
        # exists for that condition, not every method in the file.
        if where:
            src = src[src[where[0]] == where[1]]
        labels = labels or {k: LABELS[k] for k in src[key].unique()}
        expected = set(src[key].unique())
        shown = {c for row in grids[num] for c in row}
        missing = {v for v in expected if labels.get(v, v) not in shown} - allow
        if missing:
            failures.append(f"Table {num} omits {sorted(missing)} present in {csv}")
    for num, (csv, key, labels, allow) in COLUMN_CHECKS.items():
        if num not in grids:
            failures.append(f"Table {num}: not found in the document")
            continue
        header = grids[num][0]
        expected = set(pd.read_csv(RES / csv, dtype=str)[key].unique())
        missing = {v for v in expected if labels.get(v, v) not in header} - allow
        if missing:
            failures.append(f"Table {num} omits columns {sorted(missing)} present in {csv}")
    for f in failures:
        print("FAIL:", f)
    n = len(CHECKS) + len(COLUMN_CHECKS)
    print(f"{n - len(failures)} of {n} completeness checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else ROOT.parent / "Option3_Merged/EdgeCloud_Merged_Submission.docx"))
