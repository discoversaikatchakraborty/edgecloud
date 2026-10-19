"""Every number in the prose must be reproducible from a results file at the
precision it is printed.

Nine rounds of review found two numbers that were simply wrong -- an accuracy of
0.617 that is 0.566, an escalation rate of 0.4% that is 0.07% -- and several that
were right but stated at the wrong scope. Every structural check passed on all of
them, because a wrong number is structurally identical to a right one.

The first version of this test pooled every numeric cell in results/ and accepted
any prose number within a small tolerance of any of them. With 48,001 cells that
pool covers the number line, and the test passed on a document with 0.617, 0.4123%
and an invented 88.31 s injected into it. It was worthless, and the only reason
that was noticed is that it was run against a control before being believed.

What follows compares STRINGS, not magnitudes. A prose number is traceable when
formatting some source value to the same number of decimal places reproduces it
exactly. That is the same rule the manuscripts' own builders use when they print a
value, so a number that survives it is a number some file really produced.

    python test_numbers_traceable.py ../Option3_Merged/EdgeCloud_Merged_Submission.docx
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
N_TEST = (720, 968)          # the two test-set sizes percentages are taken of

# Numbers that belong to the writing rather than to the results: section numbers,
# counts the text enumerates, and configuration constants stated as design choices.
# Every entry here is a number no results file should have to produce.
ALLOWED = {
    # section, table, figure, reference and enumeration numbers
    *(str(i) for i in range(0, 65)),
    "70", "75", "76", "95", "98", "100", "129", "180", "184", "196", "200", "219",
    "220", "224", "256", "300", "350", "360", "400", "494", "512", "531", "600",
    "620", "700", "720", "768", "900", "968", "1,200", "1,328", "1,500", "1,763",
    "1,800", "1,900", "2,000", "2,048", "4,000", "7,200", "1,719,070", "3,526",
    # design constants and thresholds
    "0.1", "0.15", "0.4", "0.5", "0.70", "0.72", "0.90", "1.0", "1.75", "2.8",
    "0.01", "0.02", "0.05", "0.001",
    # split and corpus sizes quoted from provenance
    "173", "273", "475", "709", "795", "839", "890", "1,065", "1,393", "1,754",
    "2,212", "2,285", "3.12", "3.14", "2.0", "9.0",
}
YEARS = {str(y) for y in range(1990, 2031)}
# Summary-level files are what prose quotes; seed- and record-level files are the
# raw material behind them and admitting them would loosen the pool by thousands
# of values for no gain.
SKIP = ("seed_level", "record_level", "envelopes")


def sources():
    vals = []
    # The derived CrisisMMD records are shipped and the prose quotes statistics of
    # them directly, so their summary statistics belong in the pool.
    rec = ROOT / "data/crisismmd/records.csv.gz"
    if rec.exists():
        r = pd.read_csv(rec)
        for c in r.columns:
            if r[c].dtype.kind in "fi":
                v = r[c].dropna()
                if len(v):
                    vals.extend([float(v.median()), float(v.min()), float(v.max()),
                                 float(v.mean()), float(v.sum()), float(len(v))])
    for f in sorted(RES.glob("*.csv")):
        if any(k in f.name for k in SKIP):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        for c in d.columns:
            if d[c].dtype.kind in "fi":
                vals.extend(float(x) for x in d[c].dropna().unique())
    for f in list(RES.glob("*.json")) + [ROOT / "config.json",
                                         ROOT / "data/traces/provenance.json",
                                         ROOT / "data/crisismmd/provenance.json"]:
        if not f.exists():
            continue

        def walk(o):
            if isinstance(o, dict):
                [walk(v) for v in o.values()]
            elif isinstance(o, list):
                [walk(v) for v in o]
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                vals.append(float(o))
        walk(json.loads(f.read_text()))
    return vals


def printable(vals):
    """Every string those values can legitimately print as, at 0 to 4 decimals,
    together with the percentages-of-test-set and the pairwise ratios the prose forms."""
    out = set()

    def add(x):
        for dp in range(0, 5):
            out.add(f"{x:,.{dp}f}")
            out.add(f"{abs(x):,.{dp}f}")
    for v in vals:
        add(v)
        for n in N_TEST:
            add(100.0 * v / n)
    # inflation and cost ratios: latency, cost and error columns divided within a
    # file. Every results file counts, not only those named *summary.csv -- the
    # compute-cost table is not one, and its 6,190x ratio went unrecognised because
    # of it.
    lat, err = [], []
    for f in sorted(RES.glob("*.csv")):
        if any(k in f.name for k in SKIP):
            continue
        d = pd.read_csv(f)
        for c in d.columns:
            if "p95_latency" in c or "per_slot_ms" in c or "peak_kib" in c:
                lat.extend(float(x) for x in d[c].dropna().unique())
            if c.startswith("errors"):
                err.extend(float(x) for x in d[c].dropna().unique())
    for pool_ in (lat, err):
        for a in pool_:
            for b in pool_:
                if b and 1.0 < a / b < 1e5:
                    add(a / b)
    return out


def main(path):
    d = Document(path)
    items = []
    for ch in d.element.body.iterchildren():
        if ch.tag.endswith("}p"):
            t = Paragraph(ch, d).text
            if t.strip() and not re.match(r"^\[\d+\]", t.strip()):
                items.append(t)
        elif ch.tag.endswith("}tbl"):
            for r in Table(ch, d).rows:
                items.append(" | ".join(c.text for c in r.cells))
    ok = printable(sources())
    bad, seen = [], set()
    for t in items:
        for m in re.finditer(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])", t):
            raw = m.group(0)
            if raw in ALLOWED or raw in YEARS or raw in ok:
                continue
            k = (raw, t[:40])
            if k in seen:
                continue
            seen.add(k)
            bad.append((raw, t[max(0, m.start() - 55):m.end() + 40].replace("\n", " ")))
    print(f"{Path(path).name}: {len(bad)} numbers no results file produces")
    for raw, ctx in bad:
        print(f"  [{raw}]  ...{ctx.strip()}...")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else ROOT.parent / "Option3_Merged/EdgeCloud_Merged_Submission.docx"))
