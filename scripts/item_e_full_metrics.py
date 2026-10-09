"""Item E, completion: every reported metric at twenty seeds, not only three.

The first item E run recorded errors, bytes and tail latency. The manuscript's synthetic
tables also print median latency, the share of images sent and the review-queue share, so
updating only three columns would have left Tables V and VI mixing twenty-seed and
five-seed numbers in the same row. This extracts the full per-seed summary that
core.experiment already produces, for every scenario and method, so each table moves to
twenty seeds as a whole.

Output:
  results/full_metrics_20seed_rev1.csv       per scenario, method, seed
  results/full_metrics_20seed_summary_rev1.csv   twenty-seed means with 95% t intervals
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, mean_ci, run_manifest, write_csv  # noqa: E402

METRICS = ('errors', 'mean_bytes', 'p95_latency_s', 'median_latency_s', 'review_rate',
           'image_selected_rate', 'macro_f1', 'conformal_coverage',
           'reviews_per_caught_error', 'caught_errors', 'accepted_errors',
           'slot_budget_violation_rate', 'provenance_complete_rate')


def main():
    proto = load_protocol('itemE_protocol.json')
    run_manifest(os.path.join(RESULTS, 'itemE_fullmetrics_manifest_rev1.json'), proto)
    seeds = proto['seeds']
    print(f'[full metrics] {len(seeds)} seeds')

    frames = []
    for s in seeds:
        g = adapters._seed_summary(int(s)).copy()
        frames.append(g)
        print(f'  seed {s}: {len(g)} rows, {g.scenario.nunique()} scenarios')
    d = pd.concat(frames, ignore_index=True)
    keep = ['seed', 'scenario', 'method'] + [m for m in METRICS if m in d.columns]
    d = d[keep]
    write_csv(os.path.join(RESULTS, 'full_metrics_20seed_rev1.csv'), d.to_dict('records'))

    rows = []
    for (sc, m), g in d.groupby(['scenario', 'method']):
        r = dict(scenario=sc, method=m, n=len(g))
        for met in METRICS:
            if met not in g.columns:
                continue
            mean, lo, hi = mean_ci(g[met].tolist())
            r[f'{met}_mean'] = mean
            r[f'{met}_lo'] = lo
            r[f'{met}_hi'] = hi
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(['scenario', 'method'])
    write_csv(os.path.join(RESULTS, 'full_metrics_20seed_summary_rev1.csv'),
              summ.to_dict('records'))
    print(f'[full metrics] {len(summ)} scenario-method cells at {len(seeds)} seeds')
    print('MANUSCRIPT: Tables V and VI move to twenty seeds in every column, including '
          'median latency, image share and queue share.')


if __name__ == '__main__':
    main()
