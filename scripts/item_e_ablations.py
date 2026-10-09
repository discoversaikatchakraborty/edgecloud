"""Item E, completion: the component ablations at twenty seeds.

Rerun_Spec item E names results/ablation_20seed_rev1.csv among its outputs and lists
Table XVI among the manuscript updates. The first item E run produced the per-method and
paired files but not the ablation file, so Table XVI and the misaligned ablation sentence in
Section VI-C were left at five seeds while everything around them moved to twenty.

core.experiment already computes the ablation rows on the same pass that produces the method
summaries; the adapter simply discarded them. adapters.seed_ablations now returns them from
the same cached call, so this closes the gap without rerunning the benchmark for a third
time and without touching the protected upstream package.

Output:
  results/ablation_20seed_rev1.csv   per ablation, scenario and seed, with twenty-seed
                                     means and 95% t intervals in the companion summary
  results/ablation_20seed_summary_rev1.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, mean_ci, run_manifest, write_csv  # noqa: E402


def main():
    proto = load_protocol('itemE_protocol.json')
    run_manifest(os.path.join(RESULTS, 'itemE_ablation_manifest_rev1.json'), proto)
    seeds = proto['seeds']
    print(f'[item E ablations] {len(seeds)} seeds: {seeds}')

    frames = []
    for s in seeds:
        ab = adapters.seed_ablations(int(s)).copy()
        ab['seed'] = int(s)
        frames.append(ab)
        print(f'  seed {s}: {len(ab)} ablation rows')
    d = pd.concat(frames, ignore_index=True)
    write_csv(os.path.join(RESULTS, 'ablation_20seed_rev1.csv'), d.to_dict('records'))

    rows = []
    for (ab, sc), g in d.groupby(['ablation', 'scenario']):
        if len(g) != len(seeds):
            print(f'  WARNING {ab}/{sc}: {len(g)} rows for {len(seeds)} seeds')
        mean, lo, hi = mean_ci(g.errors.tolist())
        rows.append(dict(ablation=ab, scenario=sc, n=len(g), errors_mean=mean,
                         errors_lo=lo, errors_hi=hi,
                         includes_zero_vs_zero=bool(lo <= 0 <= hi)))
    summ = pd.DataFrame(rows).sort_values(['scenario', 'ablation'])
    write_csv(os.path.join(RESULTS, 'ablation_20seed_summary_rev1.csv'),
              summ.to_dict('records'))
    print()
    print(summ.to_string(index=False))
    print()
    print('MANUSCRIPT: Table XVI and the ablation sentence of Section VI-C move to twenty '
          'seeds from these files.')


if __name__ == '__main__':
    main()
