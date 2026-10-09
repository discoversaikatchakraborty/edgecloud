"""Item B: the budgeting comparison on an externally defined CrisisMMD label.

Rerun_Spec item B asks for three things. The third is delivered here in full, the first
cannot be delivered from what this machine holds and is recorded as BLOCKED with its exact
reasons, and the second is delivered on a restricted label set whose restriction is stated
rather than hidden.

WHAT IS BLOCKED, AND WHY

1. "Reproduce the published single-modality and multimodal baseline numbers on the official
   split first; report the reproduction gap."
   BLOCKED on three independent grounds:
   a. The official train/dev/test split is not present. The CrisisMMD v2.0 corpus on this
      machine holds only annotations/*.tsv; its own Readme names the split as a separate
      archive, crisismmd_datasplit_all.zip, which was not retrieved. Reproducing a published
      number on a different split does not measure a reproduction gap, it measures the split.
   b. The published baseline values of Ofli, Alam and Imran (ISCRAM 2020) are not held on
      this machine in any retained form. Writing them from recollection would be fabricating
      the reference that the gap is measured against.
   c. The image files are absent. Only the CLIP image embeddings derived from them are
      retained, so an image or multimodal baseline cannot be computed from pixels at all.
   To unblock: place crisismmd_datasplit_all.zip beside annotations/ and supply the published
   table as a file. Nothing else here needs to change.

2. The established tasks cannot be run at their published label cardinality.
   The informative-against-not-informative task has two classes and the humanitarian-category
   task has eight. The decision machinery in upstream/prototype_v2.3/core.py is fixed at
   three: CLASSES and CLASS_NAMES carry three entries, the evidence fusion falls back to a
   uniform 1/3 when no modality is selected, and the conformal set is built by indexing
   range(3) over the membership matrix. upstream is reproduced unchanged so that the paper's
   published numbers stay checkable against it, so none of that is edited.

WHAT IS DELIVERED

The budgeting comparison on the humanitarian-category field of the CrisisMMD annotations,
restricted to its three most frequent categories. That label is externally defined by the
dataset's own annotators rather than repurposed by this paper, which is the substance of
what R2.3 asks for: it tests whether the paper's real-data finding depends on the particular
image_damage label it chose. The restriction to three categories is a property of the
decision machinery, not a choice made to flatter the result, and the discarded categories
and rows are reported. Both class-handling variants are run, as in item C, so the label
change and the class handling are separate factors.

The informative-against-not-informative task is reported as NOT RUN with its reason, not
silently dropped.

Outputs:
  results/crisismmd_humanitarian_rev1.csv    per-method, per-condition, per-seed results
  results/crisismmd_humanitarian_rotations_rev1.csv   proposed minus fixed across rotations
  results/itemB_decision_rev1.json           the dispositions, the label mapping, the counts
  results/itemB_manifest_rev1.json           run manifest with the frozen protocol digest
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, mean_ci, run_manifest, write_csv  # noqa: E402

ANN = ('/Users/arlina/Downloads/all_projects/Tier-Ready/12_edgewait/data_real/ex/'
       'CrisisMMD_v2.0/annotations')


def annotations() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ANN, '*.tsv')))
    if not files:
        raise FileNotFoundError(f'no CrisisMMD annotation files under {ANN}')
    a = pd.concat([pd.read_csv(f, sep='\t', dtype=str) for f in files], ignore_index=True)
    return a[['image_id', 'text_info', 'text_human']].drop_duplicates('image_id')


def modelled_annotations() -> pd.DataFrame:
    """The annotation row for each record this project models, in records-file row order.

    Two things make a keyed join the wrong tool here. The annotations cover 18,082 rows of
    the corpus while the derived records hold 3,526, so category frequencies must be counted
    on the modelled subset: counted on the whole corpus they select a different set of three
    categories, restricting the task by a population the pipeline never sees. And window_id
    is not unique, because 202 windows carry more than one image, so a window-keyed
    dictionary would silently give every image in a multi-image window one image's label.

    image_id is unique over all 3,526 records, so the label is carried positionally.
    run_real_data.build_frame reads the records file and builds its columns from it row for
    row, with no filtering or reordering, so row i of the rotation frame is row i of the
    records file. That identity is asserted rather than assumed.
    """
    rec = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'upstream', 'prototype_v2.3', 'data', 'crisismmd',
                                   'records.csv.gz'))
    m = rec[['image_id', 'window_id']].merge(annotations(), on='image_id', how='left',
                                             validate='one_to_one')
    missing = int(m.text_human.isna().sum())
    if missing:
        raise AssertionError(f'{missing} modelled records have no annotation row')
    assert len(m) == len(rec), 'positional correspondence with the records file broken'
    return m


def build_label(field: str, keep: int):
    """The externally defined label, restricted to its `keep` most frequent categories.

    Returns a (n,) integer array aligned with the records file where a kept category gives
    its code and a dropped one gives -1, the mapping used so the manuscript can print
    category names rather than integers, and the full category counts on the modelled subset
    so the dropped categories can be reported.
    """
    m = modelled_annotations()
    counts = m[field].value_counts()
    cats = list(counts.index[:keep])
    code = {c: i for i, c in enumerate(cats)}
    y = m[field].map(code).fillna(-1).to_numpy(int)
    return y, code, {c: int(counts[c]) for c in counts.index}


def relabelled_split(rotation: int, y_external: np.ndarray):
    """One rotation with y replaced by the external label and dropped-category rows removed.

    y_external is aligned with the records file by position, with -1 for a category outside
    the kept set.
    """
    s = adapters.load_crisismmd_rotation(rotation)
    d = s.frame
    if len(d) != len(y_external):
        raise AssertionError(f'rotation frame has {len(d)} rows, label array has '
                             f'{len(y_external)}: positional alignment is broken')
    keep = y_external >= 0
    if keep.sum() == 0:
        raise AssertionError('no rows survive the label restriction')
    d2 = d.loc[keep].copy()
    d2['y'] = y_external[keep]
    s.frame = d2.reset_index(drop=True)
    for attr, arr in (('y', None), ('event', s.event), ('role', s.role),
                      ('bytes_text', s.bytes_text), ('bytes_image', s.bytes_image)):
        if arr is not None:
            setattr(s, attr, arr[keep])
    s.y = s.frame.y.to_numpy(int)
    s.X_text_full = s.X_text_full[keep]
    s.X_image_full = s.X_image_full[keep]
    s.X_text_pca8 = s.X_text_pca8[keep]
    return s, int(keep.sum()), int(len(d))


def main():
    proto = load_protocol('itemB_protocol_rev1.json')
    run_manifest(os.path.join(RESULTS, 'itemB_manifest_rev1.json'), proto)
    dec_a = json.load(open(os.path.join(RESULTS, 'itemA_decision_rev1.json')))
    ch = dec_a['best_cell']['class_handling']
    assert ch == proto['class_handling'], 'protocol and item A disagree on class handling'
    field, keep_k = proto['label_field'], int(proto['label_classes_kept'])
    print(f'[item B] external label: {field}, top {keep_k} categories, '
          f'class handling from item A: {ch}')

    y_ext, code, all_counts = build_label(field, keep_k)
    print(f'[item B] categories kept: {code}')
    print(f'[item B] categories dropped: '
          f'{ {c: n for c, n in all_counts.items() if c not in code} }')

    rows, rot_rows, coverage = [], [], []
    for rot in proto['rotations']:
        split, n_kept, n_all = relabelled_split(rot, y_ext)
        coverage.append(dict(rotation=rot, rows_kept=n_kept, rows_total=n_all,
                             n_train=int(split.mask('train').sum()),
                             n_cal=int(split.mask('cal').sum()),
                             n_test=int(split.mask('test').sum())))
        print(f'[item B] rotation {rot}: {n_kept} of {n_all} rows, '
              f'test n={int(split.mask("test").sum())}')
        for cond in proto['conditions']:
            per = {'current': [], 'handled': []}
            for seed in proto['seeds']:
                cur = adapters.run_real_budgeting(split, cond, seed, proto['methods'],
                                                  class_weight='none')
                new = adapters.run_real_budgeting(split, cond, seed, proto['methods'],
                                                  class_weight=ch)
                per['current'].append(cur)
                per['handled'].append(new)
                for variant, res in (('current', cur), ('handled', new)):
                    for m, v in res.items():
                        rows.append(dict(task=f'{field}_top{keep_k}', rotation=rot,
                                         condition=cond, variant=variant, method=m,
                                         seed=seed, **v))
            for variant in ('current', 'handled'):
                diffs = [r['Proposed (exact knapsack)']['errors'] - r['Fixed rule']['errors']
                         for r in per[variant]]
                mean, lo, hi = mean_ci(diffs)
                rot_rows.append(dict(task=f'{field}_top{keep_k}', rotation=rot,
                                     condition=cond, variant=variant,
                                     n_test=int(split.mask('test').sum()),
                                     proposed_minus_fixed=mean, lo=lo, hi=hi))

    write_csv(os.path.join(RESULTS, 'crisismmd_humanitarian_rev1.csv'), rows)
    write_csv(os.path.join(RESULTS, 'crisismmd_humanitarian_rotations_rev1.csv'), rot_rows)

    n_cmp = len([r for r in rot_rows if r['variant'] == 'handled'])
    wins = {v: sum(1 for r in rot_rows
                   if r['variant'] == v and r['proposed_minus_fixed'] < 0)
            for v in ('current', 'handled')}
    decision = {
        'protocol': 'itemB_protocol_rev1.json',
        'protocol_sha256': proto.get('_sha256'),
        'external_label_field': field,
        'categories_kept': code,
        'categories_dropped': {c: n for c, n in all_counts.items() if c not in code},
        'rows_kept_per_rotation': coverage,
        'class_handling_from_item_a': ch,
        'optimizer_ahead_of_fixed_rule': {v: f'{wins[v]} of {n_cmp}' for v in wins},
        'comparisons_per_variant': n_cmp,
        'step_1_published_baseline_reproduction': {
            'status': 'BLOCKED',
            'grounds': [
                'The official CrisisMMD train/dev/test split is not present on this '
                'machine. Only annotations/*.tsv is held; the corpus Readme names the '
                'split as a separate archive, crisismmd_datasplit_all.zip, not retrieved.',
                'The published baseline values of Ofli, Alam and Imran (ISCRAM 2020) are '
                'not held on this machine in any retained form, so the gap has no '
                'verified reference to be measured against.',
                'The image files are absent; only derived CLIP image embeddings are '
                'retained, so no baseline can be computed from pixels.'],
            'to_unblock': 'Supply crisismmd_datasplit_all.zip and the published baseline '
                          'table as files in the project.'},
        'informative_vs_not_informative_task': {
            'status': 'NOT RUN',
            'reason': 'Two classes. The decision machinery in upstream/prototype_v2.3/'
                      'core.py is fixed at three: CLASSES and CLASS_NAMES hold three '
                      'entries, evidence fusion falls back to a uniform 1/3, and the '
                      'conformal set indexes range(3) over the membership matrix. '
                      'upstream is reproduced unchanged so its published numbers stay '
                      'checkable, so it was not edited to admit a two-class task.'},
        'humanitarian_category_at_published_cardinality': {
            'status': 'NOT RUN',
            'reason': 'Eight classes in the annotations, blocked by the same fixed '
                      'three-class machinery. The three most frequent categories were '
                      'run instead and the restriction is reported, not hidden.'},
        'what_this_does_and_does_not_establish':
            'It establishes how the budgeting comparison behaves when the label comes '
            'from the dataset annotators rather than from this paper, on the three most '
            'frequent humanitarian categories, under the same conditions and the same '
            'event-based splits. It does not reproduce any published baseline, does not '
            'measure a reproduction gap and is not the published eight-class task.'}
    with open(os.path.join(RESULTS, 'itemB_decision_rev1.json'), 'w') as f:
        json.dump(decision, f, indent=2)
        f.write('\n')
    print(f'[item B] optimizer ahead of the fixed rule: current {wins["current"]} of '
          f'{n_cmp}, handled {wins["handled"]} of {n_cmp}')
    print('[item B] step 1 (published baseline on the official split): BLOCKED, see '
          'results/itemB_decision_rev1.json')
    print('MANUSCRIPT: Section VIII new table beside Table XIII; Section VIII-B first '
          'paragraph; Section X must carry the BLOCKED grounds.')


if __name__ == '__main__':
    main()
