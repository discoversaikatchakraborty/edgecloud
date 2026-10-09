"""Adapter layer between the rerun scripts and the Edge repository.

THIS IS THE ONLY FILE THAT NEEDS REPOSITORY KNOWLEDGE.

Every function below raises NotImplementedError until it is wired to the
package's own loaders. The rerun scripts call only these functions, so once
they are implemented the scripts run unchanged. Keep the contracts exactly as
documented; the scripts validate shapes and will stop on a mismatch.

Expected wiring (to be confirmed against the repository):
  * CrisisMMD derived features live under data/crisismmd/ with provenance.json.
  * Event-split rotations 0..6 are the seven pre-specified 3/2/2 assignments
    used for Table XIV of the revised manuscript (Table XVI of the original).
  * The synthetic benchmark exposes a runner that takes a seed list and a
    condition name and returns per-method error counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    """One event-split rotation of the CrisisMMD workload.

    X_text_full : (n, d_text) float array, FULL CLIP text embedding (not PCA-reduced)
    X_image_full: (n, d_img)  float array, FULL CLIP image embedding
    y           : (n,) int labels, the damage label used in the paper
    event       : (n,) str event identifiers
    role        : (n,) str in {"train", "cal", "test"} for this rotation
    bytes_text  : (n,) int measured text payload bytes
    bytes_image : (n,) int measured image payload bytes
    """

    X_text_full: np.ndarray
    X_image_full: np.ndarray
    y: np.ndarray
    event: np.ndarray
    role: np.ndarray
    bytes_text: np.ndarray
    bytes_image: np.ndarray

    def mask(self, role: str) -> np.ndarray:
        return self.role == role


def load_crisismmd_rotation(rotation: int) -> Split:
    """Return the CrisisMMD workload with roles assigned for `rotation` (0..6).

    Rotation 0 must reproduce the split reported in the paper (Irma + Iraq-Iran
    as test, n_test = 968). The scripts assert this.
    """
    return _load_crisismmd_rotation(rotation)


def edge_text_model(class_weight: str | None = None, seed: int = 0,
                    local_dims: int | None = None):
    """The package's edge text model, unfitted, with optional class weighting.

    core.fit_models builds the edge ("local") model of every modality as

        LogisticRegression(max_iter=700, C=0.75, random_state=seed)

    fitted on the leading LOCAL_DIMS[m] columns of that modality's feature matrix,
    with no feature scaling. For text LOCAL_DIMS is 4 of the 8 shipped components.

    The estimator returned here carries those hyperparameters and that absence of
    scaling and by default fits on the whole representation it is handed. That is
    deliberate: in item A's grid the number of dimensions is the representation
    factor, so holding the classifier factor to a fixed leading-4 truncation would
    make the representation factor inert for this classifier and confound the two
    things the diagnostic separates. `local_dims` restores the truncation and is
    used only for the reference cell that reproduces the deployed edge pathway
    exactly.
    """
    from sklearn.linear_model import LogisticRegression

    est = LogisticRegression(max_iter=700, C=0.75, class_weight=class_weight,
                             random_state=seed)
    if local_dims is None:
        return est
    return _LeadingDims(est, int(local_dims))


class _LeadingDims:
    """Fit and predict on the leading `k` columns only, as the edge model does."""

    def __init__(self, est, k):
        self.est, self.k = est, int(k)

    def fit(self, X, y):
        self.est.fit(X[:, :self.k], y)
        return self

    def predict(self, X):
        return self.est.predict(X[:, :self.k])


def run_synthetic_condition(condition: str, seeds: list[int], methods: list[str]) -> dict:
    """Run the synthetic benchmark for one condition over `seeds`.

    Returns {method: {seed: {"errors": int, "bytes": float, "p95": float}}}.
    Must be the same code path that produced the paper's tables; the scripts
    check that seeds 0..4 reproduce the published values before continuing.
    """
    return _run_synthetic_condition(condition, seeds, methods)


def run_real_budgeting(split: Split, condition: str, seed: int, methods: list[str],
                       delay_seconds: np.ndarray | None = None,
                       forecast_seconds: np.ndarray | None = None,
                       class_weight: str | None = None) -> dict:
    """Run the real-data budgeting comparison for one condition and seed.

    delay_seconds / forecast_seconds (optional, (n,) arrays over test windows)
    inject a constructed image arrival delay and its forecast for item D.
    class_weight (optional) applies item A's chosen class handling to every
    per-modality model for item C.

    Returns {method: {"errors": int, "bytes": float, "p95": float,
                      "median": float, "image_sent": float}}.
    """
    return _run_real_budgeting(split, condition, seed, methods, delay_seconds,
                               forecast_seconds, class_weight)


def derive_wait_threshold(split: Split, forecast_seconds: np.ndarray) -> float:
    """Return T* = median over calibration windows with a pending image of the
    closed-form threshold for the image-summary action, Eq. (8) of the revised
    manuscript. Must use calibration windows only."""
    return _derive_wait_threshold(split, forecast_seconds)


# ---------------------------------------------------------------------------
# Wiring to upstream/prototype_v2.3
# ---------------------------------------------------------------------------
import functools
import os
import sys

_UPSTREAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'upstream', 'prototype_v2.3')

# The revision package names conditions and methods as the manuscript prints them. The
# benchmark names them as the code does. The translation is stated once, here, so every item
# uses the same one.
CONDITION = {'normal': 'normal', 'low_bandwidth': 'low_bandwidth',
             'delayed_imagery': 'delayed_imagery', 'misaligned': 'misaligned_evidence',
             'contradictory': 'contradictory_evidence'}
METHOD = {'Proposed': 'proposed', 'Fixed rule': 'fixed_rule',
          'Transmit all': 'cloud_transmit_all', 'Wait if uncertain': 'wait_if_uncertain',
          'Derived threshold': 'derived_threshold', 'Confidence gate': 'fixed_confidence_gate',
          'Fixed compression': 'fixed_compression',
          'Random within budget': 'random_within_budget', 'Image only': 'image_only',
          'Text only': 'text_only', 'Fixed rule + admissibility': 'fixed_rule_admissible',
          # The real-data tables name the solver arms the way the manuscript prints them.
          # 'Proposed (exact knapsack)' is the same package policy as 'Proposed': core.py
          # solves the budgeted selection exactly and the longer name is the manuscript's
          # way of distinguishing it from the two greedy arms beside it in the table.
          'Proposed (exact knapsack)': 'proposed',
          'Greedy by uncertainty': 'greedy_uncertainty',
          'Greedy, proposed utility': 'greedy_value'}
METRIC = {'errors': 'errors', 'bytes': 'mean_bytes', 'p95': 'p95_latency_s'}


@functools.lru_cache(maxsize=None)
def _seed_experiment(seed: int):
    """The whole return tuple of core.experiment for one seed, cached.

    _seed_summary keeps only element 0, the per-scenario method summaries. Item E's
    ablation table needs element 3, which core.experiment already computes on the same pass,
    so discarding it meant the ablations could not be recomputed at twenty seeds without
    paying for the whole benchmark again.
    """
    if _UPSTREAM not in sys.path:
        sys.path.insert(0, _UPSTREAM)
    cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        import core
        return core.experiment(int(seed), log=lambda *a, **k: None)
    finally:
        os.chdir(cwd)


def seed_ablations(seed: int):
    """The ablation rows core.experiment produces for one seed."""
    return _seed_experiment(seed)[3]


@functools.lru_cache(maxsize=None)
def _seed_summary(seed: int):
    """Every scenario and method for one seed, from the package's own experiment path.

    core.py reads its configuration and data by relative path, so the working directory has
    to be the package root while it runs.
    """
    return _seed_experiment(seed)[0]


def _run_synthetic_condition(condition: str, seeds, methods) -> dict:
    """Per method, per seed, the three metrics the revision reports.

    This calls the package's own experiment function, which is the code path that produced
    the published tables, so the reproduction gate in item E is a real check rather than a
    restatement of cached numbers.
    """
    sc = CONDITION[condition]
    out = {m: {} for m in methods}
    for s in seeds:
        summary = _seed_summary(int(s))
        g = summary[summary.scenario == sc]
        for m in methods:
            row = g[g.method == METHOD[m]]
            if row.empty:
                raise KeyError(f'{condition}/{m}: the benchmark did not run {METHOD[m]} '
                               f'under {sc}')
            r = row.iloc[0]
            out[m][int(s)] = {k: float(r[v]) for k, v in METRIC.items()}
    return out


# ---------------------------------------------------------------------------
# Real-data wiring, for item D
# ---------------------------------------------------------------------------
# The package assigns events to train, calibration and test by rotation. Rotation 0 is the
# split the paper reports. The order below is the package's own, copied from
# run_split_robustness.py so that a rotation here means the same thing it means there.
_EVENTS = ['hurricane_harvey', 'hurricane_maria', 'california_wildfires',
           'mexico_earthquake', 'srilanka_floods', 'hurricane_irma', 'iraq_iran_earthquake']
_ROTATIONS = [tuple(_EVENTS[(i + k) % 7] for i in range(7)) for k in range(7)]


def _real_modules():
    if _UPSTREAM not in sys.path:
        sys.path.insert(0, _UPSTREAM)
    cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        import core
        import run_real_data as rrd
        return core, rrd
    finally:
        os.chdir(cwd)


@functools.lru_cache(maxsize=None)
def _rotation_frame(rotation: int):
    """The package's own CrisisMMD frame with roles assigned for this rotation."""
    core, rrd = _real_modules()
    cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        d = rrd.build_frame().copy()
    finally:
        os.chdir(cwd)
    order = _ROTATIONS[rotation]
    assign = {e: ('train' if i < 3 else 'calibration' if i < 5 else 'test')
              for i, e in enumerate(order)}
    d['split'] = d.event_id.map(assign)
    return d


_TEXT_FULL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'crisismmd_text_full.npz')
_ANNOTATIONS = ('/Users/arlina/Downloads/all_projects/Tier-Ready/12_edgewait/data_real/ex/'
                'CrisisMMD_v2.0/annotations')


@functools.lru_cache(maxsize=None)
def _records():
    import pandas as pd
    return pd.read_csv(os.path.join(_UPSTREAM, 'data', 'crisismmd', 'records.csv.gz'))


@functools.lru_cache(maxsize=None)
def _full_text():
    """The full-dimension CLIP text embedding and the raw text, in records order.

    The package ships text_x0..text_x7, a projection of the CLIP text embedding to eight
    components, which is not invertible, so item A's full-dimension and 32-component arms
    cannot be served from it. scripts/build_full_text_embeddings.py recomputes the 512
    dimensional embedding with the encoder named in provenance.json and caches it here; its
    manifest records the check that refitting PCA to eight components on the training events
    reproduces the shipped columns. The raw text is recovered from the CrisisMMD annotation
    tables by image_id and is what the TF-IDF control needs.
    """
    import glob
    import pandas as pd
    if not os.path.exists(_TEXT_FULL):
        raise FileNotFoundError(
            f'{_TEXT_FULL} is missing. Run scripts/build_full_text_embeddings.py first; '
            'the shipped 8-component features cannot stand in for the full embedding.')
    z = np.load(_TEXT_FULL, allow_pickle=False)
    r = _records()
    ids = z['image_id'].astype(str)
    if not np.array_equal(ids, r.image_id.to_numpy().astype(str)):
        raise AssertionError('the cached text embedding is not aligned with records.csv.gz')
    files = sorted(glob.glob(os.path.join(_ANNOTATIONS, '*_final_data.tsv')))
    ann = pd.concat([pd.read_csv(f, sep='\t', dtype=str)[['image_id', 'tweet_text']]
                     for f in files], ignore_index=True).drop_duplicates('image_id')
    text = (r[['image_id']].merge(ann, on='image_id', how='left', validate='one_to_one')
            .tweet_text)
    if text.isna().any():
        raise AssertionError('the tweet text join by image_id is incomplete')
    return z['X'].astype(float), text.to_numpy().astype(str)


def _load_crisismmd_rotation(rotation: int):
    """The workload for one rotation, in the contract the rerun scripts expect.

    X_text_full is the full 512-dimension CLIP text embedding, recomputed by
    scripts/build_full_text_embeddings.py, because that is what the contract says and what
    item A compares against. The 8-component features the package ships are kept beside it as
    X_text_pca8 and the raw text as text_raw for the TF-IDF control.

    X_image_full remains the package's 10-component image features: item A is a text-pathway
    diagnostic and does not read it; item D reads the roles, labels, events and payload
    bytes rather than either feature array.
    """
    d = _rotation_frame(rotation)
    # build_frame packs each modality's reduced features into one column holding an array
    # per row rather than one column per component, so they are stacked here.
    def _stack(col):
        return np.vstack([np.asarray(v, float) for v in d[col].to_numpy()])
    role = d.split.map({'train': 'train', 'calibration': 'cal', 'test': 'test'}).to_numpy()
    X_text_pca8 = _stack('text_x')
    try:
        X_text_full, text_raw = _full_text()
    except FileNotFoundError:
        X_text_full, text_raw = None, None
    if X_text_full is not None and len(X_text_full) != len(d):
        raise AssertionError('text embedding row count does not match the rotation frame')
    s = Split(X_text_full=X_text_full if X_text_full is not None else X_text_pca8,
              X_image_full=_stack('image_x'),
              y=d.y.to_numpy(int), event=d.event_id.to_numpy(),
              role=role,
              bytes_text=d.text_raw_bytes.to_numpy(int),
              bytes_image=d.image_raw_bytes.to_numpy(int))
    s.text_features_are_full = X_text_full is not None
    s.image_features_are_pca_reduced = True
    s.X_text_pca8 = X_text_pca8
    s.text_raw = text_raw
    s.frame = d
    return s


def _derive_wait_threshold(split, forecast_seconds) -> float:
    """T* from the package's own closed form, on calibration windows only."""
    core, rrd = _real_modules()
    d = split.frame
    cal = d[d.split == 'calibration'].drop(columns='split').reset_index(drop=True)
    train = d[d.split == 'train'].drop(columns='split').reset_index(drop=True)
    cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        models = core.fit_models(train, cal, 0)
        cv = core.attach_probabilities(cal, models)
        sc = rrd.scenario('delayed_imagery')
        cal_view = core.scenario_view(cv, sc, 7, models)
        return float(core.derive_wait_threshold(cal_view, sc))
    finally:
        os.chdir(cwd)


def _run_real_budgeting(split, condition: str, seed: int, methods,
                        delay_seconds=None, forecast_seconds=None,
                        class_weight: str | None = None) -> dict:
    """The package's own real-data comparison for one condition and seed.

    The constructed arrival delay and its forecast are written into the test view, so the
    delayed-imagery condition is exercised on real CrisisMMD records that carry no delay of
    their own. Which records are late and by how much is drawn in the caller from the frozen
    protocol, never here.
    """
    core, rrd = _real_modules()
    d = split.frame
    train = d[d.split == 'train'].drop(columns='split').reset_index(drop=True)
    cal = d[d.split == 'calibration'].drop(columns='split').reset_index(drop=True)
    test = d[d.split == 'test'].drop(columns='split').reset_index(drop=True)
    cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        # class_weight carries item A's chosen class handling for item C. It used to be
        # accepted and then dropped on the floor here, so the handled variant silently
        # reproduced the unhandled one. 'none' still delegates to the package's own
        # fit_models, so the current column is the package code path unchanged.
        models = fit_models_handled(core, train, cal, seed, class_weight or 'none')
        cv = core.attach_probabilities(cal, models)
        tv = core.attach_probabilities(test, models)
        sc = rrd.scenario(CONDITION.get(condition, condition))
        test_view = core.scenario_view(tv, sc, seed, models)
        cal_view = core.scenario_view(cv, sc, seed + 7, models)
        if delay_seconds is not None:
            idx = test.index.to_numpy()
            src = np.asarray(delay_seconds, float)
            fsrc = np.asarray(forecast_seconds, float)
            pos = d.index.get_indexer(d[d.split == 'test'].index)
            # The package names these image_delay_actual_s and image_delay_forecast_s and
            # draws them inside scenario_view. Item D replaces that internal draw with the
            # frozen one from its protocol so the exact delays are auditable. Nothing else in
            # the view is touched: image_available is the package's own availability flag and
            # is not a function of the delay, so overwriting it collapsed every policy to
            # text only and made the four methods identical.
            test_view = test_view.copy()
            test_view['image_delay_actual_s'] = src[pos][:len(test_view)]
            test_view['image_delay_forecast_s'] = fsrc[pos][:len(test_view)]
            t_star = float(core.derive_wait_threshold(cal_view, sc))
            test_view['derived_wait_threshold_s'] = t_star
            cal_view = cal_view.copy()
            cal_view['derived_wait_threshold_s'] = t_star
        out = {}
        for m in methods:
            name = METHOD[m]
            res, met, _ = core._run(test_view, cal_view, name, sc, seed, models)
            out[m] = {'errors': float(met['errors']), 'bytes': float(met['mean_bytes']),
                      'p95': float(met['p95_latency_s']),
                      'median': float(met.get('median_latency_s', float('nan'))),
                      'image_sent': float(met.get('image_rate', float('nan')))}
        return out
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Item C: class handling in the real-data pipeline
# ---------------------------------------------------------------------------
# core.fit_models builds every per-modality model with no class handling. Item C has to
# apply item A's chosen handling to all of them. upstream/prototype_v2.3 is reproduced
# unchanged so that its published numbers stay checkable against it, so the variant is
# built here instead of edited into core.py. Everything other than the class handling is
# mirrored from core.fit_models line for line: the same estimators, the same
# hyperparameters, the same absence of feature scaling, the same quantised PCA encoder for
# the feature pathway, the same temperature tuning on calibration rows and the same ridge
# severity estimator. The handled and unhandled variants therefore differ in exactly one
# factor, which is what makes the side-by-side comparison interpretable.
#
# The representation is NOT changed here. Item A's best cell also used the full 512
# dimension text embedding, but Rerun_Spec item C asks only for the class handling to
# carry over. The deployed pipeline's shipped components are kept so that the only moved
# factor is class handling; the representation finding stays item A's, reported separately.

def balanced_resample(X, y, seed: int):
    """Oversample minority classes to the majority count.

    Identical to item A's balanced_resample, so the handling applied to the budgeting
    pipeline is the same operation that was selected in the diagnostic.
    """
    r = np.random.default_rng(int(seed))
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    idx = []
    for c in classes:
        ci = np.flatnonzero(y == c)
        idx.append(r.choice(ci, size=target, replace=len(ci) < target))
    idx = np.concatenate(idx)
    r.shuffle(idx)
    return X[idx], y[idx]


def fit_models_handled(core, train, cal, seed: int, class_handling: str):
    """core.fit_models with item A's class handling applied to every per-modality model.

    class_handling is 'none', 'class_weight' or 'resample'. 'none' delegates straight to
    core.fit_models so the unhandled variant is literally the package's own code path and
    cannot drift from it.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    if class_handling == 'none':
        return core.fit_models(train, cal, seed)
    if class_handling not in ('class_weight', 'resample'):
        raise ValueError(f'unknown class handling: {class_handling!r}')
    cw = 'balanced' if class_handling == 'class_weight' else None
    resample = class_handling == 'resample'

    ytr, ycal = train.y.to_numpy(), cal.y.to_numpy()
    models = {'temps': {}}
    for mi, m in enumerate(core.MODALITIES):
        xtr, xcal = core.stack(train, f'{m}_x'), core.stack(cal, f'{m}_x')
        ld = core.LOCAL_DIMS[m]

        def _fit(est, X, y, s=seed, mi=mi):
            # one resample draw per (modality, pathway) call, seeded from the run seed and
            # the modality index so the three modalities do not share one draw
            if resample:
                X, y = balanced_resample(X, y, s * 100 + mi)
            return est.fit(X, y)

        local = _fit(LogisticRegression(max_iter=700, C=0.75, class_weight=cw,
                                        random_state=seed), xtr[:, :ld], ytr)
        cloud = _fit(LogisticRegression(max_iter=700, C=1.15, class_weight=cw,
                                        random_state=seed), xtr, ytr)
        # the quantised PCA encoder is fitted on the unresampled training rows: it is the
        # transport code path, not a classifier, and resampling it would change what the
        # feature action transmits rather than how the model handles imbalance
        mu = xtr.mean(axis=0)
        _, _, vt = np.linalg.svd(xtr - mu, full_matrices=False)
        comps = vt[:core.FEATURE_DIMS[m]]
        scale = np.abs((xtr - mu) @ comps.T).max(axis=0) / 127.0

        def enc(x, mu=mu, comps=comps, scale=scale):
            return np.clip(np.round(((x - mu) @ comps.T) / scale), -127, 127) * scale

        feat = _fit(LogisticRegression(max_iter=700, C=1.15, class_weight=cw,
                                       random_state=seed), enc(xtr), ytr)
        models[m] = {'local': local, 'cloud': cloud, 'feature': feat, 'encode': enc}
        models['temps'][(m, 'local')] = core.tune_temperature(
            local.predict_proba(xcal[:, :ld]), ycal)
        models['temps'][(m, 'cloud')] = core.tune_temperature(cloud.predict_proba(xcal), ycal)
        models['temps'][(m, 'feature')] = core.tune_temperature(
            feat.predict_proba(enc(xcal)), ycal)

    ptr = np.hstack([core.temperature_scale(
        models[m]['local'].predict_proba(core.stack(train, f'{m}_x')[:, :core.LOCAL_DIMS[m]]),
        models['temps'][(m, 'local')]) for m in core.MODALITIES])
    models['severity'] = Ridge(alpha=1.0).fit(ptr, train.severity_true.to_numpy())
    models['severity_mean'] = float(train.severity_true.mean())
    models['severity_train_r2'] = float(
        models['severity'].score(ptr, train.severity_true.to_numpy()))
    return models
