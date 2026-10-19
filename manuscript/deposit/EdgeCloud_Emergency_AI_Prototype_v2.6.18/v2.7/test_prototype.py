"""Deterministic smoke and integrity tests for the controlled prototype (v2.2)."""

import json

import numpy as np

import core
from core import MODALITIES, SCENARIO_BY_NAME, K_CONCURRENT


def main() -> None:
    seed = 101
    a, b = core.generate_events(seed), core.generate_events(seed)
    assert a.window_id.tolist() == b.window_id.tolist()
    for m in MODALITIES:
        assert np.array_equal(core.stack(a, f"{m}_x"), core.stack(b, f"{m}_x"))

    train, cal, test = core.split_events(a)
    assert set(train.event_id).isdisjoint(cal.event_id)
    assert set(train.event_id).isdisjoint(test.event_id)
    assert set(cal.event_id).isdisjoint(test.event_id)

    models = core.fit_models(train, cal, seed)
    # The feature model sees only the transmitted embedding.
    for m in MODALITIES:
        assert models[m]["feature"].coef_.shape[1] == core.FEATURE_DIMS[m]
    test = core.attach_probabilities(test.groupby("event_id").head(36), models)

    # The transmission policy is label-free: permuting labels leaves actions unchanged.
    sc = SCENARIO_BY_NAME["low_bandwidth"]
    view = core.scenario_view(test, sc, seed, models)
    act = core.select_actions(view, "proposed", sc, seed)
    shuffled = view.copy()
    shuffled["y"] = np.random.default_rng(0).permutation(shuffled.y.to_numpy())
    shuffled["severity_true"] = np.random.default_rng(1).permutation(shuffled.severity_true.to_numpy())
    assert np.array_equal(act, core.select_actions(shuffled, "proposed", sc, seed))

    # Shared slot budget is respected by every budget-compliant method in every scenario.
    for name, scn in SCENARIO_BY_NAME.items():
        v = core.scenario_view(test, scn, seed, models)
        for method in ("proposed", "fixed_confidence_gate", "fixed_rule", "random_within_budget"):
            res = core.run_method(v, method, scn, seed)
            assert not res.slot_over_budget.any(), (name, method)
            assert res.groupby("slot_id").size().max() <= K_CONCURRENT

    # Shared-budget allocation methods (contested-budget experiment) respect a binding slot budget,
    # and the summary floor never holds an available item.
    for b in (700, 1200):
        scb = core.Scenario(name="test_contested", bandwidth_bps=256000, base_latency_s=0.18, budget_bytes=b)
        vb = core.scenario_view(test, scb, seed, models)
        for method in ("all_summary", "fixed_rule_window", "greedy_uncertainty", "greedy_value", "proposed",
                       "proposed_summary_floor", "greedy_value_summary_floor"):
            res = core.run_method(vb, method, scb, seed)
            assert not res.slot_over_budget.any(), (b, method)
        act = core.select_actions(vb, "proposed_summary_floor", scb, seed)
        assert (act[vb[["image_available", "text_available", "sensor_available"]].to_numpy()] > 0).all()

    # Admissibility rejects misaligned text whose metadata was not spoofed.
    scn = SCENARIO_BY_NAME["misaligned_evidence"]
    v = core.scenario_view(test, scn, seed, models)
    mis = v.text_misaligned.to_numpy()
    assert mis.any() and (~v.text_admissible.to_numpy()[~mis]).sum() == 0

    # Fixed rule with admissibility never transmits an inadmissible item.
    act = core.select_actions(v, "fixed_rule_admissible", scn, seed)
    for k, m in enumerate(MODALITIES):
        assert (act[~v[f"{m}_admissible"].to_numpy(), k] == 0).all()

    # The derived wait threshold uses calibration data only: it ignores labels, and the rule
    # waits exactly for images whose forecast delay is below it.
    dsc = SCENARIO_BY_NAME["delayed_imagery"]
    cal_v = core.scenario_view(core.attach_probabilities(cal.groupby("event_id").head(36), models), dsc, seed + 7, models)
    t_star = core.derive_wait_threshold(cal_v, dsc)
    shuffled = cal_v.copy()
    shuffled["y"] = np.random.default_rng(2).permutation(shuffled.y.to_numpy())
    assert t_star == core.derive_wait_threshold(shuffled, dsc) and t_star > 0
    dv = core.scenario_view(test, dsc, seed, models)
    dv["derived_wait_threshold_s"] = t_star
    act = core.select_actions(dv, "derived_threshold", dsc, seed)
    waits = (act[:, 0] > 0) & dv.image_available.to_numpy()
    assert np.array_equal(waits, (dv.image_delay_forecast_s.to_numpy() < t_star) & dv.image_available.to_numpy())

    # Envelopes carry the required provenance fields and completeness is computed, not asserted.
    res = core.annotate(core.run_method(v, "proposed", scn, seed), 0.1, "test", envelopes=True)
    assert res.provenance_complete.all()
    env = json.loads(res.evidence_envelope.iloc[0])
    for key in ("model_version", "policy_version", "calibration_version", "status_rule", "incident", "evidence"):
        assert key in env
    for m in MODALITIES:
        for key in ("observed_time_s", "cell", "hazard", "source_digest_sha256", "reason_code"):
            assert key in env["evidence"][m]

    print("All prototype tests passed.")


if __name__ == "__main__":
    main()
