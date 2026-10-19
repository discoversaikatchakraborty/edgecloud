"""Controlled prototype for EdgeCloud Emergency AI (version 2.1).

The benchmark is synthetic by design. It tests system behavior under known
missingness, disagreement, misalignment, delay, and bandwidth conditions. It
does not claim field effectiveness or replace evaluation on real disaster data.

Changes relative to version 1 (see CHANGELOG.md for the full list):
  * every transmitted representation is scored by a model that sees only that
    representation (the "feature" action no longer mixes in cloud-model output);
  * severity used by the policy and by review routing is estimated from edge
    model outputs, never from the generating label;
  * the byte budget is shared by K concurrent incident windows and solved as a
    multiple-choice knapsack by dynamic programming; the link is shared, so
    queueing delay is modeled for every method;
  * image arrival delay is stochastic, and the policy sees only a noisy
    forecast of it; the latency term includes transfer time;
  * split-conformal sets (LAC score) are calibrated per method and scenario on
    calibration events processed under the same scenario; shift is reported
    separately by applying the normal-condition threshold;
  * spatial/temporal/taxonomy admissibility is implemented and tested;
  * ablations remove a component everywhere it is used (policy and fusion).

Added in 2.1: two baselines that isolate the optimizer from the information it uses
(a constant wait threshold derived from the closed form on calibration events, and the
fixed rule with the admissibility filter), a sweep of that threshold, and a
sensitivity analysis of the review rule (conformal error level, conflict threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
POLICY = CONFIG["policy"]
BASELINE_CFG = CONFIG["baselines"]
MODEL_VERSION = CONFIG["model_version"]
RECENCY_TAU = CONFIG["recency_tau_seconds"]
ADMIT = CONFIG["admissibility"]
K_CONCURRENT = int(CONFIG["concurrency_k"])

CLASSES = np.array([0, 1, 2])
CLASS_NAMES = {0: "no_event", 1: "flood", 2: "infrastructure_impact"}


def set_classes(names):
    """v2.7: run the unchanged pipeline on a label with any number of classes (the synthetic
    generator keeps three). Call before fitting models; every class-indexed array reads K()."""
    global CLASSES, CLASS_NAMES
    CLASSES = np.arange(len(names))
    CLASS_NAMES = {i: n for i, n in enumerate(names)}


def K() -> int:
    return len(CLASSES)


def PCOLS():
    return [f"p{i}" for i in range(K())]


USE_PRED_CONTRIB = False   # v2.7: multiply evidence value by the held-out predictive contribution of the modality
HAZARDS = ("flood", "storm", "earthquake")
MODALITIES = ("image", "text", "sensor")
ACTIONS = ("hold", "summary", "feature", "raw")
TIER = np.array([0.0, 0.34, 0.64, 1.0])
WEIGHTS = {"image": 0.40, "text": 0.35, "sensor": 0.25}
LOCAL_DIMS = {"image": 5, "text": 4, "sensor": 3}
FULL_DIMS = {"image": 10, "text": 8, "sensor": 6}
FEATURE_DIMS = {"image": 4, "text": 3, "sensor": 3}
SUMMARY_BYTES = CONFIG["byte_model"]["summary"]
FEATURE_BYTES = CONFIG["byte_model"]["feature"]
RAW_RANGE = CONFIG["byte_model"]["raw_range"]
COMBOS = np.array(list(product(range(4), repeat=3)))  # (64, 3) action indices
BASE_AGE = {"image": 60.0, "text": 30.0, "sensor": 10.0}
FIXED_OVERHEAD_S = 0.045

METHODS = (
    "image_only", "text_only", "cloud_transmit_all", "fixed_confidence_gate",
    "fixed_compression", "fixed_rule", "random_within_budget", "proposed",
)
BUDGET_COMPLIANT = {
    "image_only": True, "text_only": True, "cloud_transmit_all": False,
    "fixed_confidence_gate": True, "fixed_compression": True, "fixed_rule": True,
    "random_within_budget": True, "proposed": True, "wait_if_uncertain": True,
    "derived_threshold": True, "fixed_rule_admissible": True,
    "all_summary": True, "fixed_rule_window": True, "proposed_summary_floor": True, "greedy_value_summary_floor": True, "greedy_uncertainty": True, "greedy_value": True,
}
# Baselines that differ from the fixed rule only under one condition are run only there.
EXTRA_METHODS = {"delayed_imagery": ("wait_if_uncertain", "derived_threshold"),
                 "misaligned_evidence": ("fixed_rule_admissible",)}


@dataclass(frozen=True)
class Scenario:
    name: str
    bandwidth_bps: float
    base_latency_s: float
    budget_bytes: int
    image_delay_prob: float = 0.0
    image_delay_median_s: float = 0.0
    delay_forecast_sigma: float = 0.0
    missing_image: float = 0.0
    missing_text: float = 0.0
    missing_sensor: float = 0.0
    noisy_text: float = 0.0
    contradictory_text: float = 0.0
    misaligned_text: float = 0.0
    spoofed_metadata: float = 0.0
    outage_probability: float = 0.0
    interruption_s: float = 0.0
    packet_loss: float = 0.0          # per-packet loss probability; 0.0 reproduces earlier results
    link_trace: str = ""              # measured capacity trace in data/traces/; "" keeps the constant rate
    controllers: int = 1              # independent allocators per slot; 1 is the centralised design
    rng_name: str = ""                # draw the scenario view as if named this; "" means use `name`


SCENARIOS = tuple(Scenario(**spec) for spec in CONFIG["scenarios"])
SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


POLICY_VERSION = hashlib.sha256(json.dumps(CONFIG["policy"], sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- utilities

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def temperature_scale(prob: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.log(np.clip(prob, 1e-8, 1.0)) / t)


def tune_temperature(prob: np.ndarray, y: np.ndarray) -> float:
    best_t, best = 1.0, np.inf
    for t in np.linspace(0.55, 2.8, 46):
        p = temperature_scale(prob, float(t))
        loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()
        if loss < best:
            best_t, best = float(t), float(loss)
    return best_t


def js_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence (natural log)."""
    m = 0.5 * (p + q)
    a = np.sum(p * np.log(np.clip(p, 1e-12, None) / np.clip(m, 1e-12, None)), axis=-1)
    b = np.sum(q * np.log(np.clip(q, 1e-12, None) / np.clip(m, 1e-12, None)), axis=-1)
    return 0.5 * a + 0.5 * b


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.vstack(df[col].to_numpy())


# --------------------------------------------------------------------------- data

def generate_events(seed: int, events: int = 12, rows_per_event: int = 180, noise_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = {}
    for m in MODALITIES:
        d = FULL_DIMS[m]
        base = rng.normal(0, 0.35, (K(), d))
        for c in CLASSES:
            base[c, c % d] += 2.5
            base[c, (c + 3) % d] += 1.5
        centers[m] = base
    rows = []
    for event in range(events):
        shift = {m: rng.normal(0, 0.38, FULL_DIMS[m]) for m in MODALITIES}
        priors = rng.dirichlet(np.array([5.0, 4.0, 2.5]))
        cx, cy = rng.integers(0, 400, 2)
        hazard = HAZARDS[int(rng.integers(0, len(HAZARDS)))]
        for j in range(rows_per_event):
            y = int(rng.choice(CLASSES, p=priors))
            severity = float(np.clip(rng.beta(2.2 + 1.2 * y, 2.3), 0.05, 1.0))
            rec = {
                "event_id": f"D{event:02d}", "window_id": f"D{event:02d}_{j:04d}", "y": y,
                "severity_true": severity, "hazard": hazard,
                "cell_x": int(cx + rng.integers(-6, 7)), "cell_y": int(cy + rng.integers(-6, 7)),
                "decision_time_s": float(rng.uniform(0, 86400)),
            }
            for m in MODALITIES:
                quality = float(np.clip(rng.beta(5.0, 2.0), 0.15, 1.0))
                rel = float(np.clip(rng.normal({"image": .88, "text": .74, "sensor": .82}[m], .07), .45, .99))
                x = centers[m][y] + shift[m] + rng.normal(0, 0.78 * noise_scale / quality, FULL_DIMS[m])
                rec[f"{m}_quality"] = quality
                rec[f"{m}_reliability"] = rel
                rec[f"{m}_x"] = x
                lo, hi = RAW_RANGE[m]
                rec[f"{m}_raw_bytes"] = int(rng.integers(lo, hi))
            rows.append(rec)
    return pd.DataFrame(rows)


def split_events(df: pd.DataFrame):
    s = CONFIG["event_splits"]
    return (df[df.event_id.isin(s["train"])].copy(), df[df.event_id.isin(s["calibration"])].copy(),
            df[df.event_id.isin(s["test"])].copy())


# --------------------------------------------------------------------------- models

def fit_models(train: pd.DataFrame, cal: pd.DataFrame, seed: int) -> dict:
    """Local (edge) model, feature-only cloud model, raw cloud model, severity estimator.

    The feature model is trained on the quantized PCA embedding that the
    feature action actually transmits, so it cannot see information that was
    not sent.
    """
    ytr, ycal = train.y.to_numpy(), cal.y.to_numpy()
    models = {"temps": {}}
    for m in MODALITIES:
        xtr, xcal = stack(train, f"{m}_x"), stack(cal, f"{m}_x")
        local = LogisticRegression(max_iter=700, C=0.75, random_state=seed).fit(xtr[:, :LOCAL_DIMS[m]], ytr)
        cloud = LogisticRegression(max_iter=700, C=1.15, random_state=seed).fit(xtr, ytr)
        mu = xtr.mean(axis=0)
        _, _, vt = np.linalg.svd(xtr - mu, full_matrices=False)
        comps = vt[:FEATURE_DIMS[m]]
        scale = np.abs((xtr - mu) @ comps.T).max(axis=0) / 127.0
        enc = lambda x, mu=mu, comps=comps, scale=scale: np.clip(np.round(((x - mu) @ comps.T) / scale), -127, 127) * scale
        feat = LogisticRegression(max_iter=700, C=1.15, random_state=seed).fit(enc(xtr), ytr)
        models[m] = {"local": local, "cloud": cloud, "feature": feat, "encode": enc}
        # v2.7: held-out predictive contribution of the modality for this label, from the
        # calibration events only: balanced accuracy of the cloud model above chance, on [0, 1].
        # A majority-class predictor scores exactly 0 whatever the class imbalance.
        from sklearn.metrics import balanced_accuracy_score
        ba = balanced_accuracy_score(ycal, cloud.predict(xcal))
        models.setdefault("pred_contrib_raw", {})[m] = float(max(0.0, (ba - 1.0 / K()) / (1.0 - 1.0 / K())))
        models["temps"][(m, "local")] = tune_temperature(local.predict_proba(xcal[:, :LOCAL_DIMS[m]]), ycal)
        models["temps"][(m, "cloud")] = tune_temperature(cloud.predict_proba(xcal), ycal)
        models["temps"][(m, "feature")] = tune_temperature(feat.predict_proba(enc(xcal)), ycal)
    # Severity estimator: edge-side regression from local class probabilities.
    ptr = np.hstack([temperature_scale(models[m]["local"].predict_proba(stack(train, f"{m}_x")[:, :LOCAL_DIMS[m]]),
                                       models["temps"][(m, "local")]) for m in MODALITIES])
    models["severity"] = Ridge(alpha=1.0).fit(ptr, train.severity_true.to_numpy())
    models["severity_mean"] = float(train.severity_true.mean())
    models["severity_train_r2"] = float(models["severity"].score(ptr, train.severity_true.to_numpy()))
    # Relative to the best modality, so a workload where every modality carries signal is
    # left close to unchanged and only a modality with no held-out signal is driven to zero.
    top = max(models["pred_contrib_raw"].values())
    models["pred_contrib"] = {m: (c / top if top > 0 else 1.0) for m, c in models["pred_contrib_raw"].items()}
    return models


SOURCE_DIGEST_DECIMALS = 9


def source_digest(v) -> str:
    """SHA-256 of a source observation, quantized before hashing.

    Hashing raw IEEE-754 bytes makes the digest depend on the last bit of every
    element, so it changes across CPU architectures even when every reported
    quantity is identical. Rounding to SOURCE_DIGEST_DECIMALS decimals is far
    coarser than floating-point noise (about 1e-15 at this scale) and far finer
    than any difference the benchmark can express, so the digest identifies the
    same observation on any host. Zeros are normalized so that -0.0 and 0.0
    cannot produce different digests.
    """
    x = np.round(np.asarray(v, dtype="<f8"), SOURCE_DIGEST_DECIMALS)
    x = np.where(x == 0.0, 0.0, x)
    return hashlib.sha256(x.tobytes()).hexdigest()


def attach_probabilities(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = df.copy()
    t = models["temps"]
    for m in MODALITIES:
        x = stack(out, f"{m}_x")
        mm = models[m]
        out[f"{m}_local_p"] = list(temperature_scale(mm["local"].predict_proba(x[:, :LOCAL_DIMS[m]]), t[(m, "local")]))
        out[f"{m}_feature_p"] = list(temperature_scale(mm["feature"].predict_proba(mm["encode"](x)), t[(m, "feature")]))
        out[f"{m}_cloud_p"] = list(temperature_scale(mm["cloud"].predict_proba(x), t[(m, "cloud")]))
        out[f"{m}_source_digest"] = [source_digest(v) for v in out[f"{m}_x"]]
        out[f"{m}_pred_contrib"] = float(models.get("pred_contrib", {}).get(m, 1.0))
    return out


# --------------------------------------------------------------------------- scenarios

def scenario_view(df: pd.DataFrame, sc: Scenario, seed: int, models: dict) -> pd.DataFrame:
    # The perturbation stream is seeded from the scenario name. A variant that means
    # to hold a condition fixed and change one factor must therefore inherit that
    # condition's name here, or it silently draws a different workload and is not a
    # controlled comparison at all.
    key = sc.rng_name or sc.name
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % 10000
    rng = np.random.default_rng(seed + offset)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    for m, miss in (("image", sc.missing_image), ("text", sc.missing_text), ("sensor", sc.missing_sensor)):
        out[f"{m}_available"] = rng.random(n) >= miss
        out[f"{m}_age_s"] = rng.uniform(0.0, BASE_AGE[m], n)
        out[f"{m}_cell_x"], out[f"{m}_cell_y"] = out.cell_x, out.cell_y
        out[f"{m}_obs_time_s"] = out.decision_time_s - out[f"{m}_age_s"]
        out[f"{m}_hazard"] = out.hazard
    # Stochastic image arrival delay; the policy sees only a noisy forecast.
    if sc.image_delay_prob > 0:
        late = rng.random(n) < sc.image_delay_prob
        actual = np.where(late, sc.image_delay_median_s * rng.lognormal(0, 0.5, n), rng.uniform(0, 30, n))
        forecast = actual * rng.lognormal(0, sc.delay_forecast_sigma, n)
    else:
        actual = np.zeros(n)
        forecast = np.zeros(n)
    out["image_delay_actual_s"], out["image_delay_forecast_s"] = actual, forecast
    out["link_interruption_s"] = np.where(rng.random(n) < sc.outage_probability, sc.interruption_s, 0.0)
    prob_cols = ("text_local_p", "text_feature_p", "text_cloud_p")
    if sc.noisy_text:
        idx = rng.random(n) < sc.noisy_text
        for col in prob_cols:
            arr = stack(out, col)
            arr[idx] = 0.55 * arr[idx] + 0.45 / 3
            out[col] = list(arr / arr.sum(axis=1, keepdims=True))
        # Imperfect, overlapping quality indicator (not an oracle flag).
        f = np.where(idx, rng.uniform(0.55, 0.95, n), rng.uniform(0.80, 1.05, n))
        out["text_quality"] = np.clip(out.text_quality * f, 0.15, 1.0)
    if sc.contradictory_text:
        idx = rng.random(n) < sc.contradictory_text
        wrong = (out.y.to_numpy() + 1) % 3
        for col in prob_cols:
            arr = stack(out, col)
            arr[idx] = 0.08
            arr[np.where(idx)[0], wrong[idx]] = 0.84
            out[col] = list(arr / arr.sum(axis=1, keepdims=True))
        f = np.where(idx, rng.uniform(0.55, 0.95, n), rng.uniform(0.80, 1.05, n))
        out["text_reliability"] = np.clip(out.text_reliability * f, 0.30, 0.99)
    out["text_misaligned"] = False
    if sc.misaligned_text:
        idx = np.where(rng.random(n) < sc.misaligned_text)[0]
        ev = out.event_id.to_numpy()
        src = out.copy()
        for i in idx:
            d = int(rng.choice(np.where(ev != ev[i])[0]))
            for col in prob_cols + ("text_quality", "text_reliability", "text_source_digest"):
                out.at[i, col] = src.at[d, col]
            if rng.random() >= sc.spoofed_metadata:
                out.at[i, "text_cell_x"], out.at[i, "text_cell_y"] = src.at[d, "cell_x"], src.at[d, "cell_y"]
                out.at[i, "text_obs_time_s"] = src.at[d, "decision_time_s"] - src.at[d, "text_age_s"]
                out.at[i, "text_hazard"] = src.at[d, "hazard"]
            out.at[i, "text_misaligned"] = True
    # Admissibility: spatial, temporal and taxonomy compatibility (metadata only, never labels).
    for m in MODALITIES:
        dist = np.maximum(np.abs(out[f"{m}_cell_x"] - out.cell_x), np.abs(out[f"{m}_cell_y"] - out.cell_y))
        dt = np.abs(out.decision_time_s - out[f"{m}_obs_time_s"])
        out[f"{m}_admissible"] = ((dist <= ADMIT["max_cell_distance"]) & (dt <= ADMIT["time_tolerance_s"][m])
                                  & (out[f"{m}_hazard"] == out.hazard)).to_numpy()
    # Edge severity estimate from local probabilities of admissible evidence present at the edge.
    feats = []
    for m in MODALITIES:
        p = stack(out, f"{m}_local_p")
        present = out[f"{m}_available"].to_numpy() & out[f"{m}_admissible"].to_numpy()
        if m == "image":
            present &= out.image_delay_forecast_s.to_numpy() <= 0
        p[~present] = 1.0 / 3
        feats.append(p)
    out["severity"] = np.clip(models["severity"].predict(np.hstack(feats)), 0.0, 1.0)
    out["severity_mean_train"] = models["severity_mean"]
    # Concurrency slots: K windows of one event, ordered by time, share a budget and a link.
    out = out.sort_values(["event_id", "decision_time_s"]).reset_index(drop=True)
    out["slot"] = out.groupby("event_id").cumcount() // K_CONCURRENT
    out["slot_id"] = out.event_id + "_S" + out.slot.astype(str).str.zfill(3)
    return out


# --------------------------------------------------------------------------- policy


# --------------------------------------------------------------- link model
MTU_BYTES = 1500
RETRANSMIT_TIMEOUT_S = 0.2
_TRACE_CACHE: dict[str, np.ndarray] = {}


def load_trace(name: str) -> np.ndarray:
    """Cumulative bytes deliverable by each millisecond of a measured capacity trace.

    A Mahimahi-format trace lists, one per line, the millisecond at which one
    MTU-sized packet could be delivered. Accumulating those gives a capacity
    function C(t); inverting it by search gives the time to deliver a given
    number of bytes. The trace is replayed unmodified.
    """
    if name not in _TRACE_CACHE:
        path = Path(__file__).resolve().parent / "data" / "traces" / name
        ms = np.asarray(path.read_text().split(), dtype=np.int64)
        span = int(ms[-1]) + 1
        per_ms = np.bincount(ms, minlength=span).astype(np.float64)
        _TRACE_CACHE[name] = np.cumsum(per_ms) * MTU_BYTES
    return _TRACE_CACHE[name]


def trace_offset_ms(scenario_name: str, slot_key: int) -> int:
    """Reproducible starting offset into a trace. Never uses Python's salted hash()."""
    base = sum((i + 1) * ord(ch) for i, ch in enumerate(scenario_name))
    return (base * 7919 + slot_key * 104729) % 600_000


def serialization_time(cum_bytes: np.ndarray, sc: Scenario, offset_ms: int = 0) -> np.ndarray:
    """Seconds to deliver each cumulative byte count, under the scenario's link.

    With no trace and no loss this is exactly cum_bytes * 8 / bandwidth, which is
    what every earlier result used. Loss inflates the bytes actually placed on the
    wire, because a lost packet is retransmitted.
    """
    eff = cum_bytes / (1.0 - sc.packet_loss) if sc.packet_loss else cum_bytes
    if not sc.link_trace:
        return eff * 8.0 / sc.bandwidth_bps
    cap = load_trace(sc.link_trace)
    base = cap[offset_ms % len(cap)]
    idx = np.searchsorted(cap, base + eff, side="left")
    idx = np.minimum(idx, len(cap) - 1)
    return (idx - (offset_ms % len(cap))) / 1000.0


def retransmission_stall(bytes_sent: np.ndarray, sc: Scenario) -> np.ndarray:
    """Timeout cost of lost packets, which grows with payload size.

    Expected lost packets for a payload of n packets is n * p / (1 - p), and each
    costs one retransmission timeout. This is what makes a raw transmission lose
    disproportionately more time than a summary under the same loss rate.
    """
    if not sc.packet_loss:
        return np.zeros(len(bytes_sent))
    packets = np.ceil(bytes_sent / MTU_BYTES)
    return packets * sc.packet_loss / (1.0 - sc.packet_loss) * RETRANSMIT_TIMEOUT_S


def _one_bytes(a: int, k: int, raw: int) -> int:
    m = MODALITIES[k]
    return (0, SUMMARY_BYTES[m], FEATURE_BYTES[m], int(raw))[int(a)]


def _bytes_table(v: pd.DataFrame, m: str) -> np.ndarray:
    n = len(v)
    return np.stack([np.zeros(n), np.full(n, SUMMARY_BYTES[m]), np.full(n, FEATURE_BYTES[m]),
                     v[f"{m}_raw_bytes"].to_numpy().astype(float)], axis=1)


def _evidence_values(v: pd.DataFrame, ablation: str | None):
    """Per-modality evidence value (before the action-fidelity multiplier) and forecast delay.

    Shared by the proposed policy and by the closed-form wait threshold, so both use one definition.
    """
    n = len(v)
    sev = np.full(n, float(v.severity_mean_train.iloc[0])) if ablation == "no_severity" else v.severity.to_numpy()
    pending_img = v.image_delay_forecast_s.to_numpy() > 0
    elig = np.stack([v[f"{m}_available"].to_numpy() & (True if ablation == "no_admissibility" else v[f"{m}_admissible"].to_numpy())
                     for m in MODALITIES], axis=1)
    loc = np.stack([stack(v, f"{m}_local_p") for m in MODALITIES], axis=1)  # (n,3,3)
    present = elig.copy()
    present[:, 0] &= ~pending_img
    w = present[..., None].astype(float)
    mean_local = np.where(w.sum(1) > 0, (loc * w).sum(1) / np.clip(w.sum(1), 1e-9, None), 1 / K())
    partial_conf = mean_local.max(axis=1)
    out = []
    for k, m in enumerate(MODALITIES):
        p = loc[:, k]
        unc = 1.0 - p.max(axis=1)
        nov = np.zeros(n) if ablation == "no_novelty" else js_rows(p, mean_local)
        q = np.ones(n) if ablation == "no_quality" else v[f"{m}_quality"].to_numpy().copy()
        r = np.ones(n) if ablation == "no_reliability" else v[f"{m}_reliability"].to_numpy()
        age = v[f"{m}_age_s"].to_numpy()
        fresh = np.ones(n) if ablation == "no_recency" else np.exp(-age / RECENCY_TAU[m])
        delay = np.zeros(n)
        if m == "image":
            # A pending image has no local output yet: its value uses the uncertainty of the
            # evidence already present, a prior quality (Beta(5,2) mean), and zero novelty.
            unc = np.where(pending_img, 1.0 - partial_conf, unc)
            if ablation != "no_quality":
                q = np.where(pending_img, 5.0 / 7.0, q)
            nov = np.where(pending_img, 0.0, nov)
            delay = v.image_delay_forecast_s.to_numpy()
        core_val = sev * r * (POLICY["base_value"] + POLICY["uncertainty_weight"] * unc
                              + POLICY["quality_weight"] * q + POLICY["novelty_weight"] * nov) * fresh
        if USE_PRED_CONTRIB and ablation != "no_pred_contrib" and f"{m}_pred_contrib" in v:
            # v2.7: a modality that carries no held-out predictive signal for the label is worth
            # nothing to send, whatever its severity, reliability, recency, quality or novelty say.
            core_val = core_val * v[f"{m}_pred_contrib"].to_numpy()
        out.append((core_val, delay))
    return out, elig


def proposed_options(v: pd.DataFrame, sc: Scenario, ablation: str | None, lam: float):
    """Utility and byte cost of all 64 action combinations for every window."""
    n = len(v)
    values, elig = _evidence_values(v, ablation)
    vals, byts = [], []
    for k, m in enumerate(MODALITIES):
        core_val, delay = values[k]
        b = _bytes_table(v, m)
        latency_term = b * 8.0 / sc.bandwidth_bps + delay[:, None] * (np.arange(4)[None, :] > 0)
        val = core_val[:, None] * TIER[None, :] - POLICY["byte_cost_weight"] * b / sc.budget_bytes - lam * latency_term
        val[:, 0] = 0.0
        vals.append(val)
        byts.append(b)
    util = sum(vals[k][:, COMBOS[:, k]] for k in range(3))
    tot_bytes = sum(byts[k][:, COMBOS[:, k]] for k in range(3))
    feasible = np.ones((n, len(COMBOS)), dtype=bool)
    for k in range(3):
        feasible &= (COMBOS[:, k] == 0)[None, :] | elig[:, k][:, None]
    all_hold = (COMBOS == 0).all(axis=1)
    feasible &= ~(all_hold[None, :] & elig.any(axis=1)[:, None])
    return util, tot_bytes, feasible


def image_wait_threshold(v: pd.DataFrame, sc: Scenario, lam: float | None = None, action: int = 1) -> np.ndarray:
    """Closed-form wait threshold (Section III-D) for one image action, per window, in seconds.

    Sending the image with `action` has positive net utility exactly when the forecast delay is below
    this value. The derived-threshold baseline collapses it to one constant chosen on calibration events.
    """
    lam = POLICY["latency_penalty_per_second"] if lam is None else lam
    values, _ = _evidence_values(v, None)
    core_val = values[0][0]
    b = _bytes_table(v, "image")[:, action]
    return (core_val * TIER[action] - POLICY["byte_cost_weight"] * b / sc.budget_bytes) / lam - 8.0 * b / sc.bandwidth_bps


def derive_wait_threshold(cal_view: pd.DataFrame, sc: Scenario) -> float:
    """One constant threshold: the median closed-form threshold for an image summary over calibration
    windows whose image is pending. Uses calibration events only, never test data or labels."""
    pending = cal_view.image_delay_forecast_s.to_numpy() > 0
    return float(np.median(image_wait_threshold(cal_view, sc)[pending]))



def controller_groups(idx: np.ndarray, sc: Scenario) -> list:
    """Disjoint window groups, one per independent controller.

    With one controller this returns the whole slot, which is the centralised
    allocator every other experiment uses. With C > 1 the slot's windows are dealt
    round-robin to C controllers that do not coordinate, and each controller gets a
    share of the byte budget proportional to the number of windows it owns, so the
    total budget is unchanged. The physical link is still shared by all of them:
    partitioning control does not partition the medium.
    """
    if sc.controllers <= 1:
        return [idx]
    return [g for g in (idx[c::sc.controllers] for c in range(sc.controllers)) if len(g)]


def solve_slot(util: np.ndarray, byts: np.ndarray, feas: np.ndarray, budget: float) -> np.ndarray:
    """Exact multiple-choice knapsack for one slot; bytes are discretized conservatively (rounded up)."""
    k = util.shape[0]
    unit = max(1.0, budget / 2000.0)
    cap = int(np.floor(budget / unit))
    bu = np.ceil(byts / unit).astype(int)
    NEG = -1e18
    dp = np.full(cap + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((k, cap + 1), dtype=int)
    prev = np.zeros((k, cap + 1), dtype=int)
    grid = np.arange(cap + 1)
    for i in range(k):
        new = np.full(cap + 1, NEG)
        bo = np.zeros(cap + 1, dtype=int)
        bp = np.zeros(cap + 1, dtype=int)
        for o in np.where(feas[i] & (bu[i] <= cap))[0]:
            c = bu[i, o]
            cand = np.full(cap + 1, NEG)
            cand[c:] = dp[: cap + 1 - c] + util[i, o]
            better = cand > new
            new[better] = cand[better]
            bo[better] = o
            bp[better] = grid[better] - c
        dp, choice[i], prev[i] = new, bo, bp
    b = int(np.argmax(dp))
    if dp[b] <= NEG / 2:
        return np.zeros(k, dtype=int)
    picks = np.zeros(k, dtype=int)
    for i in range(k - 1, -1, -1):
        picks[i] = choice[i, b]
        b = prev[i, b]
    return picks


def select_actions(v: pd.DataFrame, method: str, sc: Scenario, seed: int, ablation: str | None = None,
                   lam: float | None = None) -> np.ndarray:
    """Return (n, 3) action indices into ACTIONS."""
    n = len(v)
    lam = POLICY["latency_penalty_per_second"] if lam is None else lam
    avail = np.stack([v[f"{m}_available"].to_numpy() for m in MODALITIES], axis=1)
    raw = np.stack([v[f"{m}_raw_bytes"].to_numpy() for m in MODALITIES], axis=1)
    act = np.zeros((n, 3), dtype=int)
    if method == "image_only":
        act[:, 0] = np.where(avail[:, 0], 1, 0)
    elif method == "text_only":
        act[:, 1] = np.where(avail[:, 1], 1, 0)
    elif method == "cloud_transmit_all":
        act = np.where(avail, 3, 0)
    elif method == "fixed_compression":
        act = np.where(avail, 2, 0)
    elif method in ("fixed_rule", "wait_if_uncertain", "derived_threshold", "fixed_rule_admissible"):
        act = np.where(avail, np.array([1, 3, 3])[None, :], 0)
        if method == "derived_threshold":
            # Fixed rule, but wait for a late image only if its forecast delay is below one constant
            # threshold derived from the closed form on calibration events (see derive_wait_threshold).
            act[v.image_delay_forecast_s.to_numpy() >= v.derived_wait_threshold_s.to_numpy(), 0] = 0
        if method == "fixed_rule_admissible":
            # Fixed rule with the same metadata admissibility filter the proposed method applies.
            for k, m in enumerate(MODALITIES):
                act[~v[f"{m}_admissible"].to_numpy(), k] = 0
        if method == "wait_if_uncertain":
            pending = v.image_delay_forecast_s.to_numpy() > 0
            loc = np.stack([stack(v, f"{m}_local_p") for m in ("text", "sensor")], axis=1)
            w = avail[:, 1:, None].astype(float)
            conf = np.where(w.sum(1) > 0, (loc * w).sum(1) / np.clip(w.sum(1), 1e-9, None), 1 / 3).max(1)
            act[pending & (conf >= BASELINE_CFG["wait_if_uncertain_confidence"]), 0] = 0
    elif method == "all_summary":
        act = np.where(avail, 1, 0)
    elif method == "fixed_rule_window":
        # Fixed rule; within each window downgrade the largest non-summary item until it fits its share.
        act = np.where(avail, np.array([1, 3, 3])[None, :], 0)
        for i in range(n):
            while sum(_one_bytes(act[i, k], k, raw[i, k]) for k in range(3)) > sc.budget_bytes:
                ks = [k for k in range(3) if act[i, k] >= 2]
                if not ks:
                    break
                act[i, max(ks, key=lambda k: _one_bytes(act[i, k], k, raw[i, k]))] = 1
    elif method == "greedy_uncertainty":
        # Shared slot budget: start from summaries, then upgrade items across the slot in order of
        # local uncertainty (text/sensor to raw, image to feature) while they fit.
        act = np.where(avail, 1, 0)
        unc = np.stack([1.0 - stack(v, f"{m}_local_p").max(axis=1) for m in MODALITIES], axis=1)
        target = np.array([2, 3, 3])
        for _, slot in v.groupby("slot_id").indices.items():
            for idx in controller_groups(np.sort(slot), sc):
                rem = sc.budget_bytes * len(idx) - sum(_one_bytes(act[i, k], k, raw[i, k]) for i in idx for k in range(3))
                cands = [(-unc[i, k], i, k) for i in idx for k in range(3) if avail[i, k]]
                for _, i, k in sorted(cands):
                    extra = _one_bytes(target[k], k, raw[i, k]) - _one_bytes(act[i, k], k, raw[i, k])
                    if extra <= rem:
                        act[i, k] = target[k]
                        rem -= extra
    elif method in ("greedy_value", "greedy_value_summary_floor"):
        # Proposed utility, standard greedy heuristic for the multiple-choice knapsack.
        util, byts, feas = proposed_options(v, sc, ablation, lam)
        if method == "greedy_value_summary_floor":
            elig = np.stack([avail[:, k] & v[f"{m}_admissible"].to_numpy() for k, m in enumerate(MODALITIES)], axis=1)
            for k in range(3):
                feas &= (COMBOS[:, k] > 0)[None, :] | ~elig[:, k][:, None]
        for _, idx in v.groupby("slot_id").indices.items():
            idx = np.sort(idx)
            u, b, f = util[idx], byts[idx], feas[idx]
            cur = np.array([np.lexsort((-u[j][f[j]], b[j][f[j]]))[0] for j in range(len(idx))])
            cur = np.array([np.where(f[j])[0][cur[j]] for j in range(len(idx))])
            rem = sc.budget_bytes * len(idx) - b[np.arange(len(idx)), cur].sum()
            while True:
                best, pick = 0.0, None
                for j in range(len(idx)):
                    du = u[j] - u[j, cur[j]]
                    db = b[j] - b[j, cur[j]]
                    ok = f[j] & (du > 1e-12) & (db <= rem)
                    if not ok.any():
                        continue
                    ratio = np.where(ok, np.where(db > 0, du / np.maximum(db, 1e-9), np.inf), -np.inf)
                    o = int(np.argmax(ratio))
                    if ratio[o] > best:
                        best, pick = ratio[o], (j, o)
                if pick is None:
                    break
                j, o = pick
                rem -= b[j, o] - b[j, cur[j]]
                cur[j] = o
            act[idx] = COMBOS[cur]
    elif method == "fixed_confidence_gate":
        for k, m in enumerate(MODALITIES):
            conf = stack(v, f"{m}_local_p").max(axis=1)
            act[:, k] = np.where(avail[:, k], np.where(conf < BASELINE_CFG["confidence_gate_threshold"], 3, 1), 0)
        # Budget-aware: downgrade the largest raw/feature item until the window fits its share.
        for i in range(n):
            while sum(_one_bytes(act[i, k], k, raw[i, k]) for k in range(3)) > sc.budget_bytes:
                ks = [k for k in range(3) if act[i, k] >= 2]
                if not ks:
                    break
                kk = max(ks, key=lambda k: _one_bytes(act[i, k], k, raw[i, k]))
                act[i, kk] -= 1
    elif method == "random_within_budget":
        rng = np.random.default_rng(seed + 9001 + sum(map(ord, sc.name)))
        for i in range(n):
            tot = np.array([sum(_one_bytes(c[k], k, raw[i, k]) for k in range(3)) for c in COMBOS])
            ok = (tot <= sc.budget_bytes) & (COMBOS > 0).any(1)
            for k in range(3):
                ok &= (COMBOS[:, k] == 0) | avail[i, k]
            opts = np.where(ok)[0]
            if len(opts):
                act[i] = COMBOS[opts[int(rng.integers(len(opts)))]]
    elif method in ("proposed", "proposed_summary_floor"):
        util, byts, feas = proposed_options(v, sc, ablation, lam)
        if method == "proposed_summary_floor":
            # Exploratory (not pre-specified): never hold an eligible item; the optimizer only chooses upgrades.
            elig = np.stack([avail[:, k] & v[f"{m}_admissible"].to_numpy() for k, m in enumerate(MODALITIES)], axis=1)
            for k in range(3):
                feas &= (COMBOS[:, k] > 0)[None, :] | ~elig[:, k][:, None]
        for _, idx in v.groupby("slot_id").indices.items():
            idx = np.sort(idx)
            for g in controller_groups(idx, sc):
                act[g] = COMBOS[solve_slot(util[g], byts[g], feas[g], sc.budget_bytes * len(g))]
    else:
        raise ValueError(method)
    return act


# --------------------------------------------------------------------------- fusion and outcomes

def run_method(v: pd.DataFrame, method: str, sc: Scenario, seed: int, ablation: str | None = None,
               lam: float | None = None) -> pd.DataFrame:
    act = select_actions(v, method, sc, seed, ablation, lam)
    n = len(v)
    per_bytes = np.zeros((n, 3))
    for k, m in enumerate(MODALITIES):
        per_bytes[:, k] = np.take_along_axis(_bytes_table(v, m), act[:, k:k + 1], axis=1)[:, 0]
    bytes_sent = per_bytes.sum(axis=1)
    wait = np.where(act[:, 0] > 0, v.image_delay_actual_s.to_numpy(), 0.0)
    # Weighted logarithmic opinion pool (normalized weights). Waiting for an image ages the other evidence.
    logs = np.zeros((n, K()))
    tot_w = np.zeros(n)
    sel = []
    for k, m in enumerate(MODALITIES):
        P = np.stack([np.full((n, K()), 1 / K()), stack(v, f"{m}_local_p"), stack(v, f"{m}_feature_p"), stack(v, f"{m}_cloud_p")])
        p = P[act[:, k], np.arange(n)]
        age = v[f"{m}_age_s"].to_numpy() + (0.0 if m == "image" else wait)
        fresh = np.ones(n) if ablation == "no_recency" else np.exp(-age / RECENCY_TAU[m])
        q = np.ones(n) if ablation == "no_quality" else v[f"{m}_quality"].to_numpy()
        r = np.ones(n) if ablation == "no_reliability" else v[f"{m}_reliability"].to_numpy()
        wm = np.where(act[:, k] > 0, WEIGHTS[m] * q * r * fresh, 0.0)
        logs += wm[:, None] * np.log(np.clip(p, 1e-8, 1.0))
        tot_w += wm
        sel.append((act[:, k] > 0, p))
    fused = np.where(tot_w[:, None] > 0, softmax(logs / np.clip(tot_w, 1e-12, None)[:, None]), 1 / K())
    disagreement = np.zeros(n)
    for i in range(3):
        for j in range(i + 1, 3):
            ok = sel[i][0] & sel[j][0]
            d = np.zeros(n)
            d[ok] = js_rows(sel[i][1][ok], sel[j][1][ok])
            disagreement = np.maximum(disagreement, d)
    # Shared link inside each slot, served in order of estimated severity (same rule for all methods).
    queue = np.zeros(n)
    sev = v.severity.to_numpy()
    for _, idx in v.groupby("slot_id").indices.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(-sev[idx], kind="stable")]
        # Deterministic, slot-dependent offset into a measured trace so that
        # different slots see different parts of it; ignored when no trace is set.
        # Python's hash() is salted per process, so the offset is derived arithmetically.
        off = trace_offset_ms(sc.name, int(idx[0])) if sc.link_trace else 0
        queue[order] = serialization_time(np.cumsum(bytes_sent[order]), sc, off)
    latency = (queue + FIXED_OVERHEAD_S + sc.base_latency_s + v.link_interruption_s.to_numpy()
               + wait + retransmission_stall(bytes_sent, sc))
    slot_bytes = pd.Series(bytes_sent).groupby(v.slot_id.to_numpy()).transform("sum").to_numpy()
    slot_size = v.groupby("slot_id").slot_id.transform("size").to_numpy()
    res = pd.DataFrame({
        "window_id": v.window_id, "event_id": v.event_id, "slot_id": v.slot_id, "y": v.y,
        "severity_est": sev, "severity_true": v.severity_true, "method": method, "scenario": sc.name,
        "ablation": ablation or "", **{f"p{i}": fused[:, i] for i in range(K())},
        "prediction": fused.argmax(1), "confidence": fused.max(1), "bytes_sent": bytes_sent,
        "latency_s": latency, "waited_for_image_s": wait, "disagreement": disagreement,
        "slot_over_budget": slot_bytes > sc.budget_bytes * slot_size,
        "text_misaligned": v.text_misaligned.to_numpy(), "model_version": MODEL_VERSION,
        "decision_time_s": v.decision_time_s, "cell_x": v.cell_x, "cell_y": v.cell_y, "hazard": v.hazard,
    })
    for k, m in enumerate(MODALITIES):
        res[f"action_{m}"] = np.array(ACTIONS)[act[:, k]]
        res[f"bytes_{m}"] = per_bytes[:, k]
        for col in ("available", "admissible", "age_s", "quality", "reliability", "obs_time_s", "cell_x", "cell_y",
                    "hazard", "source_digest"):
            res[f"{m}_{col}"] = v[f"{m}_{col}"].to_numpy()
    return res


def lac_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-ambiguous set-valued classifier score (Sadinle et al., 2019): 1 - p(true class)."""
    return 1.0 - P[np.arange(len(y)), y]


def conformal_sets(P: np.ndarray, qhat: float) -> np.ndarray:
    """Membership (n, 3): every class whose probability is at least 1 - qhat.

    When qhat < 0.5 a set can hold at most one class, so a non-singleton set is
    an empty set: no class reaches the calibrated threshold (an abstention).
    """
    return P >= 1.0 - qhat - 1e-12


def conformal_qhat(res: pd.DataFrame, alpha: float) -> float:
    s = np.sort(lac_scores(res[PCOLS()].to_numpy(), res.y.to_numpy()))
    n = len(s)
    return float(s[min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n) - 1])


def annotate(res: pd.DataFrame, qhat: float, calibration_version: str, envelopes: bool = False,
             conflict_threshold: float | None = None) -> pd.DataFrame:
    out = res.copy()
    ct = POLICY["conflict_threshold"] if conflict_threshold is None else conflict_threshold
    member = conformal_sets(out[PCOLS()].to_numpy(), qhat)
    size = member.sum(1)
    n_sel = np.stack([(out[f"action_{m}"] != "hold").to_numpy() for m in MODALITIES], axis=1).sum(1)
    conflict = out.disagreement.to_numpy() >= ct
    uncertain = size != 1
    high = out.severity_est.to_numpy() >= POLICY["high_severity_threshold"]
    conf_ok = out.confidence.to_numpy() >= POLICY["confirmed_threshold"]
    status = np.select([high & (conflict | uncertain), uncertain, conflict, conf_ok & (n_sel >= 2)],
                       ["human_review_required", "insufficient_evidence", "conflicting", "confirmed"], "likely")
    rules = {"human_review_required": "R1:severity_est>=0.70 and (set_size!=1 or conflict)",
             "insufficient_evidence": "R2:set_size!=1", "conflicting": "R3:max_JS>=0.15",
             "confirmed": "R4:set_size==1 and confidence>=0.90 and modalities>=2", "likely": "R5:set_size==1 otherwise"}
    out["conformal_set"] = ["|".join(CLASS_NAMES[c] for c in range(K()) if row[c]) for row in member]
    out["set_size"] = size
    out["qhat"] = qhat
    out["decision_status"] = status
    out["status_rule"] = [rules[s] for s in status]
    out["review_required"] = np.isin(status, ["human_review_required", "insufficient_evidence", "conflicting"])
    out["policy_version"] = POLICY_VERSION
    out["calibration_version"] = calibration_version
    ok = np.ones(len(out), dtype=bool)
    for m in MODALITIES:
        chosen = (out[f"action_{m}"] != "hold").to_numpy()
        for c in ("source_digest", "obs_time_s", "cell_x", "cell_y", "hazard", "admissible"):
            ok &= ~chosen | out[f"{m}_{c}"].notna().to_numpy()
    for c in ("model_version", "policy_version", "calibration_version", "status_rule", "decision_time_s"):
        ok &= out[c].notna().to_numpy()
    out["provenance_complete"] = ok
    if envelopes:
        env = []
        for d in out.to_dict("records"):
            e = {
                "window_id": d["window_id"], "event_id": d["event_id"], "slot_id": d["slot_id"],
                "incident": {"cell": [int(d["cell_x"]), int(d["cell_y"])], "hazard": d["hazard"],
                             "decision_time_s": round(float(d["decision_time_s"]), 3)},
                "model_version": d["model_version"], "policy_version": d["policy_version"],
                "calibration_version": d["calibration_version"],
                "evidence": {m: {
                    "evidence_id": f"{d['window_id']}:{m}", "available": bool(d[f"{m}_available"]),
                    "admissible": bool(d[f"{m}_admissible"]), "observed_time_s": round(float(d[f"{m}_obs_time_s"]), 3),
                    "cell": [int(d[f"{m}_cell_x"]), int(d[f"{m}_cell_y"])], "hazard": d[f"{m}_hazard"],
                    "age_s": round(float(d[f"{m}_age_s"]), 3), "quality": round(float(d[f"{m}_quality"]), 6),
                    "reliability": round(float(d[f"{m}_reliability"]), 6), "action": d[f"action_{m}"],
                    "bytes": int(d[f"bytes_{m}"]), "source_digest_sha256": d[f"{m}_source_digest"],
                    "reason_code": ("not_available" if not d[f"{m}_available"] else "inadmissible" if not d[f"{m}_admissible"]
                                    else "held_by_optimizer" if d[f"action_{m}"] == "hold" else "selected_under_budget"),
                } for m in MODALITIES},
                "bytes_sent": int(d["bytes_sent"]), "modeled_latency_s": round(float(d["latency_s"]), 6),
                "probabilities": [round(float(d[c]), 8) for c in PCOLS()],
                "conformal_set": d["conformal_set"], "decision_status": d["decision_status"], "status_rule": d["status_rule"],
            }
            s = json.dumps(e, sort_keys=True, separators=(",", ":"))
            e["envelope_digest_sha256"] = hashlib.sha256(s.encode()).hexdigest()
            env.append(json.dumps(e, sort_keys=True, separators=(",", ":")))
        out["evidence_envelope"] = env
    return out


def ece(y, pred, conf, bins=10) -> float:
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


def aurc(y, pred, conf) -> float:
    # Stable sort: confidences tie often, and an unstable order makes the
    # risk-coverage curve depend on the sorting implementation.
    order = np.argsort(-conf, kind="stable")
    err = (pred[order] != y[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(np.trapezoid(risk, np.arange(1, len(err) + 1) / len(err)))


def summarize(r: pd.DataFrame, normal_qhat: float | None = None) -> dict:
    y, pred, conf = r.y.to_numpy(), r.prediction.to_numpy(), r.confidence.to_numpy()
    P = r[PCOLS()].to_numpy()
    review = r.review_required.to_numpy()
    err = pred != y
    member = conformal_sets(P, float(r.qhat.iloc[0]))
    high_true = r.severity_true.to_numpy() >= POLICY["high_severity_threshold"]
    out = {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "ece": ece(y, pred, conf), "aurc": aurc(y, pred, conf), "errors": int(err.sum()),
        "mean_bytes": float(r.bytes_sent.mean()), "p95_latency_s": float(r.latency_s.quantile(.95)),
        "median_latency_s": float(r.latency_s.median()), "review_rate": float(review.mean()),
        "accepted_error_rate": float((err & ~review).sum() / max((~review).sum(), 1)),
        "accepted_errors": int((err & ~review).sum()), "caught_errors": int((err & review).sum()),
        "reviews_per_caught_error": float(review.sum() / max((err & review).sum(), 1)),
        "high_severity_accepted_error_rate": float((err & ~review & high_true).sum() / max((~review & high_true).sum(), 1)),
        "conformal_coverage": float(member[np.arange(len(y)), y].mean()), "mean_set_size": float(member.sum(1).mean()),
        "singleton_rate": float((member.sum(1) == 1).mean()),
        "slot_budget_violation_rate": float(r.groupby("slot_id").slot_over_budget.first().mean()),
        "provenance_complete_rate": float(r.provenance_complete.mean()),
        "image_selected_rate": float((r.action_image != "hold").mean()),
    }
    if normal_qhat is not None:
        out["coverage_normal_calibrated"] = float(conformal_sets(P, normal_qhat)[np.arange(len(y)), y].mean())
    if r.text_misaligned.any():
        mis = r.text_misaligned.to_numpy()
        used = (r.action_text != "hold").to_numpy()
        out["misaligned_text_fused_rate"] = float(used[mis].mean())
        out["aligned_text_rejected_rate"] = float((~used & r.text_available.to_numpy())[~mis].mean())
    return out


# --------------------------------------------------------------------------- experiment

ABLATIONS = (
    ("no_severity", ("low_bandwidth", "normal")),
    ("no_novelty", ("contradictory_evidence", "noisy_text")),
    ("no_reliability", ("contradictory_evidence", "noisy_text")),
    ("no_quality", ("noisy_text", "contradictory_evidence")),
    ("no_recency", ("delayed_imagery",)),
    ("no_delay_penalty", ("delayed_imagery",)),
    ("no_admissibility", ("misaligned_evidence",)),
)


def _run(test_v, cal_v, method, sc, seed, models, ablation=None, lam=None, normal_q=None, envelopes=False):
    if ablation == "no_delay_penalty":
        lam = 0.0
    q = conformal_qhat(run_method(cal_v, method, sc, seed, ablation, lam), float(CONFIG["conformal_alpha"]))
    cal_version = hashlib.sha256(json.dumps({"q": round(q, 10), "temps": {f"{a}_{b}": t for (a, b), t in models["temps"].items()}},
                                            sort_keys=True).encode()).hexdigest()[:16]
    res = annotate(run_method(test_v, method, sc, seed, ablation, lam), q, cal_version, envelopes)
    return res, summarize(res, normal_q), q


def experiment(seed: int, log=print):
    train, cal, test = split_events(generate_events(seed))
    models = fit_models(train, cal, seed)
    cal, test = attach_probabilities(cal, models), attach_probabilities(test, models)
    views = {sc.name: (scenario_view(test, sc, seed, models), scenario_view(cal, sc, seed + 7, models)) for sc in SCENARIOS}
    # Derived wait threshold: one constant from the closed form, on calibration events only.
    dsc = SCENARIO_BY_NAME["delayed_imagery"]
    t_star = derive_wait_threshold(views["delayed_imagery"][1], dsc)
    for vv in views["delayed_imagery"]:
        vv["derived_wait_threshold_s"] = t_star
    summaries, records, envelopes, ablations, sweep, tsweep, review = [], [], [], [], [], [], []
    normal_q = {m: _run(*views["normal"], m, SCENARIO_BY_NAME["normal"], seed, models)[2] for m in METHODS}
    for sc in SCENARIOS:
        methods = METHODS + EXTRA_METHODS.get(sc.name, ())
        for method in methods:
            res, met, q = _run(*views[sc.name], method, sc, seed, models, normal_q=normal_q.get(method),
                               envelopes=(method == "proposed"))
            summaries.append({"seed": seed, "method": method, "scenario": sc.name, "qhat": q,
                              "budget_compliant_design": BUDGET_COMPLIANT[method], **met})
            if "evidence_envelope" in res:
                envelopes.extend(res.evidence_envelope.tolist())
                res = res.drop(columns=["evidence_envelope"])
            res["seed"] = seed
            records.append(res)
    for ab, scs in ABLATIONS:
        for sname in scs:
            _, met, _ = _run(*views[sname], "proposed", SCENARIO_BY_NAME[sname], seed, models, ablation=ab)
            ablations.append({"seed": seed, "ablation": ab, "scenario": sname, **met})
    # Sensitivity: do always-wait baselines lose accuracy only because evidence ages while they wait?
    for method in ("fixed_rule", "cloud_transmit_all", "wait_if_uncertain", "derived_threshold", "proposed"):
        _, met, _ = _run(*views["delayed_imagery"], method, dsc, seed, models, ablation="no_recency")
        ablations.append({"seed": seed, "ablation": f"sensitivity:{method}:no_recency", "scenario": "delayed_imagery", **met})
    for lam in CONFIG["latency_penalty_sweep"]:
        _, met, _ = _run(*views["delayed_imagery"], "proposed", dsc, seed, models, lam=lam)
        sweep.append({"seed": seed, "latency_penalty_per_second": lam, **met})
    # Sensitivity of the constant-threshold rule to its threshold (reported, not used for selection).
    for t in CONFIG["wait_threshold_sweep_s"]:
        tv, cv = (x.copy() for x in views["delayed_imagery"])
        tv["derived_wait_threshold_s"] = cv["derived_wait_threshold_s"] = float(t)
        for abl in (None, "no_recency"):
            _, met, _ = _run(tv, cv, "derived_threshold", dsc, seed, models, ablation=abl)
            tsweep.append({"seed": seed, "wait_threshold_s": float(t), "recency_decay": abl is None, **met})
    # Review-rule sensitivity: conformal error level and conflict threshold (reported, not re-tuned).
    grid = CONFIG["review_sensitivity"]
    for sc in SCENARIOS:
        test_v, cal_v = views[sc.name]
        cal_res, test_res = run_method(cal_v, "proposed", sc, seed), run_method(test_v, "proposed", sc, seed)
        for a in grid["alpha"]:
            qa = conformal_qhat(cal_res, a)
            for ct in grid["conflict_threshold"]:
                res = annotate(test_res, qa, "sensitivity", conflict_threshold=(np.inf if ct is None else ct))
                err = (res.prediction != res.y).to_numpy()
                rv = res.review_required.to_numpy()
                st = res.decision_status.to_numpy()
                review.append({"seed": seed, "scenario": sc.name, "alpha": a,
                               "conflict_threshold": "off" if ct is None else ct,
                               "review_rate": float(rv.mean()), "errors": int(err.sum()),
                               "caught_errors": int((err & rv).sum()), "accepted_errors": int((err & ~rv).sum()),
                               "queued_correct_share": float((~err & rv).sum() / max(rv.sum(), 1)),
                               "empty_set_rate": float((res.set_size == 0).mean()),
                               "conflicting_rate": float((st == "conflicting").mean()),
                               "conflicting_errors": int((err & (st == "conflicting")).sum())})
    info = {"train_events": sorted(train.event_id.unique()), "calibration_events": sorted(cal.event_id.unique()),
            "test_events": sorted(test.event_id.unique()),
            "temperatures": {f"{a}_{b}": t for (a, b), t in models["temps"].items()},
            "severity_estimator_train_r2": models["severity_train_r2"],
            "derived_wait_threshold_s": t_star}
    log(f"seed {seed} complete")
    return (pd.DataFrame(summaries), pd.concat(records, ignore_index=True), envelopes,
            pd.DataFrame(ablations), pd.DataFrame(sweep), pd.DataFrame(tsweep), pd.DataFrame(review), info)


TCRIT_4DF = 2.776  # two-sided 95% t critical value for 5 seeds


def _ci(g: pd.Series):
    mean = float(g.mean())
    half = float(TCRIT_4DF * g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
    return mean, mean - half, mean + half


METRICS = ["macro_f1", "errors", "ece", "aurc", "mean_bytes", "p95_latency_s", "median_latency_s", "review_rate",
           "accepted_error_rate", "accepted_errors", "caught_errors", "reviews_per_caught_error",
           "high_severity_accepted_error_rate", "conformal_coverage", "coverage_normal_calibrated", "mean_set_size",
           "singleton_rate", "slot_budget_violation_rate", "provenance_complete_rate", "image_selected_rate",
           "misaligned_text_fused_rate", "aligned_text_rejected_rate"]
BOUNDED = {"macro_f1", "ece", "aurc", "review_rate", "accepted_error_rate", "high_severity_accepted_error_rate",
           "conformal_coverage", "coverage_normal_calibrated", "singleton_rate", "slot_budget_violation_rate",
           "provenance_complete_rate", "image_selected_rate", "misaligned_text_fused_rate", "aligned_text_rejected_rate"}


def ci_table(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for m in METRICS:
            if m not in g or g[m].isna().all():
                continue
            mean, lo, hi = _ci(g[m].dropna())
            lo = max(0.0, lo)
            if m in BOUNDED:
                hi = min(1.0, hi)
            row.update({f"{m}_mean": mean, f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def run_all(output_dir: str | Path, seeds=None, log=print):
    seeds = tuple(CONFIG["seeds"] if seeds is None else seeds)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, R, E, A, W, T, V, info = [], [], [], [], [], [], [], {}
    for seed in seeds:
        s, r, e, a, w, t, v, i = experiment(seed, log)
        S.append(s); R.append(r); E.extend(e); A.append(a); W.append(w); T.append(t); V.append(v); info[str(seed)] = i
    summary, record = pd.concat(S, ignore_index=True), pd.concat(R, ignore_index=True)
    ablation, sweep = pd.concat(A, ignore_index=True), pd.concat(W, ignore_index=True)
    ci = ci_table(summary, ["method", "scenario"])
    full = summary[summary.method == "proposed"].assign(ablation="full")
    ab_ci = pd.concat([ci_table(full, ["ablation", "scenario"]), ci_table(ablation, ["ablation", "scenario"])], ignore_index=True)
    sweep_ci = ci_table(sweep, ["latency_penalty_per_second"])
    tsweep_ci = ci_table(pd.concat(T, ignore_index=True), ["recency_decay", "wait_threshold_s"])
    rv = pd.concat(V, ignore_index=True)
    review_ci = rv.groupby(["scenario", "alpha", "conflict_threshold"], sort=False) \
        .agg(review_rate=("review_rate", "mean"), errors=("errors", "mean"), caught_errors=("caught_errors", "mean"),
             accepted_errors=("accepted_errors", "mean"), queued_correct_share=("queued_correct_share", "mean"),
             empty_set_rate=("empty_set_rate", "mean"), conflicting_rate=("conflicting_rate", "mean"),
             conflicting_errors=("conflicting_errors", "mean")).reset_index()
    paired = []
    for sname in summary.scenario.unique():
        pr = summary[(summary.method == "proposed") & (summary.scenario == sname)].set_index("seed")
        for base in [m for m in summary[summary.scenario == sname].method.unique() if m != "proposed"]:
            cp = summary[(summary.method == base) & (summary.scenario == sname)].set_index("seed")
            for m in ("macro_f1", "errors", "mean_bytes", "p95_latency_s", "review_rate"):
                mean, lo, hi = _ci(pr.loc[cp.index, m] - cp[m])
                paired.append({"scenario": sname, "comparison": f"proposed_minus_{base}", "metric": m,
                               "mean_difference": mean, "ci_low": lo, "ci_high": hi, "n_seed_pairs": len(cp)})
    ab_paired = []
    for (ab, sname), g in ablation.groupby(["ablation", "scenario"]):
        if ab.startswith("sensitivity:"):
            continue
        pr = summary[(summary.method == "proposed") & (summary.scenario == sname)].set_index("seed")
        g = g.set_index("seed")
        for m in ("macro_f1", "errors", "mean_bytes", "p95_latency_s", "review_rate", "misaligned_text_fused_rate"):
            if m in g and g[m].notna().all():
                mean, lo, hi = _ci(g[m] - pr.loc[g.index, m])
                ab_paired.append({"ablation": ab, "scenario": sname, "metric": m, "ablated_minus_full": mean,
                                  "ci_low": lo, "ci_high": hi})
    record.groupby(["method", "scenario", "decision_status"], as_index=False).size() \
        .rename(columns={"size": "record_count"}).to_csv(out / "status_summary.csv", index=False)
    record[record.prediction != record.y].groupby(["method", "scenario", "event_id", "decision_status"], as_index=False) \
        .size().rename(columns={"size": "error_count"}).to_csv(out / "error_analysis.csv", index=False)
    summary.to_csv(out / "seed_level_results.csv", index=False)
    ci.to_csv(out / "confidence_intervals.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_differences.csv", index=False)
    ab_ci.to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(ab_paired).to_csv(out / "ablation_paired_differences.csv", index=False)
    sweep_ci.to_csv(out / "latency_penalty_sweep.csv", index=False)
    tsweep_ci.to_csv(out / "wait_threshold_sweep.csv", index=False)
    review_ci.to_csv(out / "review_rule_sensitivity.csv", index=False)
    # Record-level file keeps decision-relevant fields; full provenance for the proposed method is in the envelopes.
    drop = [c for c in record.columns if c.endswith(("_source_digest", "_cell_x", "_cell_y", "_hazard", "_obs_time_s"))]
    drop += ["cell_x", "cell_y", "hazard", "model_version", "policy_version", "calibration_version"]
    record.drop(columns=drop).to_csv(out / "record_level_results.csv.gz", index=False, float_format="%.6g",
                                     compression={"method": "gzip", "mtime": 0})
    with gzip.GzipFile(out / "evidence_envelopes_proposed.jsonl.gz", "wb", mtime=0) as fh:
        fh.write(("\n".join(E) + "\n").encode("utf-8"))
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}
    manifest = {
        "seeds": list(seeds), "synthetic_benchmark": True, "model_version": MODEL_VERSION, "policy_version": POLICY_VERSION,
        "config_sha256": sha256_file(CONFIG_PATH), "core_sha256": sha256_file(Path(__file__)), "splits": info,
        "config": CONFIG, "runtime": {"python": platform.python_version(), "platform": platform.platform(), **versions},
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, ci, ab_ci, sweep_ci
