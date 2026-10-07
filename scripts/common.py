"""Shared utilities for the ICCA revision reruns.

Protocol freezing, hashing, seeded RNG, paired statistics and CSV output.
Nothing here depends on the Edge repository.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROTOCOLS = os.path.join(ROOT, "protocols")
RESULTS = os.path.join(ROOT, "results")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(name: str) -> dict:
    """Load a frozen protocol and verify its recorded digest.

    The digest is stored in a sidecar `<name>.sha256`. On first use the
    sidecar is written; afterwards any edit to the protocol fails loudly.
    """
    path = os.path.join(PROTOCOLS, name)
    digest = sha256_file(path)
    side = path + ".sha256"
    if os.path.exists(side):
        recorded = open(side).read().strip()
        if recorded != digest:
            sys.exit(
                f"PROTOCOL MODIFIED: {name}\n recorded {recorded}\n current  {digest}\n"
                "Restore the file or create a new protocol with a new name."
            )
    else:
        with open(side, "w") as f:
            f.write(digest + "\n")
        print(f"[protocol] froze {name} sha256={digest}")
    with open(path) as f:
        proto = json.load(f)
    proto["_sha256"] = digest
    proto["_name"] = name
    return proto


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str] | None = None) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path} ({len(rows)} rows)")
    return path


def run_manifest(path: str, proto: dict, extra: dict | None = None) -> None:
    """Record provenance for a run next to its output."""
    m = {
        "protocol": proto.get("_name"),
        "protocol_sha256": proto.get("_sha256"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": sys.argv,
    }
    try:
        import sklearn  # noqa: F401
        m["sklearn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        m["sklearn"] = "not installed"
    if extra:
        m.update(extra)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[manifest] {path}")


# --------------------------------------------------------------------------
# Paired statistics used throughout the paper
# --------------------------------------------------------------------------

@dataclass
class Paired:
    mean: float
    lo: float
    hi: float
    t: float
    p: float
    n: int


def paired_t(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Paired:
    """Two-sided paired t interval and test for a - b over seeds."""
    from scipy import stats

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError("need at least two seeds")
    m = d.mean()
    s = d.std(ddof=1)
    se = s / np.sqrt(n)
    if se == 0:
        return Paired(m, m, m, float("inf") if m != 0 else 0.0, 0.0 if m != 0 else 1.0, n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    return Paired(m, m - tcrit * se, m + tcrit * se, t, p, n)


def mean_ci(x: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    from scipy import stats

    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    return m, m - tcrit * se, m + tcrit * se


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()
