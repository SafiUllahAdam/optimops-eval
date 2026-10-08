"""Cached access to the coverage simulator, plus the Pareto and noise tools (Acts 2 and 3).

The simulator is a black box with a budget of 2000 replays for the whole test, counted across
kernel restarts and logged in its own journal_simulateur.csv. This module never touches the
simulator's files or internals. It calls Simulateur.evaluer and keeps a copy of each answer in
results/simulation_runs.csv, so a notebook can be re-run top to bottom without spending again.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from helpers import ROOT, RESULTS

SIM_DIR = ROOT / "docuu" / "adam" / "simulateur"
RUNS = RESULTS / "simulation_runs.csv"
BUDGET_TOTAL = 2000
MAX_VEHICLES = 12
TYPES = ("VSAV", "FPT")

_sim = None


def _simulator():
    """Build the simulator only when a replay is really needed."""
    global _sim
    if _sim is None:
        sys.path.insert(0, str(SIM_DIR))
        from simulateur import Simulateur

        _sim = Simulateur()
    return _sim


def budget_left() -> int:
    """Replays left, read (never written) from the simulator's own counter file."""
    counter = SIM_DIR / ".budget.json"
    used = json.loads(counter.read_text())["consomme"] if counter.exists() else 0
    return BUDGET_TOTAL - int(used)


def canonical(config: dict) -> str:
    """One text form per reinforcement: centres and types sorted, zero entries dropped."""
    clean = {
        cis: {t: int(k) for t, k in sorted(per_type.items()) if int(k) > 0}
        for cis, per_type in sorted(config.items())
    }
    return json.dumps({cis: v for cis, v in clean.items() if v}, sort_keys=True)


def vehicles(config: dict) -> int:
    return sum(int(k) for per_type in config.values() for k in per_type.values())


def load_runs() -> pd.DataFrame:
    if RUNS.exists():
        return pd.read_csv(RUNS, dtype={"config": str})
    return pd.DataFrame(columns=["tag", "config", "n", "vehicles", "p90", "p90_sd", "inc", "inc_sd", "when"])


def evaluate(config: dict, n: int, tag: str, reserve: int = 0) -> dict:
    """Mean and spread of the two simulated objectives over n replays, cached by (tag, config, n).

    A cached answer is returned as is and costs nothing. A new call refuses to run if it would
    leave fewer than `reserve` replays for later work.
    """
    key = canonical(config)
    runs = load_runs()
    hit = runs[(runs["tag"] == tag) & (runs["config"] == key) & (runs["n"] == n)]
    if len(hit):
        return hit.iloc[0].to_dict()
    if vehicles(config) > MAX_VEHICLES:
        raise ValueError(f"{vehicles(config)} vehicles, the service funds at most {MAX_VEHICLES}")
    if budget_left() - n < reserve:
        raise RuntimeError(f"{n} replays would break the reserve: {budget_left()} left, {reserve} kept back")

    r = _simulator().evaluer(json.loads(key), n_replicats=n)
    row = {
        "tag": tag, "config": key, "n": n, "vehicles": r["engins_ajoutes"],
        "p90": r["p90_delai_sap_min"], "p90_sd": r["ecart_type_p90"],
        "inc": r["inc_non_honorees"], "inc_sd": r["ecart_type_inc"],
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
    return row


def pool(runs: list[dict]) -> dict:
    """Merge several calls of one configuration. The spread includes the gap between the calls."""
    n = np.array([r["n"] for r in runs], dtype=float)
    out = {"tag": "pooled", "config": runs[0]["config"], "n": int(n.sum()), "vehicles": runs[0]["vehicles"]}
    for k in ("p90", "inc"):
        m = np.array([r[k] for r in runs])
        s = np.array([r[f"{k}_sd"] for r in runs])
        mean = (n * m).sum() / n.sum()
        squares = ((n - 1) * s**2).sum() + (n * (m - mean) ** 2).sum()
        out[k], out[f"{k}_sd"] = mean, float(np.sqrt(squares / (n.sum() - 1)))
    return out


def half_width(sd, n, level: float = 0.95):
    """Half-width of the confidence interval on a mean of n replays (Student t)."""
    n = np.asarray(n, dtype=float)
    return stats.t.ppf(0.5 + level / 2, n - 1) * np.asarray(sd, dtype=float) / np.sqrt(n)


def difference(a: dict, b: dict, objective: str) -> tuple[float, float]:
    """a minus b on one objective, with the Welch 95% half-width of that difference."""
    va, vb = a[f"{objective}_sd"] ** 2 / a["n"], b[f"{objective}_sd"] ** 2 / b["n"]
    dof = (va + vb) ** 2 / (va**2 / (a["n"] - 1) + vb**2 / (b["n"] - 1))
    return a[objective] - b[objective], stats.t.ppf(0.975, dof) * np.sqrt(va + vb)


def pareto_mask(objectives) -> np.ndarray:
    """True for the rows no other row dominates. Every column is minimised."""
    obj = np.asarray(objectives, dtype=float)
    dominated = np.array([((obj <= row).all(axis=1) & (obj < row).any(axis=1)).any() for row in obj])
    return ~dominated
