# bandit_sim/data_processing.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ["action", "position", "reward"]  # pscore optional


def validate_bandit_df(df: pd.DataFrame, n_actions: int) -> None:
    """Validate input df: required columns, dtypes, and categorical ranges."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not np.issubdtype(df["action"].dtype, np.integer):
        raise TypeError("df['action'] must be integer dtype.")
    if not np.issubdtype(df["position"].dtype, np.integer):
        raise TypeError("df['position'] must be integer dtype.")
    if not np.issubdtype(df["reward"].dtype, np.integer):
        raise TypeError("df['reward'] must be integer dtype (0/1).")

    if df["action"].min() < 0 or df["action"].max() >= n_actions:
        raise ValueError("df['action'] must be in [0, n_actions-1].")

    if not set(df["position"].unique()).issubset({0, 1, 2}):
        raise ValueError("df['position'] must be in {0,1,2} (left, center, right).")

    if df["reward"].min() < 0 or df["reward"].max() > 1:
        raise ValueError("df['reward'] must be binary {0,1}.")


def sample_feedback_from_logs(
    df: pd.DataFrame,
    *,
    position: int,
    action: int,
    rng: np.random.Generator,
) -> Tuple[int, str, float]:
    """
    Uniformly sample (with replacement) ONE row for (position, action) to produce feedback.

    If no rows exist, return Bernoulli(0.5) as fallback.

    Returns
    -------
    reward : int
    source : str
        "logs" if sampled from df; "fallback" otherwise.
    pscore : float
        Optional logging propensity if df contains 'pscore'; else np.nan.
    """
    mask = (df["position"] == position) & (df["action"] == action)
    idx = np.flatnonzero(mask.values)
    if idx.size > 0:
        choice = int(rng.integers(0, idx.size))
        row = df.iloc[idx[choice]]
        reward = int(row["reward"])
        pscore = float(row["pscore"]) if "pscore" in df.columns else float("nan")
        return reward, "logs", pscore

    # No rows for (position, action): fallback Bernoulli(0.5)
    reward = int(rng.integers(0, 2))
    return reward, "fallback", float("nan")


def initialize_probability_distribution(
    n_actions: int,
    n_positions: int,
    distribution: str = "uniform",
) -> pd.DataFrame:
    """
    Create an explicit PMF table for each position and action.

    Returns
    -------
    pmf_df : pd.DataFrame with columns [position, action, prob]
    """
    if distribution.lower() != "uniform":
        raise NotImplementedError("Only 'uniform' initialization is supported.")
    pmf = np.full((n_positions, n_actions), 1.0 / n_actions, dtype=float)
    rows = []
    for pos in range(n_positions):
        for a in range(n_actions):
            rows.append({"position": pos, "action": a, "prob": pmf[pos, a]})
    return pd.DataFrame(rows, columns=["position", "action", "prob"])
