# bandit_sim/history.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class History:
    """
    Container for all run-time logs and snapshots.

    Stores: PMF snapshots, chosen actions, observed rewards, and run metadata.
    Exposes: tidy pandas.DataFrame views for analysis/plotting.

    Attributes
    ----------
    n_actions : int
        Number of actions (arms).
    n_positions : int
        Number of positions (independent bandits), e.g., 3.
    seed : int
        RNG seed used for the whole run (logged for exact reproducibility).
    pmf_interval : int
        Store PMF snapshots every N rounds (P2 requirement).
    pmf_mc_draws : int
        Monte Carlo draws per snapshot to approximate TS selection PMF.
    """
    n_actions: int
    n_positions: int
    seed: int
    pmf_interval: int
    pmf_mc_draws: int

    pmf_records: List[Dict] = field(default_factory=list)
    action_records: List[Dict] = field(default_factory=list)
    reward_records: List[Dict] = field(default_factory=list)
    run_metadata: Dict = field(default_factory=dict)

    def set_run_metadata(self, **kwargs) -> None:
        """Attach run-level metadata (e.g., dataset info, timestamp, rng state digest)."""
        self.run_metadata.update(kwargs)

    # -------- PMF snapshots --------
    def log_uniform_init(self) -> None:
        """Store the explicit uniform PMF at 'round=0' for every position."""
        probs = np.full(self.n_actions, 1.0 / self.n_actions, dtype=float)
        for pos in range(self.n_positions):
            self.log_pmf(round_idx=0, position=pos, phase="init", probs=probs)

    def log_pmf(self, *, round_idx: int, position: int, phase: str, probs: np.ndarray) -> None:
        """Append a long-form PMF snapshot for (round, position, phase)."""
        if probs.shape[0] != self.n_actions:
            raise ValueError("probs length must equal n_actions.")
        for a, p in enumerate(probs):
            self.pmf_records.append(
                {
                    "round": round_idx,
                    "position": int(position),
                    "phase": phase,  # "pre" | "post" | "init"
                    "action": int(a),
                    "prob": float(p),
                }
            )

    # -------- Actions & rewards --------
    def log_actions(self, *, round_idx: int, actions_by_pos: Dict[int, int]) -> None:
        """Append sampled actions for this round across positions."""
        for pos, a in actions_by_pos.items():
            self.action_records.append(
                {"round": round_idx, "position": int(pos), "action": int(a)}
            )

    def log_rewards(
        self,
        *,
        round_idx: int,
        rewards: List[Dict],
    ) -> None:
        """Append observed rewards with provenance."""
        for r in rewards:
            self.reward_records.append(
                {
                    "round": round_idx,
                    "position": int(r["position"]),
                    "action": int(r["action"]),
                    "reward": int(r["reward"]),
                    "source": r["source"],         # "logs" | "fallback"
                    "pscore": r.get("pscore", np.nan),
                }
            )

    # -------- DataFrame views --------
    def pmf_df(self) -> pd.DataFrame:
        cols = ["round", "position", "phase", "action", "prob"]
        return pd.DataFrame(self.pmf_records, columns=cols)

    def actions_df(self) -> pd.DataFrame:
        cols = ["round", "position", "action"]
        return pd.DataFrame(self.action_records, columns=cols)

    def rewards_df(self) -> pd.DataFrame:
        cols = ["round", "position", "action", "reward", "source", "pscore"]
        return pd.DataFrame(self.reward_records, columns=cols)
