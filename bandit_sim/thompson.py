# bandit_sim/thompson.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data_processing import (
    initialize_probability_distribution,
    sample_feedback_from_logs,
    validate_bandit_df,
)
from history import History


@dataclass
class ThompsonBandit:
    """One independent Bernoulli-TS bandit (for a single position)."""
    n_actions: int

    def __post_init__(self) -> None:
        self.alpha = np.ones(self.n_actions, dtype=float)  # Beta(1,1)
        self.beta = np.ones(self.n_actions, dtype=float)

    def sample_action(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        """Sample all thetas and return (argmax_action, thetas)."""
        thetas = rng.beta(self.alpha, self.beta)
        action = int(np.argmax(thetas))
        return action, thetas

    def update(self, action: int, reward: int) -> None:
        """Update the Beta posterior for the chosen action."""
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1.")
        if reward == 1:
            self.alpha[action] += 1.0
        else:
            self.beta[action] += 1.0

    def posterior_means(self) -> np.ndarray:
        """Convenience: return current E[theta_a] for all actions."""
        denom = self.alpha + self.beta
        return self.alpha / denom

    def ts_selection_pmf(
        self, rng: np.random.Generator, mc_draws: int
    ) -> np.ndarray:
        """
        Monte Carlo approx to P(a = argmax) under current posterior.

        Uses the single RNG to satisfy determinism requirement.
        """
        # Shape: (mc_draws, n_actions)
        a = np.tile(self.alpha, (mc_draws, 1))
        b = np.tile(self.beta, (mc_draws, 1))
        samples = rng.beta(a, b)
        winners = np.argmax(samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_actions).astype(float)
        return counts / counts.sum()


class TSMultiBanditSimulator:
    """
    Orchestrates 3 independent TS bandits (one per position) and maintains history.

    Parameters
    ----------
    df : pd.DataFrame
        Logged-feedback table with columns:
        - action (int in [0, n_actions-1])
        - position (int in {0,1,2} for left/center/right)
        - reward (int in {0,1})
        - pscore (float, optional; logged propensity)
    n_actions : int
    seed : int
        One global RNG seed; used for TS draws, row re-sampling, and PMF Monte Carlo.
    pmf_interval : int
        Snapshot cadence: store PMFs every N rounds (pre & post).
    pmf_mc_draws : int
        Monte Carlo draws per PMF snapshot.

    Notes
    -----
    - At round 0 we log the explicit uniform PMF (1/n_actions) for each position.
    - For each round r (1..N), we can store "pre" and "post" PMFs if r % pmf_interval == 0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        n_actions: int,
        n_positions: int = 3,
        seed: int = 0,
        pmf_interval: int = 1,
        pmf_mc_draws: int = 10_000,
    ) -> None:
        validate_bandit_df(df, n_actions=n_actions)
        self.df = df
        self.n_actions = n_actions
        self.n_positions = n_positions
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.bandits = [ThompsonBandit(n_actions) for _ in range(n_positions)]
        self.history = History(
            n_actions=n_actions,
            n_positions=n_positions,
            seed=self.seed,
            pmf_interval=int(pmf_interval),
            pmf_mc_draws=int(pmf_mc_draws),
        )
        # Log run metadata & initial uniform PMF (Beta(1,1) prior)
        self.history.set_run_metadata(
            seed=self.seed,
            rng_bit_generator=self.rng.bit_generator.__class__.__name__,
        )
        self.history.log_uniform_init()
        self._round = 0

    # ---------- Public API (round-level) ----------
    def run_round(self) -> None:
        """
        Execute a single round across all positions:
        - optional "pre" PMF snapshot
        - sample actions (TS)
        - sample rewards from logs (or fallback)
        - update posteriors
        - optional "post" PMF snapshot
        """
        r = self._round + 1  # rounds start at 1

        if r % self.history.pmf_interval == 0:
            self._snapshot_all_positions(round_idx=r, phase="pre")

        actions_by_pos = self._sample_actions()
        self.history.log_actions(round_idx=r, actions_by_pos=actions_by_pos)

        rewards = self._sample_rewards(actions_by_pos)
        self.history.log_rewards(round_idx=r, rewards=rewards)

        # Update posteriors
        for rw in rewards:
            pos, act, reward = rw["position"], rw["action"], rw["reward"]
            self.bandits[pos].update(act, reward)

        if r % self.history.pmf_interval == 0:
            self._snapshot_all_positions(round_idx=r, phase="post")

        self._round = r

    def run(self, n_rounds: int) -> None:
        """Run for N rounds."""
        for _ in range(n_rounds):
            self.run_round()

    # ---------- Snapshots ----------
    def _snapshot_all_positions(self, *, round_idx: int, phase: str) -> None:
        for pos in range(self.n_positions):
            pmf = self.bandits[pos].ts_selection_pmf(
                rng=self.rng, mc_draws=self.history.pmf_mc_draws
            )
            self.history.log_pmf(round_idx=round_idx, position=pos, phase=phase, probs=pmf)

    # ---------- Action & reward sampling ----------
    def _sample_actions(self) -> Dict[int, int]:
        """TS action per position. Returns {position: action}."""
        actions: Dict[int, int] = {}
        for pos, b in enumerate(self.bandits):
            action, _ = b.sample_action(self.rng)
            actions[pos] = action
        return actions

    def _sample_rewards(self, actions_by_pos: Dict[int, int]) -> List[Dict]:
        """Sample one feedback row per (position, chosen_action)."""
        events: List[Dict] = []
        for pos, act in actions_by_pos.items():
            reward, source, pscore = sample_feedback_from_logs(
                self.df, position=pos, action=act, rng=self.rng
            )
            events.append(
                {"position": pos, "action": act, "reward": reward, "source": source, "pscore": pscore}
            )
        return events

    # ---------- Quick checks ----------
    def after_first_round_is_non_uniform(self) -> bool:
        """
        Check (soft) that at least one position's 'post' PMF differs from uniform after round 1.
        """
        pmf = self.history.pmf_df()
        post_r1 = pmf[(pmf["round"] == 1) & (pmf["phase"] == "post")]
        if post_r1.empty:
            return False
        # Compare to uniform
        uni = 1.0 / self.n_actions
        diffs = np.abs(post_r1["prob"].to_numpy() - uni)
        return bool((diffs > 1e-12).any())
