from typing import Iterable, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ThompsonSlateTracker:
    def __init__(
        self,
        df: pd.DataFrame,
        n_actions: int,
        positions: Iterable[int] = (0, 1, 2),
        seed: int = 7,
        factorize_by_position: bool = True,  # False => vanilla TS
    ):
        """
        df: DataFrame with columns ['action','position','reward','pscore']
        n_actions: number of actions (e.g., 80)
        positions: iterable of slot indices (only used if factorize_by_position=True)
        factorize_by_position: if True -> one bandit per position; else -> single vanilla bandit
        """
        self.df = df
        self.n_actions = int(n_actions)
        self.factorize = bool(factorize_by_position)
        self.rng = np.random.RandomState(seed)

        if self.factorize:
            # independent bandit per position
            self.positions = tuple(positions)
            self.bandit_keys = list(self.positions)  # rows to plot
        else:
            # single bandit that ignores position
            self.positions = tuple(positions)  # kept for compatibility, not used in updates
            self.bandit_keys = ["all"]  # one row to plot

        # priors α,β per bandit_key
        self.alpha = {k: np.ones(self.n_actions, dtype=float) for k in self.bandit_keys}
        self.beta = {k: np.ones(self.n_actions, dtype=float) for k in self.bandit_keys}

        # history
        self.round = 0
        self.pmfs: Dict[int, Dict[Any, np.ndarray]] = {}     # round -> {bandit_key -> np.array(n_actions)}
        self.samples: Dict[int, pd.DataFrame] = {}           # round -> DataFrame of sampled rows

        # round 0 PMF snapshot (uniform)
        pmf0 = {k: np.full(self.n_actions, 1.0 / self.n_actions) for k in self.bandit_keys}
        self.pmfs[0] = pmf0
        self.samples[0] = pd.DataFrame(columns=["action", "position", "reward", "pscore"])

    # ---------- PMF backends ----------
    @staticmethod
    def _pmf_exact_one_update(K: int, updated_arm: int, reward: int):
        """Exact closed-form PMF when starting from Beta(1,1) and exactly one arm updated once."""
        p = np.empty(K, dtype=float)
        if reward == 1:
            p_updated = 2.0 / (K + 1.0)
            p_others = 1.0 / (K + 1.0)
        else:
            p_updated = 2.0 / (K * (K + 1.0))
            p_others = (1.0 - p_updated) / (K - 1.0)
        p.fill(p_others)
        p[updated_arm] = p_updated
        return p

    @staticmethod
    def _pmf_mc(alpha_vec, beta_vec, n_sim=200_000, rng=None):
        """Monte Carlo PMF: P(action = argmax)."""
        rng = np.random.RandomState() if rng is None else rng
        K = len(alpha_vec)
        samples = rng.beta(alpha_vec[None, :], beta_vec[None, :], size=(n_sim, K))
        winners = samples.argmax(axis=1)
        counts = np.bincount(winners, minlength=K)
        return counts / counts.sum()

    def _can_use_exact(self, key) -> Tuple[bool, Optional[int], Optional[int]]:
        """Exact valid iff all (α,β)=(1,1) except possibly one arm with (2,1) or (1,2)."""
        a = self.alpha[key]
        b = self.beta[key]
        mask_11 = (a == 1) & (b == 1)
        if mask_11.all():
            return True, None, None
        idx = np.where(~mask_11)[0]
        if len(idx) != 1:
            return False, None, None
        j = int(idx[0])
        pair = (a[j], b[j])
        if pair == (2, 1):
            return True, j, 1
        if pair == (1, 2):
            return True, j, 0
        return False, None, None

    # ---------- Sampling / updates ----------
    def _sample_batch(self, random_state=None) -> pd.DataFrame:
        """
        Returns sampled rows used for this round:
          - factorized: one row per position
          - vanilla:    a single row from the entire df
        """
        if self.factorize:
            return (
                self.df.groupby("position", group_keys=False)
                .apply(lambda x: x.sample(1, random_state=random_state))
                .reset_index(drop=True)
            )
        else:
            return self.df.sample(1, random_state=random_state).reset_index(drop=True)

    def apply_updates(self, sampled_rows: pd.DataFrame):
        """Update α/β from sampled rows."""
        for _, r in sampled_rows.iterrows():
            a = int(r["action"])
            y = int(r["reward"])
            if self.factorize:
                key = int(r["position"])
            else:
                key = "all"  # ignore position
            self.alpha[key][a] += y
            self.beta[key][a] += (1 - y)

    def snapshot_pmfs(self, use_mc_if_needed=True, n_sim=200_000):
        """Compute PMF for each bandit_key; store under current round."""
        pmf = {}
        for key in self.bandit_keys:
            ok_exact, idx, rew = self._can_use_exact(key)
            if ok_exact:
                if idx is None:
                    pmf[key] = np.full(self.n_actions, 1.0 / self.n_actions, dtype=float)
                else:
                    pmf[key] = self._pmf_exact_one_update(self.n_actions, idx, rew)
            else:
                if not use_mc_if_needed:
                    raise RuntimeError(
                        f"Exact closed-form not valid for '{key}' after round {self.round}. "
                        "Set use_mc_if_needed=True to compute PMFs."
                    )
                pmf[key] = self._pmf_mc(self.alpha[key], self.beta[key], n_sim=n_sim, rng=self.rng)
        self.pmfs[self.round] = pmf

    def next_round(self, random_state=None, use_mc_if_needed=True, n_sim=200_000) -> pd.DataFrame:
        """Advance one round: sample, update, snapshot."""
        self.round += 1
        samp = self._sample_batch(random_state=random_state)
        self.samples[self.round] = samp
        self.apply_updates(samp)
        self.snapshot_pmfs(use_mc_if_needed=use_mc_if_needed, n_sim=n_sim)
        return samp

    # ---------- Plot ----------
    def _round_picks_by_key(self, round_id: int):
        """
        Return a dict: key -> {'action': int, 'reward': int} for that round's sampled row(s).
        - factorized: one per position key
        - vanilla:    one under key 'all'
        If round_id == 0 (no samples), returns {}.
        """
        if round_id not in self.samples or self.samples[round_id].empty:
            return {}
        picks = {}
        df_s = self.samples[round_id]
        if self.factorize:
            for _, r in df_s.iterrows():
                pos = int(r["position"])
                picks[pos] = {"action": int(r["action"]), "reward": int(r["reward"])}
        else:
            r = df_s.iloc[0]
            picks["all"] = {"action": int(r["action"]), "reward": int(r["reward"])}
        return picks

    def plot_compare(
        self,
        round_a: int,
        round_b: int,
        height=900,
        title=None,
        show_picks: bool = True,
        pick_marker_a=dict(color="crimson", symbol="triangle-up", size=12),
        pick_marker_b=dict(color="forestgreen", symbol="diamond", size=11),
    ):
        """Grouped bars comparing PMFs of two rounds; optionally highlight sampled picks."""
        if round_a not in self.pmfs or round_b not in self.pmfs:
            raise KeyError("Requested rounds not in history. Call next_round() first.")

        pmf_a = self.pmfs[round_a]
        pmf_b = self.pmfs[round_b]
        n_actions = self.n_actions
        row_keys = self.bandit_keys  # ['all'] or [positions...]

        # gather picks (if any) for each round
        picks_a = self._round_picks_by_key(round_a)
        picks_b = self._round_picks_by_key(round_b)

        fig = make_subplots(
            rows=len(row_keys),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Position {k}" if k != "all" else "Vanilla TS" for k in row_keys],
            vertical_spacing=0.08,
        )

        xs = np.arange(n_actions)
        labels = [str(a) for a in xs]

        for i, key in enumerate(row_keys, start=1):
            yA = pmf_a[key]
            yB = pmf_b[key]
            ymax = float(max(yA.max(), yB.max())) * 1.25

            # Bars for the two rounds
            fig.add_bar(
                x=xs,
                y=yA,
                name=f"Round {round_a}",
                marker_color="steelblue",
                opacity=0.6,
                showlegend=(i == 1),
                row=i,
                col=1,
            )
            fig.add_bar(
                x=xs,
                y=yB,
                name=f"Round {round_b}",
                marker_color="darkorange",
                opacity=0.85,
                showlegend=(i == 1),
                row=i,
                col=1,
            )

            # Optional markers for the sampled pick(s) that updated the round
            if show_picks:
                # Round A pick on its PMF
                if key in picks_a:
                    a_pick = picks_a[key]["action"]
                    r_pick = picks_a[key]["reward"]
                    fig.add_trace(
                        go.Scatter(
                            x=[a_pick],
                            y=[yA[a_pick]],
                            mode="markers",
                            marker=pick_marker_a,
                            name=f"Picked (r={round_a})",
                            showlegend=(i == 1),
                            hovertemplate=(
                                f"Round {round_a}<br>"
                                f"{'Position' if key!='all' else 'Bandit'}: {key}<br>"
                                "Action: %{x}<br>"
                                "P: %{y:.6f}<br>"
                                f"Reward: {r_pick}<extra></extra>"
                            ),
                        ),
                        row=i,
                        col=1,
                    )

                # Round B pick on its PMF
                if key in picks_b:
                    b_pick = picks_b[key]["action"]
                    r_pick = picks_b[key]["reward"]
                    fig.add_trace(
                        go.Scatter(
                            x=[b_pick],
                            y=[yB[b_pick]],
                            mode="markers",
                            marker=pick_marker_b,
                            name=f"Picked (r={round_b})",
                            showlegend=(i == 1),
                            hovertemplate=(
                                f"Round {round_b}<br>"
                                f"{'Position' if key!='all' else 'Bandit'}: {key}<br>"
                                "Action: %{x}<br>"
                                "P: %{y:.6f}<br>"
                                f"Reward: {r_pick}<extra></extra>"
                            ),
                        ),
                        row=i,
                        col=1,
                    )

            # Axes
            fig.update_yaxes(title_text="P(select)", range=[0, ymax], row=i, col=1)
            fig.update_xaxes(
                tickmode="array",
                tickvals=xs,
                ticktext=labels,
                title_text=f"Action ID (0–{n_actions-1})" if i == len(row_keys) else "",
                row=i,
                col=1,
            )

        ttl = title or (
            f"{'Position-aware' if self.factorize else 'Vanilla'} Thompson Sampling: "
            f"Round {round_a} vs Round {round_b}"
        )
        fig.update_layout(
            title=ttl,
            barmode="group",
            bargap=0.15,
            template="plotly_white",
            height=max(350 * len(row_keys), 450),
        )
        fig.show()

    # ---------- small conveniences ----------
    def get_pmf(self, round_id: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """Return PMFs for a given round (default: current)."""
        rid = self.round if round_id is None else round_id
        if rid not in self.pmfs:
            raise KeyError(f"No PMF stored for round {rid}.")
        return self.pmfs[rid]

    def rollout(self, n_rounds: int, **kwargs) -> None:
        """Run multiple rounds back-to-back (stores samples/pmfs as usual)."""
        for _ in range(int(n_rounds)):
            self.next_round(**kwargs)

    def reset(self):
        """Reset posteriors and history to round 0 (uniform)."""
        for k in self.bandit_keys:
            self.alpha[k].fill(1.0)
            self.beta[k].fill(1.0)
        self.round = 0
        self.pmfs = {0: {k: np.full(self.n_actions, 1.0 / self.n_actions) for k in self.bandit_keys}}
        self.samples = {0: pd.DataFrame(columns=["action", "position", "reward", "pscore"])}

