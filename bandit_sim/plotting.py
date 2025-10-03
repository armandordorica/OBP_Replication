# bandit_sim/plotting.py
from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objects as go

def plot_pmf_snapshot(
    pmf_df: pd.DataFrame,
    *,
    position: int,
    round_idx: int,
    phase: str = "post",
    title: str | None = None,
) -> None:
    """
    Bar plot of the TS selection PMF for (position, round, phase).
    """
    snap = pmf_df[
        (pmf_df["position"] == position)
        & (pmf_df["round"] == round_idx)
        & (pmf_df["phase"] == phase)
    ]
    if snap.empty:
        raise ValueError("No snapshot found for the specified filters.")
    fig = plt.figure()
    x = snap["action"].to_numpy()
    y = snap["prob"].to_numpy()
    plt.bar(x, y)
    plt.xlabel("Action")
    plt.ylabel("P_TS(select action)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()



def compare_pmf_across_rounds(
    pmf_df: pd.DataFrame,
    *,
    position: int,
    rounds: Sequence[int],
    phase: str = "post",
) -> None:
    """
    Overlay PMFs as grouped bar plots for a fixed position across multiple rounds using Plotly.
    Hovering shows round, action, and probability.
    """
    fig = go.Figure()

    for r in rounds:
        snap = pmf_df[
            (pmf_df["position"] == position)
            & (pmf_df["round"] == r)
            & (pmf_df["phase"] == phase)
        ].sort_values("action")
        if snap.empty:
            continue

        fig.add_trace(
            go.Bar(
                x=snap["action"],
                y=snap["prob"],
                name=f"Round {r}",
                hovertemplate="Round: %{name}<br>Action: %{x}<br>Prob: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",  # set to 'overlay' if you prefer stacked overlay
        title=f"TS Selection PMF for Position {position} ({phase})",
        xaxis_title="Action",
        yaxis_title="P_TS(select action)",
        legend_title="Round",
    )

    fig.show()

