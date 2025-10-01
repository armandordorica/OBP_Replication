import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_empirical_ctr_by_position(
    df: pd.DataFrame,
    positions: list[int] | None = None,
    n_actions: int | None = None,
    height_per_row: int = 300,
    title: str = "Empirical CTR per Action (split by Position)",
) -> go.Figure:
    """
    Make a multi-row bar chart where each row is a position and bars show
    empirical CTR (= mean reward) per action for that position.

    Parameters
    ----------
    df : DataFrame with columns ['action','position','reward', ...]
    positions : list[int] | None
        Which positions to plot. Defaults to all positions present in df (sorted).
    n_actions : int | None
        If given, ensures x-axis shows 0..n_actions-1 and fills missing actions with CTR=0.
        If None, uses only the actions present for each position.
    height_per_row : int
        Height per subplot row.
    title : str
        Figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # 1) compute mean reward per (position, action)
    ctr = (
        df.groupby(["position", "action"])["reward"]
          .mean()
          .reset_index(name="ctr")
    )

    # positions to plot
    if positions is None:
        positions = sorted(df["position"].unique().tolist())

    # 2) build subplots
    fig = make_subplots(
        rows=len(positions), cols=1, shared_xaxes=False,
        subplot_titles=[f"Position {p}" for p in positions],
        vertical_spacing=0.08
    )

    # helper for x-axis domain and filling
    def _prep_pos_slice(pos: int):
        sub = ctr[ctr["position"] == pos][["action", "ctr"]].copy()
        if n_actions is not None:
            # fill missing actions with 0 CTR to show full 0..n_actions-1
            all_actions = pd.DataFrame({"action": np.arange(n_actions)})
            sub = all_actions.merge(sub, on="action", how="left").fillna({"ctr": 0.0})
        sub.sort_values("action", inplace=True)
        return sub

    # 3) per-row bars + axes
    for i, pos in enumerate(positions, start=1):
        sub = _prep_pos_slice(pos)
        xs = sub["action"].to_numpy()
        ys = sub["ctr"].to_numpy()
        ymax = float(ys.max() * 1.25 if ys.size else 0.02)

        fig.add_bar(
            x=xs, y=ys,
            name=f"Pos {pos}",
            marker_color="steelblue",
            showlegend=False,
            row=i, col=1
        )
        fig.update_yaxes(title_text="Empirical CTR", range=[0, ymax], row=i, col=1)
        fig.update_xaxes(
            title_text=f"Action ID (0–{int(xs.max())})" if i == len(positions) else "",
            tickmode="array",
            tickvals=xs,
            ticktext=[str(a) for a in xs],
            row=i, col=1
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height_per_row * len(positions) + 120
    )
    return fig

def plot_ctr_grouped_by_action_position_discrete(
    df: pd.DataFrame,
    positions=(0, 1, 2),
    title="Empirical CTR per Action by Position (discrete, grouped)"
) -> go.Figure:
    # mean CTR per (position, action) and fill missing combos with 0
    actions_all = np.sort(df["action"].unique())
    ctr = (df.groupby(["position", "action"])["reward"]
             .mean()
             .reset_index(name="ctr"))
    full = (pd.MultiIndex.from_product([positions, actions_all],
            names=["position", "action"])
            .to_frame(index=False))
    ctr_full = (full.merge(ctr, on=["position", "action"], how="left")
                    .fillna({"ctr": 0.0}))

    # pivot to wide: rows = action, cols = position (ensures discrete series)
    wide = ctr_full.pivot(index="action", columns="position", values="ctr").reindex(actions_all).fillna(0.0)

    # build grouped bars (one trace per discrete position)
    fig = go.Figure()
    for pos in positions:
        y = wide[pos].to_numpy() if pos in wide.columns else np.zeros(len(wide))
        fig.add_bar(
            name=f"Position {pos}",
            x=wide.index.to_numpy(),
            y=y,
            offsetgroup=str(pos),  # guarantees side-by-side (not stacked/overlapped)
            legendgroup=str(pos),
        )

    fig.update_layout(
        title=title,
        barmode="group",          # grouped (not stacked)
        bargap=0.15,
        bargroupgap=0.05,
        template="plotly_white",
        xaxis=dict(
            title="Action ID",
            tickmode="array",
            tickvals=actions_all,
            ticktext=[str(a) for a in actions_all],
        ),
        yaxis=dict(title="Empirical CTR"),
        legend_title_text="Position",
        height=600
    )
    return fig


def plot_position_pmfs(pmfs, title="Probability mass functions per position", show=False):

    positions = list(pmfs.keys())
    n_actions = len(next(iter(pmfs.values())))
    fig, axes = plt.subplots(len(positions), 1, figsize=(12, 3*len(positions)), sharex=True)
    if len(positions) == 1:
        axes = [axes]

    for pos, ax in zip(positions, axes):
        ax.bar(np.arange(n_actions), pmfs[pos])
        ax.set_ylabel(f"P(action) @ pos {pos}")
        ax.set_ylim(0, pmfs[pos].max() * 1.2)

    axes[-1].set_xlabel(f"Action ID (0–{n_actions-1})")
    fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_empirical_means(
    df: pd.DataFrame,
    group_col: str = "action",
    value_col: str = "reward",
    title: str | None = None,
    height: int = 500,
):
    """
    Plot empirical mean (e.g. CTR) of a value_col grouped by group_col.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain group_col and value_col.
    group_col : str, default="action"
        Column to group by (x-axis).
    value_col : str, default="reward"
        Column to average (y-axis).
    title : str, optional
        Plot title (default auto-generated).
    height : int, default=500
        Plot height in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The bar chart figure.
    """
    # compute means
    means = df.groupby(group_col)[value_col].mean().reset_index()
    means.rename(columns={value_col: f"mean_{value_col}"}, inplace=True)

    # plot
    fig = px.bar(
        means,
        x=group_col,
        y=f"mean_{value_col}",
        labels={group_col: group_col.capitalize(), f"mean_{value_col}": f"Mean {value_col}"},
        title=title or f"Empirical mean of {value_col} by {group_col}",
    )

    # beautify x-axis ticks
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=means[group_col],
            ticktext=[str(v) for v in means[group_col]],
            title=group_col.capitalize(),
        ),
        yaxis=dict(title=f"Mean {value_col}"),
        template="plotly_white",
        height=height,
    )
    return fig



def plot_pmf_compare_by_position(
    pmf_round_A: dict[int, np.ndarray],
    pmf_round_B: dict[int, np.ndarray],
    positions: list[int],
    n_actions: int,
    title: str = "Selection probabilities by position",
    round_labels: tuple[str, str] = ("Round 0 (uniform)", "Round 1 (updated)"),
    colors: tuple[str, str] = ("steelblue", "darkorange"),
    height_per_row: int = 300,
) -> go.Figure:
    fig = make_subplots(
        rows=len(positions), cols=1, shared_xaxes=False,
        subplot_titles=[f"Position {p}" for p in positions],
        vertical_spacing=0.08
    )

    xs = np.arange(n_actions)
    tick_vals = xs
    tick_txt  = [str(a) for a in xs]

    for i, pos in enumerate(positions, start=1):
        yA = pmf_round_A[pos]
        yB = pmf_round_B[pos]
        ymax = float(max(yA.max(), yB.max())) * 1.25 if (yA.size and yB.size) else 0.02

        fig.add_trace(
            go.Bar(
                x=xs, y=yA, name=round_labels[0],
                marker=dict(color=colors[0]),
                opacity=0.6, legendgroup="roundA",
                showlegend=(i == 1),
                customdata=np.column_stack([np.full(n_actions, pos)]),
                hovertemplate=(
                    "Position: %{customdata[0]}<br>"
                    "Action: %{x}<br>"
                    f"{round_labels[0]}<br>P: %{{y:.6f}}<extra></extra>"
                ),
            ),
            row=i, col=1
        )
        fig.add_trace(
            go.Bar(
                x=xs, y=yB, name=round_labels[1],
                marker=dict(color=colors[1]),
                opacity=0.85, legendgroup="roundB",
                showlegend=(i == 1),
                customdata=np.column_stack([np.full(n_actions, pos)]),
                hovertemplate=(
                    "Position: %{customdata[0]}<br>"
                    "Action: %{x}<br>"
                    f"{round_labels[1]}<br>P: %{{y:.6f}}<extra></extra>"
                ),
            ),
            row=i, col=1
        )

        fig.update_yaxes(title_text="P(action selected)", range=[0, ymax], row=i, col=1)
        fig.update_xaxes(
            tickmode="array", tickvals=tick_vals, ticktext=tick_txt,
            showticklabels=True, ticks="outside",
            title_text=f"Action ID (0–{n_actions-1})" if i == len(positions) else "",
            row=i, col=1
        )

    fig.update_layout(
        title=title, barmode="group", bargap=0.15,
        template="plotly_white", height=height_per_row * len(positions) + 120,
    )
    return fig


# ---------- Plotting ----------
def plot_learned_vs_ctr(ctr_pos_act: pd.DataFrame, learned_p: dict[int, np.ndarray], title: str):
    positions = list(ctr_pos_act.index)
    K = ctr_pos_act.shape[1]

    fig = make_subplots(
        rows=len(positions), cols=1, shared_xaxes=False,
        subplot_titles=[f"Position {p}" for p in positions],
        vertical_spacing=0.08
    )

    for i, p in enumerate(positions, start=1):
        xs = np.arange(K)
        y_ctr = ctr_pos_act.loc[p].to_numpy()
        y_pol = learned_p[p]
        ymax = float(max(y_ctr.max(), y_pol.max())) * 1.25 if (y_ctr.max() or y_pol.max()) else 0.02

        fig.add_bar(x=xs, y=y_ctr, name="Empirical CTR",
                    marker_color="steelblue", opacity=0.6, showlegend=(i==1),
                    row=i, col=1)
        fig.add_bar(x=xs, y=y_pol, name="RL policy P(select)",
                    marker_color="darkorange", opacity=0.85, showlegend=(i==1),
                    row=i, col=1)

        fig.update_yaxes(title_text="Value", range=[0, ymax], row=i, col=1)
        fig.update_xaxes(
            title_text="Action ID (0–{})".format(K-1) if i==len(positions) else "",
            tickmode="array", tickvals=xs, ticktext=[str(a) for a in xs],
            row=i, col=1
        )

    fig.update_layout(
        title=title, barmode="group", bargap=0.15, template="plotly_white",
        height=300*len(positions) + 120
    )
    return fig
