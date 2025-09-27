#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Offline simulation: probability of each action over 5 rounds (single slot) ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obp.dataset import OpenBanditDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SEED = 7
rng = np.random.RandomState(SEED)


# ### Helper Functions

# In[2]:


def exact_pmf_one_update(K: int, updated_arm: int, reward: int):
    """
    Closed-form TS selection probabilities when all arms start Beta(1,1),
    and exactly one arm is updated once at this position (no MC).
    """
    p = np.empty(K, dtype=float)
    if reward == 1:
        p_updated = 2.0 / (K + 1.0)              # updated arm
        p_others  = 1.0 / (K + 1.0)              # every other arm
    else:
        p_updated = 2.0 / (K * (K + 1.0))        # updated arm
        p_others  = (1.0 - p_updated) / (K - 1.) # spread remainder evenly
    p.fill(p_others)
    p[updated_arm] = p_updated
    return p

def make_probs_df(round_num: int, alpha, beta, pmf_by_pos):
    """Build a wide DataFrame for a given round with selection probs and α/β per position."""
    rows = []
    for a in range(n_actions):
        row = {"round": round_num, "action": a}
        for pos in positions:
            row[f"p_pos{pos}"]     = pmf_by_pos[pos][a]
            row[f"alpha_pos{pos}"] = alpha[pos][a]
            row[f"beta_pos{pos}"]  = beta[pos][a]
        rows.append(row)
    return pd.DataFrame(rows)

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



class ThompsonSlateTracker:
    def __init__(self, df, n_actions: int, positions=(0,1,2), seed: int = 7):
        """
        df: DataFrame with columns ['action','position','reward','pscore']
        n_actions: number of actions (e.g., 80)
        positions: iterable of slot indices present in df (0,1,2)
        """
        self.df = df
        self.n_actions = int(n_actions)
        self.positions = tuple(positions)
        self.rng = np.random.RandomState(seed)

        # priors
        self.alpha = {p: np.ones(self.n_actions, dtype=float) for p in self.positions}
        self.beta  = {p: np.ones(self.n_actions, dtype=float) for p in self.positions}

        # history
        self.round = 0
        self.pmfs = {}        # round -> {pos -> np.array(n_actions)}
        self.samples = {}     # round -> DataFrame rows (one per position)

        # round 0 snapshot (uniform PMF)
        pmf0 = {p: np.full(self.n_actions, 1.0/self.n_actions) for p in self.positions}
        self.pmfs[0] = pmf0
        self.samples[0] = pd.DataFrame(columns=["action","position","reward","pscore"])

    # ---------- PMF backends ----------
    @staticmethod
    def _pmf_exact_one_update(K: int, updated_arm: int, reward: int):
        """Exact closed-form PMF when starting from Beta(1,1) and exactly one arm updated once."""
        p = np.empty(K, dtype=float)
        if reward == 1:
            p_updated = 2.0 / (K + 1.0)
            p_others  = 1.0 / (K + 1.0)
        else:
            p_updated = 2.0 / (K * (K + 1.0))
            p_others  = (1.0 - p_updated) / (K - 1.0)
        p.fill(p_others)
        p[updated_arm] = p_updated
        return p

    @staticmethod
    def _pmf_mc(alpha_vec, beta_vec, n_sim=200_000, rng=None):
        """Monte Carlo PMF: P(action = argmax) for one position."""
        rng = np.random.RandomState() if rng is None else rng
        K = len(alpha_vec)
        samples = rng.beta(alpha_vec[None, :], beta_vec[None, :], size=(n_sim, K))
        winners = samples.argmax(axis=1)
        counts = np.bincount(winners, minlength=K)
        return counts / counts.sum()

    def _can_use_exact(self, pos: int, samples_this_round: pd.DataFrame):
        """Exact is valid iff all actions at this pos are Beta(1,1) except possibly ONE with a single +/-1 update."""
        # Check α,β before applying new updates? We call this AFTER updates for the round,
        # but only exact if the *cumulative* state at this pos is either:
        # - all ones (no updates), or
        # - exactly one action has (alpha,beta) in {(2,1),(1,2)} and everyone else is (1,1)
        a = self.alpha[pos]; b = self.beta[pos]
        mask_ones = (a == 1) & (b == 1)
        if mask_ones.all():
            return True, None, None  # uniform exact
        # find candidates with one step changes
        changed_idx = np.where(~mask_ones)[0]
        if len(changed_idx) != 1:
            return False, None, None
        j = int(changed_idx[0])
        pair = (a[j], b[j])
        if (pair == (2,1)) or (pair == (1,2)):
            # also ensure all others are (1,1)
            return True, j, int(pair[0] == 2 and pair[1] == 1)  # reward=1 if (2,1)
        return False, None, None

    # ---------- Round operations ----------
    def sample_one_per_position(self, random_state=None):
        """Return 3-row DataFrame with one random row per position."""
        samp = (
            self.df.groupby("position", group_keys=False)
            .apply(lambda x: x.sample(1, random_state=random_state))
            .reset_index(drop=True)
        )
        return samp

    def apply_updates(self, sampled_rows: pd.DataFrame):
        """Update alpha/beta using the sampled rows."""
        for _, r in sampled_rows.iterrows():
            a_obs = int(r["action"])
            p_obs = int(r["position"])
            r_obs = int(r["reward"])
            self.alpha[p_obs][a_obs] += r_obs
            self.beta[p_obs][a_obs]  += (1 - r_obs)

    def snapshot_pmfs(self, use_mc_if_needed=True, n_sim=200_000):
        """Compute PMF for every position given current alpha/beta; store under current round."""
        pmf = {}
        for pos in self.positions:
            ok_exact, idx, rew = self._can_use_exact(pos, None)
            if ok_exact:
                if idx is None:
                    # truly uniform
                    pmf[pos] = np.full(self.n_actions, 1.0/self.n_actions, dtype=float)
                else:
                    pmf[pos] = self._pmf_exact_one_update(self.n_actions, idx, rew)
            else:
                if not use_mc_if_needed:
                    raise RuntimeError(
                        f"Exact closed-form is not valid at position {pos} after round {self.round}. "
                        "Set use_mc_if_needed=True (default) to compute PMFs."
                    )
                pmf[pos] = self._pmf_mc(self.alpha[pos], self.beta[pos], n_sim=n_sim, rng=self.rng)
        self.pmfs[self.round] = pmf

    def next_round(self, random_state=None, use_mc_if_needed=True, n_sim=200_000):
        """Advance one round: sample, update, snapshot PMFs."""
        self.round += 1
        samp = self.sample_one_per_position(random_state=random_state)
        self.samples[self.round] = samp
        self.apply_updates(samp)
        self.snapshot_pmfs(use_mc_if_needed=use_mc_if_needed, n_sim=n_sim)
        return samp  # useful to print/inspect

    # ---------- Plot ----------
    def plot_compare(self, round_a: int, round_b: int, height=900, title=None):
        """Plotly grouped bars comparing PMFs of two rounds for each position."""
        if round_a not in self.pmfs or round_b not in self.pmfs:
            raise KeyError("Requested rounds not in history. Call snapshot/next_round first.")

        pmf_a = self.pmfs[round_a]
        pmf_b = self.pmfs[round_b]
        n_actions = self.n_actions

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=False,
            subplot_titles=[f"Position {p}" for p in self.positions],
            vertical_spacing=0.08
        )

        for i, pos in enumerate(self.positions, start=1):
            xs = np.arange(n_actions)
            yA = pmf_a[pos]
            yB = pmf_b[pos]
            ymax = float(max(yA.max(), yB.max())) * 1.25

            fig.add_trace(
                go.Bar(
                    x=xs, y=yA, name=f"Round {round_a}",
                    marker=dict(color="steelblue"), opacity=0.6,
                    showlegend=(i == 1),
                    customdata=np.column_stack([np.full(n_actions, pos)]),
                    hovertemplate="Position: %{customdata[0]}<br>Action: %{x}<br>Round: "+str(round_a)+"<br>P: %{y:.6f}<extra></extra>",
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=xs, y=yB, name=f"Round {round_b}",
                    marker=dict(color="darkorange"), opacity=0.8,
                    showlegend=(i == 1),
                    customdata=np.column_stack([np.full(n_actions, pos)]),
                    hovertemplate="Position: %{customdata[0]}<br>Action: %{x}<br>Round: "+str(round_b)+"<br>P: %{y:.6f}<extra></extra>",
                ),
                row=i, col=1
            )
            fig.update_yaxes(range=[0, ymax], row=i, col=1)

        # force ticks 0..n_actions-1 on all three subplots
        ticks = np.arange(n_actions)
        labels = [str(a) for a in ticks]
        for i in range(1, 4):
            fig.update_xaxes(
                tickmode="array", tickvals=ticks, ticktext=labels,
                showticklabels=True, ticks="outside",
                row=i, col=1
            )

        fig.update_xaxes(title_text="Action ID (0–79)", row=3, col=1)
        for i in range(1, 4):
            fig.update_yaxes(title_text="P(action selected)", row=i, col=1)

        fig.update_layout(
            title=title or f"Selection probabilities by position: Round {round_a} vs Round {round_b}",
            barmode="group", bargap=0.15, height=height, template="plotly_white"
        )
        fig.show()


# ### Load the data

# In[3]:


# --- Load OBD (Random / all as example) ---
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()

print("bandit_feedback keys:", bf.keys())
print(f"Rounds: {bf['n_rounds']:,} | n_actions: {bf['n_actions']} | len_list: {ds.len_list}")

# Build a DataFrame for convenience
df = pd.DataFrame({
    "action": bf["action"],
    "position": bf["position"],
    "reward": bf["reward"],
    "pscore": bf["pscore"],
})

df.head()


# In[4]:


df.shape


# In[5]:


df.groupby("action")["reward"].mean()


# In[6]:


# For CTR per action (your case)
fig = plot_empirical_means(df, group_col="action", value_col="reward", title="Empirical CTR per Action")
fig.show()

# For reward by position
fig2 = plot_empirical_means(df, group_col="position", value_col="reward", title="CTR per Position")
fig2.show()


# In[ ]:





# ### Every action being equally likely of being chosen

# In[7]:


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


n_actions = bf["n_actions"]   # 80
positions = [0, 1, 2]

# Uniform PMFs for each position
pmfs = {pos: np.ones(n_actions) / n_actions for pos in positions}

fig = plot_position_pmfs(pmfs, title="Initial uniform probability mass functions (per position)")


# # Round 1 

# ### 1A. Initialize alphas and betas

# In[8]:


# ---------- Setup ----------
n_actions = bf["n_actions"]     # K=80
positions = [0, 1, 2]

# Priors: Beta(1,1) everywhere (Round 0 state)
alpha = {pos: np.ones(n_actions, dtype=float) for pos in positions}
beta  =  {pos: np.ones(n_actions, dtype=float) for pos in positions}

# ---------- Round 0: before any updates ----------
pmf_round0 = {pos: np.full(n_actions, 1.0 / n_actions) for pos in positions}  # uniform
df_round0  = make_probs_df(round_num=0, alpha=alpha, beta=beta, pmf_by_pos=pmf_round0)


# ### 1B. Sample Observation

# In[9]:


# ---------- Sample one observation per position from your log DF ----------
# NOTE: set random_state=<int> for reproducibility; None for fresh randomness.
sampled = (
    df.groupby("position", group_keys=False)
      .apply(lambda x: x.sample(1, random_state=None))
      .reset_index(drop=True)
)
sampled


# ### 1C. Update based on observations

# In[10]:


# ---------- Round 1: apply sampled updates ----------
for _, r in sampled.iterrows():
    a_obs = int(r["action"])
    p_obs = int(r["position"])
    r_obs = int(r["reward"])
    alpha[p_obs][a_obs] += r_obs
    beta[p_obs][a_obs]  += (1 - r_obs)


# Build PMFs after the (one) update at each position (closed-form)
pmf_round1 = {}
for pos in positions:
    obs_row = sampled.loc[sampled["position"] == pos].iloc[0]
    pmf_round1[pos] = exact_pmf_one_update(
        K=n_actions,
        updated_arm=int(obs_row["action"]),
        reward=int(obs_row["reward"]),
    )

df_round1 = make_probs_df(round_num=1, alpha=alpha, beta=beta, pmf_by_pos=pmf_round1)


# ---------- Combine rounds for tracking ----------
df_probs_all_rounds = pd.concat([df_round0, df_round1], ignore_index=True)

# Peek: show just the three updated actions across rounds
updated_actions = sampled["action"].tolist()
print("\nUpdated actions across rounds:")
df_probs_all_rounds[df_probs_all_rounds["action"].isin(updated_actions)].sort_values(["round","action"])


# In[11]:


print("Sampled (one row per position):")
print(sampled[["action","position","reward","pscore"]])

fig = plot_pmf_compare_by_position(
    pmf_round_A=pmf_round0,
    pmf_round_B=pmf_round1,
    positions=positions,
    n_actions=n_actions,
    title="Selection probabilities by position: Round 0 vs Round 1",
    round_labels=("Round 0 (uniform)", "Round 1 (updated)")
)
fig.show()


# In[24]:


break


# In[22]:


# Assumes you already built df (action, position, reward, pscore) and bf for n_actions
tracker = ThompsonSlateTracker(df=df, n_actions=bf["n_actions"], positions=[0,1,2], seed=7)

# Round 0 is already recorded (uniform)
# Take Round 1 (samples per position, updates α/β, stores PMFs)
samp1 = tracker.next_round(random_state=None, use_mc_if_needed=False)  # exact is valid here
print("Round 1 samples:\n", samp1)

# Plot Round 0 vs Round 1
tracker.plot_compare(0, 1, title="Round 0 vs Round 1")

# Take Round 2 (now exact may be invalid; allow MC to compute PMFs)
samp2 = tracker.next_round(random_state=None, use_mc_if_needed=True)   # enables MC fallback
print("Round 2 samples:\n", samp2)

# Plot Round 1 vs Round 2
tracker.plot_compare(1, 2, title="Round 1 vs Round 2")


# In[14]:


# Round 3 (same tracker instance)
samp3 = tracker.next_round(
    random_state=None,     # or an int for reproducibility
    use_mc_if_needed=True, # True recommended for rounds >= 2
    n_sim=200_000          # tweak if you want faster/lower-variance
)
print("Round 3 samples:\n", samp3)

# Plot Round 2 vs Round 3
tracker.plot_compare(2, 3, title="Round 2 vs Round 3")


# In[15]:


# Run 100 rounds
n_rounds = 100
for r in range(1, n_rounds+1):   # Round 1 .. 100
    samp = tracker.next_round(
        random_state=None,     # or an int for reproducibility
        use_mc_if_needed=True, # needed beyond round 1
        n_sim=50_000           # you can lower this for speed if needed
    )
    print(f"Round {r} samples:\n", samp.head())  # head() just to avoid giant printouts


# In[16]:


# make sure you’ve advanced to round 100 already
# for r in range(1, 101):
#     _ = tracker.next_round(use_mc_if_needed=True, n_sim=50_000)

tracker.plot_compare(0, 100, title="Round 0 vs Round 100")


# In[23]:


break 


# In[17]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_ts_and_plot(df, n_actions, n_rounds, seed=7, alpha0=1.0, beta0=1.0, title=None):
    """
    Online Thompson Sampling (single-slot) using empirical CTR per action as ground-truth.
    Plots selection frequency vs empirical CTR and returns (fig, results_df).

    Parameters
    ----------
    df : pd.DataFrame with columns ["action","reward", ...]
    n_actions : int
    n_rounds : int
    seed : int
    alpha0, beta0 : float
        Beta prior for all actions (default Beta(1,1))
    title : str or None
    """
    # Ground-truth Bernoulli p per action from logs (missing actions -> 0 CTR)
    p_true = (
        df.groupby("action")["reward"].mean()
          .reindex(range(n_actions), fill_value=0.0)
          .to_numpy()
    )
    assert p_true.ndim == 1 and p_true.shape[0] == n_actions

    # TS state
    alpha_ts = np.full(n_actions, float(alpha0))
    beta_ts  = np.full(n_actions, float(beta0))
    chosen_counts = np.zeros(n_actions, dtype=int)

    rng = np.random.RandomState(seed)

    # Online TS loop (draw max, pick, observe, update)
    for _ in range(int(n_rounds)):
        theta = rng.beta(alpha_ts, beta_ts)
        a = int(theta.argmax())
        chosen_counts[a] += 1
        r = rng.rand() < p_true[a]   # simulate Bernoulli reward from empirical CTR
        alpha_ts[a] += int(r)
        beta_ts[a]  += int(not r)

    sel_freq = chosen_counts / float(n_rounds)

    # Assemble results
    res = pd.DataFrame({
        "action": np.arange(n_actions),
        "empirical_ctr": p_true,
        "ts_selection_freq": sel_freq,
        "alpha_final": alpha_ts,
        "beta_final": beta_ts,
        "impressions": df.groupby("action")["reward"].size()
                           .reindex(range(n_actions), fill_value=0)
                           .to_numpy()
    })

    # Plot
    fig = go.Figure()
    fig.add_bar(x=res["action"], y=res["empirical_ctr"], name="Empirical CTR")
    fig.add_bar(x=res["action"], y=res["ts_selection_freq"], name=f"TS selection freq over {n_rounds:,} rounds")
    fig.update_layout(
        title=title or f"Online TS: selection frequency vs empirical CTR (n_rounds={n_rounds:,})",
        barmode="group", template="plotly_white",
        xaxis=dict(title="Action ID",
                   tickmode="array",
                   tickvals=res["action"],
                   ticktext=[str(a) for a in res["action"]]),
        yaxis=dict(title="Value")
    )
    return fig, res


# In[18]:


K = bf["n_actions"]

# 100 rounds
fig100, res100 = run_ts_and_plot(df, n_actions=K, n_rounds=100,  seed=7, title="Round 100")
fig100.show()

# 1,000 rounds
fig1k, res1k = run_ts_and_plot(df, n_actions=K, n_rounds=1_000, seed=7, title="Round 1,000")
fig1k.show()

# 10,000 rounds
fig10k, res10k = run_ts_and_plot(df, n_actions=K, n_rounds=10_000, seed=7, title="Round 10,000")
fig10k.show()


fig100k, res100k = run_ts_and_plot(df, n_actions=K, n_rounds=100_000, seed=7, title="Round 100,000")
fig100k.show()

# 1,000,000 rounds (⚠️ slow; pure Python loop)
# Consider 50k–200k if runtime is high.
# fig1m, res1m = run_ts_and_plot(df, n_actions=K, n_rounds=1_000_000, seed=7, title="Round 1,000,000")
# fig1m.show()


# In[19]:


fig100k, res100k = run_ts_and_plot(df, n_actions=K, n_rounds=100_000, seed=7, title="Round 100,000")
fig100k.show()


# In[ ]:





# ### Position Aware TS

# In[ ]:


import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def run_ts_by_position_and_plot(
    df: pd.DataFrame,
    n_actions: int,
    positions=(0,1,2),
    n_rounds: int = 10_000,
    seed: int = 7,
    alpha0: float = 1.0,
    beta0: float  = 1.0,
    title: str | None = None,
):
    """
    Position-aware online Thompson Sampling (independent bandit per position).
    Uses per-position empirical CTR as the 'true' Bernoulli rate for simulation.

    df must have columns: ['action','position','reward', ...]
    Returns (fig, results_df) where results_df has one row per (position, action).
    """
    positions = list(positions)

    # ----- Ground-truth p_true per (position, action) from logs -----
    # shape: dict[pos] -> length-n_actions float array
    grp_mean = (
        df.groupby(["position","action"])["reward"].mean().unstack(fill_value=np.nan)
    )
    # reindex to ensure we have full action range for each position
    full_cols = pd.Index(range(n_actions), name="action")
    grp_mean = grp_mean.reindex(columns=full_cols)

    p_true = {}
    for pos in positions:
        # if a position is missing in df, fill with zeros
        if pos in grp_mean.index:
            vec = grp_mean.loc[pos].to_numpy()
        else:
            vec = np.zeros(n_actions, dtype=float)
        # NaNs (no impressions for an action at that pos) -> 0 CTR
        vec = np.nan_to_num(vec, nan=0.0)
        p_true[pos] = vec

    # impressions per (pos, action) (optional diagnostics)
    grp_count = (
        df.groupby(["position","action"])["reward"].size().unstack(fill_value=0)
         .reindex(index=positions, columns=full_cols, fill_value=0)
    )

    # ----- TS state per position -----
    alpha = {pos: np.full(n_actions, float(alpha0)) for pos in positions}
    beta  = {pos: np.full(n_actions, float(beta0))  for pos in positions}
    chosen_counts = {pos: np.zeros(n_actions, dtype=int) for pos in positions}

    rng = np.random.RandomState(seed)

    # ----- Online TS loop (independent per position each round) -----
    for _ in range(int(n_rounds)):
        for pos in positions:
            theta = rng.beta(alpha[pos], beta[pos])
            a = int(theta.argmax())
            chosen_counts[pos][a] += 1
            # simulate reward with per-position ground-truth CTR
            r = rng.rand() < p_true[pos][a]
            alpha[pos][a] += int(r)
            beta[pos][a]  += int(not r)

    # ----- Collect results into a tidy DataFrame -----
    records = []
    for pos in positions:
        sel_freq = chosen_counts[pos] / float(n_rounds)
        for a in range(n_actions):
            records.append({
                "position": pos,
                "action": a,
                "empirical_ctr": float(p_true[pos][a]),
                "ts_selection_freq": float(sel_freq[a]),
                "alpha_final": float(alpha[pos][a]),
                "beta_final": float(beta[pos][a]),
                "impressions": int(grp_count.loc[pos, a]) if pos in grp_count.index else 0,
            })
    res = pd.DataFrame(records)

    # ----- Plot (one row per position, grouped bars) -----
    fig = make_subplots(
        rows=len(positions), cols=1, shared_xaxes=False,
        subplot_titles=[f"Position {p}" for p in positions],
        vertical_spacing=0.08
    )

    for i, pos in enumerate(positions, start=1):
        df_pos = res[res["position"] == pos].sort_values("action")
        xs = df_pos["action"].to_numpy()

        fig.add_bar(
            x=xs, y=df_pos["empirical_ctr"], name="Empirical CTR",
            marker_color="steelblue", opacity=0.6,
            showlegend=(i == 1), row=i, col=1
        )
        fig.add_bar(
            x=xs, y=df_pos["ts_selection_freq"], name=f"TS selection freq (n={n_rounds:,})",
            marker_color="darkorange", opacity=0.85,
            showlegend=(i == 1), row=i, col=1
        )

        # pretty axes per row
        fig.update_yaxes(title_text="Value", row=i, col=1)
        fig.update_xaxes(
            title_text="Action ID (0–{})".format(n_actions-1) if i==len(positions) else "",
            tickmode="array",
            tickvals=xs,
            ticktext=[str(a) for a in xs],
            row=i, col=1
        )

    fig.update_layout(
        title=title or f"Position-aware TS vs Empirical CTR (n_rounds={n_rounds:,})",
        barmode="group", template="plotly_white",
        height=300*len(positions) + 100,
    )
    return fig, res


# In[ ]:


K = bf["n_actions"]
fig, res = run_ts_by_position_and_plot(
    df=df,
    n_actions=K,
    positions=[0,1,2],
    n_rounds=1000,   # try 100, 1_000, 10_000
    seed=7,
    alpha0=1.0, beta0=1.0,
    title="Thompson Sampling per position after 1000 rounds"
)
fig.show()

# Inspect top actions TS focused on at each position:
(res.sort_values(["position","ts_selection_freq"], ascending=[True,False])
    .groupby("position").head(5))


# In[ ]:


K = bf["n_actions"]
fig, res = run_ts_by_position_and_plot(
    df=df,
    n_actions=K,
    positions=[0,1,2],
    n_rounds=10_000,   # try 100, 1_000, 10_000
    seed=7,
    alpha0=1.0, beta0=1.0,
    title="Thompson Sampling per position after 10k rounds"
)
fig.show()

# Inspect top actions TS focused on at each position:
(res.sort_values(["position","ts_selection_freq"], ascending=[True,False])
    .groupby("position").head(5))


# In[ ]:


K = bf["n_actions"]
fig, res = run_ts_by_position_and_plot(
    df=df,
    n_actions=K,
    positions=[0,1,2],
    n_rounds=100_000,   # try 100, 1_000, 10_000
    seed=7,
    alpha0=1.0, beta0=1.0,
    title="Thompson Sampling per position after 100k rounds"
)
fig.show()

# Inspect top actions TS focused on at each position:
(res.sort_values(["position","ts_selection_freq"], ascending=[True,False])
    .groupby("position").head(5))


# ### Using Open Bandit Pipeline

# In[ ]:


import numpy as np
import pandas as pd
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# 1) Load OBD (Random/all) and prep a simple df
# -----------------------------
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()

df = pd.DataFrame({
    "action": bf["action"],
    "position": bf["position"],   # 0,1,2 in OBP small
    "reward": bf["reward"].astype(int),
    "pscore": bf["pscore"],
})

K = bf["n_actions"]           # 80
positions = [0, 1, 2]
SEED = 7

# -----------------------------
# 2) Init OBP’s BernoulliTS (evaluation policy)
#    - You can use ZOZOTOWN prior or plain Beta(1,1) prior
# -----------------------------
pi_e = BernoulliTS(
    n_actions=K,
    len_list=len(positions),
    is_zozotown_prior=False,   # set True to use ZOZOTOWN prior
    campaign="all",
    random_state=SEED,
)

def snapshot_pmf(policy: BernoulliTS):
    """
    Returns per-position selection probs as a dict {pos -> np.array(K)}
    OBP's compute_batch_action_dist returns shape (n_rounds, n_actions, len_list),
    so we take [0] and transpose to get (len_list, n_actions).
    """
    ad = policy.compute_batch_action_dist(n_rounds=1, n_sim=100_000)[0]  # (K, len_list)
    ad = ad.T  # -> (len_list, K)
    return {pos: ad[i] for i, pos in enumerate(positions)}

def sample_one_per_position(df):
    """One logged event per position."""
    return (df.groupby("position", group_keys=False)
              .apply(lambda x: x.sample(1, random_state=None))
              .reset_index(drop=True))

# -----------------------------
# 3) Round 0 snapshot (before any learning)
# -----------------------------
pmf_round0 = snapshot_pmf(pi_e)   # dict: pos -> (K,)

# -----------------------------
# 4) Run R rounds of "offline online" updates from the logs
#    (factorized by position: one update per position per round)
# -----------------------------
R = 100  # change to 1_000, 10_000, etc.
for _ in range(R):
    samp = sample_one_per_position(df)   # 3 rows (pos 0,1,2)
    for _, r in samp.iterrows():
        a = int(r["action"])
        y = int(r["reward"])
        # Update OBP policy with that arm's observed reward
        pi_e.update_params(action=a, reward=y)

# After R rounds, snapshot again
pmf_roundR = snapshot_pmf(pi_e)

# -----------------------------
# 5) Plot Round 0 vs Round R (per position)
# -----------------------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False,
    subplot_titles=[f"Position {p}" for p in positions],
    vertical_spacing=0.08
)

for i, pos in enumerate(positions, start=1):
    xs = np.arange(K)
    y0 = pmf_round0[pos]
    yR = pmf_roundR[pos]
    ymax = float(max(y0.max(), yR.max())) * 1.25

    fig.add_bar(x=xs, y=y0, name=f"Round 0", marker_color="steelblue",
                opacity=0.6, showlegend=(i==1), row=i, col=1)
    fig.add_bar(x=xs, y=yR, name=f"Round {R}", marker_color="darkorange",
                opacity=0.85, showlegend=(i==1), row=i, col=1)

    fig.update_yaxes(range=[0, ymax], title_text="P(select)", row=i, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=xs,
        ticktext=[str(a) for a in xs],
        title_text="Action ID (0–79)" if i==3 else "",
        row=i, col=1
    )

fig.update_layout(
    title=f"OBP BernoulliTS selection probabilities — Round 0 vs Round {R}",
    barmode="group", bargap=0.15, template="plotly_white",
    height=900
)
fig.show()

# -----------------------------
# 6) (Optional) Export a tidy DataFrame of the PMFs
# -----------------------------
def pmf_to_df(pmf_dict, round_id):
    recs = []
    for pos in positions:
        for a, p in enumerate(pmf_dict[pos]):
            recs.append({"round": round_id, "position": pos, "action": a, "p_select": float(p)})
    return pd.DataFrame(recs)

df_pmf0 = pmf_to_df(pmf_round0, 0)
df_pmfR = pmf_to_df(pmf_roundR, R)
df_pmfs = pd.concat([df_pmf0, df_pmfR], ignore_index=True)
# df_pmfs.to_csv(f"obp_pmf_rounds_0_{R}.csv", index=False)


# In[ ]:


import numpy as np
import pandas as pd
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# Load logs and prep
# -----------------------------
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()

df = pd.DataFrame({
    "action":  bf["action"],
    "position": bf["position"],    # 0,1,2
    "reward":  bf["reward"].astype(int),
    "pscore":  bf["pscore"],
})
K = bf["n_actions"]          # 80
positions = [0,1,2]
SEED = 7

# -----------------------------
# Empirical CTR per (position, action)
# -----------------------------
# mean reward by (pos, action), fill missing with 0
ctr_pos_act = (
    df.groupby(["position","action"])["reward"].mean().unstack(fill_value=np.nan)
      .reindex(index=positions, columns=range(K))
      .fillna(0.0)
)

# -----------------------------
# Init BernoulliTS policy
#   - set is_zozotown_prior=True if you want their production prior
# -----------------------------
pi_e = BernoulliTS(
    n_actions=K,
    len_list=len(positions),
    is_zozotown_prior=False,
    campaign="all",
    random_state=SEED,
)

def snapshot_pmf(policy: BernoulliTS, n_sim=50_000):
    """
    Returns dict: {pos -> np.array(K)} of selection probabilities.
    OBP returns (n_rounds, n_actions, len_list); take [0] and transpose.
    """
    ad = policy.compute_batch_action_dist(n_rounds=1, n_sim=n_sim)[0]  # (K, len_list)
    ad = ad.T  # -> (len_list, K)
    return {pos: ad[i] for i, pos in enumerate(positions)}

def sample_one_per_position(df):
    """Sample exactly one logged event per position."""
    return (df.groupby("position", group_keys=False)
              .apply(lambda x: x.sample(1, random_state=None))
              .reset_index(drop=True))

# -----------------------------
# Run R rounds of “offline online” updates
# -----------------------------
R = 10_000
for _ in range(R):
    samp = sample_one_per_position(df)
    for _, r in samp.iterrows():
        a = int(r["action"])
        y = int(r["reward"])
        pi_e.update_params(action=a, reward=y)

pmf_after = snapshot_pmf(pi_e, n_sim=100_000)  # selection probabilities after 10k rounds

# -----------------------------
# Plot: Empirical CTR vs. Policy Selection Probability (per position)
# -----------------------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False,
    subplot_titles=[f"Position {p}" for p in positions],
    vertical_spacing=0.08
)

for i, pos in enumerate(positions, start=1):
    xs = np.arange(K)
    y_ctr = ctr_pos_act.loc[pos].to_numpy()
    y_sel = pmf_after[pos]

    fig.add_bar(
        x=xs, y=y_ctr, name="Empirical CTR",
        marker_color="steelblue", opacity=0.6,
        showlegend=(i==1), row=i, col=1
    )
    fig.add_bar(
        x=xs, y=y_sel, name=f"Policy selection prob (after {R:,})",
        marker_color="darkorange", opacity=0.85,
        showlegend=(i==1), row=i, col=1
    )

    ymax = float(max(y_ctr.max(), y_sel.max())) * 1.25 if (y_ctr.max() or y_sel.max()) else 0.02
    fig.update_yaxes(title_text="Value", range=[0, ymax], row=i, col=1)
    fig.update_xaxes(
        title_text="Action ID (0–79)" if i==3 else "",
        tickmode="array",
        tickvals=xs,
        ticktext=[str(a) for a in xs],
        row=i, col=1
    )

fig.update_layout(
    title=f"OBP BernoulliTS: Selection Probability vs Empirical CTR (after {R:,} rounds)",
    barmode="group", bargap=0.15, template="plotly_white",
    height=900
)
fig.show()

# -----------------------------
# (Optional) Tidy table you can export
# -----------------------------
rows = []
for pos in positions:
    for a in range(K):
        rows.append({
            "position": pos,
            "action": a,
            "empirical_ctr": float(ctr_pos_act.loc[pos, a]),
            "policy_p_select_after": float(pmf_after[pos][a]),
        })
df_compare = pd.DataFrame(rows)
# df_compare.to_csv("obp_compare_ctr_vs_policy_p_after_10k.csv", index=False)


# In[ ]:


import numpy as np
import pandas as pd
from obp.policy import BernoulliTS
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def run_obp_ts_factorized_and_plot(
    df: pd.DataFrame,
    n_actions: int,
    positions=(0,1,2),
    R: int = 10_000,
    seed: int = 7,
    use_zozo_prior: bool = False,
    n_sim_snapshot: int = 100_000,
    title: str | None = None,
):
    """
    Factorized slate: one BernoulliTS per position (slot).
    Updates each policy only with samples from its slot, then compares
    empirical CTR vs policy selection probability per position.

    df columns: ['action','position','reward', ...]
    """
    positions = list(positions)

    # per-(pos,action) empirical CTR as ground truth for plotting reference
    ctr_pos_act = (
        df.groupby(["position","action"])["reward"].mean().unstack(fill_value=np.nan)
          .reindex(index=positions, columns=range(n_actions))
          .fillna(0.0)
    )

    # one TS policy per position (len_list=1 because each policy controls one slot)
    policies = {
        pos: BernoulliTS(
            n_actions=n_actions,
            len_list=1,
            is_zozotown_prior=use_zozo_prior,
            campaign="all",
            random_state=seed + pos,
        )
        for pos in positions
    }

    # helper: snapshot PMF for a single-position policy (returns length-n_actions)
    def snapshot_vec(policy: BernoulliTS) -> np.ndarray:
        # OBP shape: (n_rounds, n_actions, len_list). len_list=1 here.
        ad = policy.compute_batch_action_dist(n_rounds=1, n_sim=n_sim_snapshot)[0, :, 0]
        return ad  # (n_actions,)

    # --- run R rounds: sample one row per position, update that position's policy
    for _ in range(R):
        samp = (df.groupby("position", group_keys=False)
                  .apply(lambda x: x.sample(1, random_state=None))
                  .reset_index(drop=True))
        for _, r in samp.iterrows():
            pos = int(r["position"]); a = int(r["action"]); y = int(r["reward"])
            policies[pos].update_params(action=a, reward=y)

    # snapshot after training
    pmf_after = {pos: snapshot_vec(policies[pos]) for pos in positions}

    # --- build tidy result frame
    rows = []
    for pos in positions:
        for a in range(n_actions):
            rows.append({
                "position": pos,
                "action": a,
                "empirical_ctr": float(ctr_pos_act.loc[pos, a]),
                "policy_p_select_after": float(pmf_after[pos][a]),
            })
    res = pd.DataFrame(rows)

    # --- plot per position
    fig = make_subplots(
        rows=len(positions), cols=1, shared_xaxes=False,
        subplot_titles=[f"Position {p}" for p in positions],
        vertical_spacing=0.08
    )
    for i, pos in enumerate(positions, start=1):
        dfp = res[res["position"] == pos].sort_values("action")
        xs = dfp["action"].to_numpy()
        y_ctr = dfp["empirical_ctr"].to_numpy()
        y_sel = dfp["policy_p_select_after"].to_numpy()
        ymax = float(max(y_ctr.max(), y_sel.max())) * 1.25 if (y_ctr.max() or y_sel.max()) else 0.02

        fig.add_bar(x=xs, y=y_ctr, name="Empirical CTR",
                    marker_color="steelblue", opacity=0.6, showlegend=(i==1),
                    row=i, col=1)
        fig.add_bar(x=xs, y=y_sel, name=f"Policy P(select) after {R:,}",
                    marker_color="darkorange", opacity=0.85, showlegend=(i==1),
                    row=i, col=1)

        fig.update_yaxes(title_text="Value", range=[0, ymax], row=i, col=1)
        fig.update_xaxes(
            title_text="Action ID (0–{})".format(n_actions-1) if i==len(positions) else "",
            tickmode="array", tickvals=xs, ticktext=[str(a) for a in xs],
            row=i, col=1
        )

    fig.update_layout(
        title=title or f"Factorized BernoulliTS (one policy per position) — after {R:,} rounds",
        barmode="group", bargap=0.15, template="plotly_white",
        height=300*len(positions) + 120,
    )
    return fig, res


K = bf["n_actions"]
fig, res = run_obp_ts_factorized_and_plot(
    df=df, n_actions=K, positions=[0,1,2],
    R=10_000, seed=7, use_zozo_prior=False, n_sim_snapshot=100_000,
    title="Position-aware (factorized) TS vs Empirical CTR - 10k rounds"
)
fig.show()


# In[ ]:


K = bf["n_actions"]
fig, res = run_obp_ts_factorized_and_plot(
    df=df, n_actions=K, positions=[0,1,2],
    R=100_000, seed=7, use_zozo_prior=False, n_sim_snapshot=100_000,
    title="Position-aware (factorized) TS vs Empirical CTR - 100k rounds" 
)
fig.show()


# ### Running an RL loop to compare policies learned

# In[ ]:


import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Build the simulator from logs: p_true[pos, action] = empirical CTR ---
def make_ctr_table(df, n_actions, positions=(0,1,2)):
    ctr = (df.groupby(["position","action"])["reward"].mean()
             .unstack(fill_value=np.nan)
             .reindex(index=positions, columns=range(n_actions))
             .fillna(0.0))
    return ctr  # DataFrame: index=position, cols=action

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class PositionAwareSoftmaxPolicy:
    """
    One softmax over actions per position (no context).
    logits[pos, action] are learned. Produces a slate by sampling per position independently.
    """
    def __init__(self, n_actions, positions=(0,1,2), seed=7, init_scale=0.0):
        self.n_actions = n_actions
        self.positions = list(positions)
        rng = np.random.RandomState(seed)
        self.logits = {p: init_scale * rng.randn(n_actions) for p in self.positions}

    def action_probs(self, pos):
        return softmax(self.logits[pos])

    def sample_action(self, pos, rng):
        p = self.action_probs(pos)
        return int(rng.choice(self.n_actions, p=p)), p

    def update_reinforce(self, grads, lr=0.1):
        # grads: dict pos -> grad vector length n_actions
        for p in self.positions:
            self.logits[p] += lr * grads[p]

def run_reinforce_training(ctr_df, n_rounds=10000, lr=0.1, baseline=True, seed=7):
    """
    Simple REINFORCE on a position-factorized bandit:
      At each round, for each position pick an action from policy softmax, get Bernoulli reward with p_true,
      and do REINFORCE update (optionally with average-reward baseline per position).
    """
    positions = list(ctr_df.index)
    n_actions = ctr_df.shape[1]
    rng = np.random.RandomState(seed)

    policy = PositionAwareSoftmaxPolicy(n_actions, positions, seed=seed)
    running_baseline = {p: 0.0 for p in positions}
    beta = 0.99  # baseline EMA factor

    for t in range(n_rounds):
        grads = {p: np.zeros(n_actions) for p in positions}
        rewards = {}
        for p in positions:
            a, pvec = policy.sample_action(p, rng)
            r = float(rng.rand() < ctr_df.loc[p, a])  # simulate reward
            rewards[p] = r
            # gradient of log softmax at chosen action = (onehot - pvec)
            g = -pvec
            g[a] += 1.0
            # REINFORCE weight = (r - b)
            adv = r - (running_baseline[p] if baseline else 0.0)
            grads[p] += adv * g

        # baseline updates
        if baseline:
            for p in positions:
                running_baseline[p] = beta * running_baseline[p] + (1 - beta) * rewards[p]

        policy.update_reinforce(grads, lr=lr)

    # Return learned per-position distributions
    learned = {p: policy.action_probs(p) for p in positions}
    return policy, learned

# --- Use your df/bf from earlier ---
# df must have columns: action, position (0/1/2), reward∈{0,1}
K = bf["n_actions"]
positions = [0,1,2]
ctr_pos_act = make_ctr_table(df, n_actions=K, positions=positions)

# Train RL policy
policy, learned_p = run_reinforce_training(
    ctr_pos_act, n_rounds=10_000, lr=0.1, baseline=True, seed=7
)

# Plot learned selection probs vs empirical CTR per position
fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                    subplot_titles=[f"Position {p}" for p in positions],
                    vertical_spacing=0.08)

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
        title_text="Action ID (0–{})".format(K-1) if i==3 else "",
        tickmode="array", tickvals=xs, ticktext=[str(a) for a in xs],
        row=i, col=1
    )

fig.update_layout(
    title="Position-aware REINFORCE (simulated env): Learned selection vs Empirical CTR",
    barmode="group", bargap=0.15, template="plotly_white", height=900
)
fig.show()


# In[ ]:


import numpy as np
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW, SelfNormalizedInverseProbabilityWeighting as SNIPW

K         = bf["n_actions"]
len_list  = len(positions)          # should be 3 for OBD
n_rounds  = bf["n_rounds"]

# Build action_dist in the expected shape: (n_rounds, n_actions, len_list)
action_dist = np.zeros((n_rounds, K, len_list), dtype=float)

# learned_p[pos] must be a length-K prob vector that sums to 1
for slot_index, pos in enumerate(positions):
    vec = np.asarray(learned_p[pos], dtype=float)
    # safety: renormalize tiny numeric drift
    s = vec.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"learned_p[{pos}] is invalid (sum={s}).")
    vec = vec / s
    # repeat the same per-position distribution across all rounds
    action_dist[:, :, slot_index] = vec[None, :]

# --- validations (OBP will do similar checks) ---
assert action_dist.shape == (n_rounds, K, len_list)
assert np.all(np.isfinite(action_dist))
assert np.all(action_dist >= 0)
row_sums = action_dist.sum(axis=1)  # shape (n_rounds, len_list); sums over actions
assert np.allclose(row_sums, 1.0, atol=1e-8), (row_sums.min(), row_sums.max())

# Run OPE
ope = OffPolicyEvaluation(bandit_feedback=bf, ope_estimators=[IPW(), SNIPW()])
est = ope.estimate_policy_values(action_dist=action_dist)

logged = bf["reward"].mean()
print("Logged avg reward:", logged)
print("IPW:   ", est["ipw"])
print("SNIPW: ", est["snipw"])
print("Relative (IPW / logged):", est["ipw"] / logged)


# In[ ]:


import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW, SelfNormalizedInverseProbabilityWeighting as SNIPW

# ---------- Data prep ----------
def make_ctr_table(df: pd.DataFrame, n_actions: int, positions=(0,1,2)) -> pd.DataFrame:
    ctr = (df.groupby(["position","action"])["reward"].mean()
             .unstack(fill_value=np.nan)
             .reindex(index=list(positions), columns=range(n_actions))
             .fillna(0.0))
    return ctr

# ---------- Simple position-aware REINFORCE ----------
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

class PositionAwareSoftmaxPolicy:
    def __init__(self, n_actions, positions=(0,1,2), seed=7, init_scale=0.0):
        self.n_actions = int(n_actions)
        self.positions = list(positions)
        rng = np.random.RandomState(seed)
        self.logits = {p: init_scale * rng.randn(self.n_actions) for p in self.positions}

    def action_probs(self, pos: int) -> np.ndarray:
        return _softmax(self.logits[pos])

    def sample_action(self, pos: int, rng) -> tuple[int, np.ndarray]:
        p = self.action_probs(pos)
        a = int(rng.choice(self.n_actions, p=p))
        return a, p

    def update_reinforce(self, grads: dict[int, np.ndarray], lr: float = 0.1):
        for p in self.positions:
            self.logits[p] += lr * grads[p]

def run_reinforce_training(
    ctr_df: pd.DataFrame,
    n_rounds: int = 10_000,
    lr: float = 0.1,
    baseline: bool = True,
    seed: int = 7,
    init_scale: float = 0.0,
):
    positions = list(ctr_df.index)
    n_actions = ctr_df.shape[1]
    rng = np.random.RandomState(seed)

    policy = PositionAwareSoftmaxPolicy(n_actions, positions, seed=seed, init_scale=init_scale)
    running_baseline = {p: 0.0 for p in positions}
    beta = 0.99

    for _ in range(n_rounds):
        grads = {p: np.zeros(n_actions) for p in positions}
        rewards = {}
        for p in positions:
            a, pvec = policy.sample_action(p, rng)
            r = float(rng.rand() < ctr_df.loc[p, a])
            rewards[p] = r
            g = -pvec; g[a] += 1.0
            adv = r - (running_baseline[p] if baseline else 0.0)
            grads[p] += adv * g

        if baseline:
            for p in positions:
                running_baseline[p] = beta * running_baseline[p] + (1 - beta) * rewards[p]

        policy.update_reinforce(grads, lr=lr)

    learned_p = {p: policy.action_probs(p) for p in positions}
    return policy, learned_p

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

# ---------- OPE ----------
def learned_to_action_dist(learned_p: dict[int, np.ndarray], bf: dict, positions=(0,1,2)) -> np.ndarray:
    K = bf["n_actions"]; T = bf["n_rounds"]; L = len(positions)
    action_dist = np.zeros((T, K, L), dtype=float)
    for slot_idx, pos in enumerate(positions):
        vec = np.asarray(learned_p[pos], dtype=float)
        s = vec.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"learned_p[{pos}] invalid (sum={s}).")
        action_dist[:, :, slot_idx] = (vec / s)[None, :]
    row_sums = action_dist.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError(f"action_dist rows must sum to 1; got min/max {row_sums.min()}, {row_sums.max()}")
    return action_dist

def eval_with_ope(bf: dict, action_dist: np.ndarray):
    ope = OffPolicyEvaluation(bandit_feedback=bf, ope_estimators=[IPW(), SNIPW()])
    est = ope.estimate_policy_values(action_dist=action_dist)
    logged = float(np.mean(bf["reward"]))
    return {
        "logged_avg_reward": logged,
        "ipw": float(est["ipw"]),
        "snipw": float(est["snipw"]),
        "ipw_over_logged": float(est["ipw"] / logged if logged > 0 else np.nan),
        "snipw_over_logged": float(est["snipw"] / logged if logged > 0 else np.nan),
    }

# ---------- Title formatter & end-to-end ----------
def _format_config_title(n_rounds_rl, lr, baseline, seed, positions, prefix="REINFORCE"):
    pos_txt = ",".join(str(p) for p in positions)
    base_txt = "on" if baseline else "off"
    return (f"{prefix} — rounds={n_rounds_rl:,}, lr={lr}, baseline={base_txt}, "
            f"seed={seed}, positions=[{pos_txt}]")

def train_plot_ope(
    df: pd.DataFrame,
    bf: dict,
    positions=(0,1,2),
    n_rounds_rl: int = 10_000,
    lr: float = 0.1,
    baseline: bool = True,
    seed: int = 7,
    init_scale: float = 0.0,
    title: str | None = None,
):
    K = bf["n_actions"]
    ctr_pos_act = make_ctr_table(df, n_actions=K, positions=positions)
    _, learned_p = run_reinforce_training(
        ctr_pos_act, n_rounds=n_rounds_rl, lr=lr, baseline=baseline, seed=seed, init_scale=init_scale
    )

    # auto-title if none provided
    auto_title = _format_config_title(n_rounds_rl, lr, baseline, seed, positions)
    fig = plot_learned_vs_ctr(ctr_pos_act, learned_p, title or auto_title)

    action_dist = learned_to_action_dist(learned_p, bf, positions=positions)
    metrics = eval_with_ope(bf, action_dist)
    return fig, learned_p, action_dist, metrics

# ---------- Batch sweep ----------
def sweep_and_compare(
    df: pd.DataFrame,
    bf: dict,
    configs: list[dict],
    positions=(0,1,2),
):
    rows = []
    for cfg in configs:
        fig, learned_p, action_dist, metrics = train_plot_ope(
            df=df, bf=bf, positions=positions, **cfg
        )
        rows.append({"config": cfg, **metrics})
    return pd.DataFrame(rows)


# In[ ]:


fig, learned_p, action_dist, metrics = train_plot_ope(
    df=df, bf=bf,
    positions=[0,1,2],
    n_rounds_rl=20_000,
    lr=0.05,
    baseline=False,
    seed=42
)
fig.show()
print(metrics)


# ### Comparing Policy performance between contextual bandit and RL

# In[ ]:


import numpy as np
import pandas as pd
from obp.policy import BernoulliTS
import plotly.graph_objects as go

# --- reuse your utilities from before ---
# make_ctr_table, run_reinforce_training, learned_to_action_dist, eval_with_ope, plot_learned_vs_ctr, train_plot_ope

def train_factorized_ts(
    df: pd.DataFrame,
    n_actions: int,
    positions=(0,1,2),
    R: int = 10_000,
    seed: int = 7,
    use_zozo_prior: bool = False,
    n_sim_snapshot: int = 100_000,
):
    """
    One BernoulliTS per position; update each with samples from its slot for R rounds.
    Returns: pmf_after (dict pos -> length-K prob vector).
    """
    positions = list(positions)
    policies = {
        pos: BernoulliTS(
            n_actions=n_actions, len_list=1,
            is_zozotown_prior=use_zozo_prior, campaign="all",
            random_state=seed + pos
        )
        for pos in positions
    }

    # simulate R updates using one logged row per position each round
    for _ in range(R):
        samp = (df.groupby("position", group_keys=False)
                  .apply(lambda x: x.sample(1, random_state=None))
                  .reset_index(drop=True))
        for _, r in samp.iterrows():
            pos = int(r["position"]); a = int(r["action"]); y = int(r["reward"])
            policies[pos].update_params(action=a, reward=y)

    # snapshot per-position selection distribution (len_list=1 => (1, K, 1))
    pmf_after = {}
    for pos in positions:
        ad = policies[pos].compute_batch_action_dist(n_rounds=1, n_sim=n_sim_snapshot)
        pmf_after[pos] = ad[0, :, 0]  # (K,)
        # normalize just in case of tiny MC drift
        s = pmf_after[pos].sum()
        pmf_after[pos] = pmf_after[pos] / s if s > 0 else np.full(n_actions, 1.0/n_actions)

    return pmf_after

def ts_to_action_dist(pmf_by_pos: dict[int, np.ndarray], bf: dict, positions=(0,1,2)) -> np.ndarray:
    """
    Convert factorized TS per-position vectors into OBP shape (T, K, L).
    """
    K = bf["n_actions"]; T = bf["n_rounds"]; L = len(positions)
    action_dist = np.zeros((T, K, L), dtype=float)
    for slot_idx, pos in enumerate(positions):
        vec = np.asarray(pmf_by_pos[pos], dtype=float)
        s = vec.sum(); vec = vec / s if s > 0 else np.full(K, 1.0/K)
        action_dist[:, :, slot_idx] = vec[None, :]
    # safety check
    sums = action_dist.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-8):
        raise ValueError(f"action_dist rows must sum to 1 (min={sums.min()}, max={sums.max()})")
    return action_dist

def compare_rl_vs_ts(
    df: pd.DataFrame,
    bf: dict,
    positions=(0,1,2),
    # RL config
    n_rounds_rl=10_000, lr=0.1, baseline=True, seed_rl=7, init_scale=0.0,
    # TS config
    R_ts=10_000, seed_ts=7, use_zozo_prior=False, n_sim_snapshot=100_000,
):
    """
    Train RL and factorized TS, evaluate both with OPE, and return:
      - results_df: metrics table
      - figs: dict with 'rl_vs_ctr' and 'bar_metrics' Plotly figs
      - action_dists: dict with 'rl' and 'ts' OBP-shaped tensors
    """
    K = bf["n_actions"]

    # 1) RL: train + plot + OPE
    fig_rl, learned_p_rl, action_dist_rl, metrics_rl = train_plot_ope(
        df=df, bf=bf, positions=positions,
        n_rounds_rl=n_rounds_rl, lr=lr, baseline=baseline, seed=seed_rl, init_scale=init_scale,
        title=None  # auto-title already includes config
    )

    # 2) Factorized TS: train + OPE
    pmf_ts = train_factorized_ts(
        df=df, n_actions=K, positions=positions,
        R=R_ts, seed=seed_ts, use_zozo_prior=use_zozo_prior, n_sim_snapshot=n_sim_snapshot,
    )
    action_dist_ts = ts_to_action_dist(pmf_ts, bf, positions=positions)
    metrics_ts = eval_with_ope(bf, action_dist_ts)

    # 3) Tidy comparison table
    results = pd.DataFrame([
        {"policy": "RL (REINFORCE)",
         "logged_avg_reward": metrics_rl["logged_avg_reward"],
         "ipw": metrics_rl["ipw"],
         "snipw": metrics_rl["snipw"],
         "ipw_over_logged": metrics_rl["ipw_over_logged"],
         "snipw_over_logged": metrics_rl["snipw_over_logged"],
         "config": {"n_rounds_rl": n_rounds_rl, "lr": lr, "baseline": baseline, "seed": seed_rl}},
        {"policy": "Factorized TS",
         "logged_avg_reward": metrics_ts["logged_avg_reward"],
         "ipw": metrics_ts["ipw"],
         "snipw": metrics_ts["snipw"],
         "ipw_over_logged": metrics_ts["ipw_over_logged"],
         "snipw_over_logged": metrics_ts["snipw_over_logged"],
         "config": {"R_ts": R_ts, "seed": seed_ts, "use_zozo_prior": use_zozo_prior}},
    ])

    # 4) Small bar chart comparing estimated values
    fig_bar = go.Figure()
    for row in results.itertuples(index=False):
        fig_bar.add_bar(name=row.policy, x=["IPW","SNIPW"], y=[row.ipw, row.snipw])
    fig_bar.update_layout(
        title="OPE comparison: RL vs Factorized TS (higher is better)",
        barmode="group", template="plotly_white", yaxis_title="Estimated policy value"
    )

    figs = {"rl_vs_ctr": fig_rl, "bar_metrics": fig_bar}
    action_dists = {"rl": action_dist_rl, "ts": action_dist_ts}
    return results, figs, action_dists


# In[ ]:


bf


# In[ ]:


# Compare with defaults (10k RL steps vs 10k TS updates)
results, figs, action_dists = compare_rl_vs_ts(
    df=df, bf=bf, positions=[0,1,2],
    n_rounds_rl=10_000, lr=0.1, baseline=True, seed_rl=7,
    R_ts=10_000, seed_ts=7, use_zozo_prior=False
)

# Show plots
figs["rl_vs_ctr"].show()   # RL selection probs vs empirical CTR (per position)
figs["bar_metrics"].show() # IPW & SNIPW bars: RL vs TS

# Inspect table
print(results)

# If you want to feed either policy into other OBP estimators, you already have:
# action_dists["rl"]  and  action_dists["ts"]  in shape (n_rounds, n_actions, len_list)


# In[ ]:


results


# In[ ]:


results['config'][0]


# In[ ]:


results['config'][1]


# 

# In[ ]:


# Suppose `results` is your DataFrame
config_df = pd.json_normalize(results["config"])   # turn list of dicts into a DataFrame
results_expanded = results.drop(columns="config").join(config_df)

results_expanded


# In[ ]:


df.head()


# In[25]:


get_ipython().system('jupyter nbconvert --to script bandit.ipynb')


# In[ ]:




