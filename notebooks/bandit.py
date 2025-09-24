#!/usr/bin/env python
# coding: utf-8

# In[9]:


# --- Offline simulation: probability of each action over 5 rounds (single slot) ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obp.dataset import OpenBanditDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED = 7
rng = np.random.RandomState(SEED)


# ### Helper Functions

# In[16]:


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


# ### Load the data

# In[13]:


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


# In[14]:


df.shape


# ### Every action being equally likely of being chosen

# In[15]:


n_actions = bf["n_actions"]   # 80
positions = [0, 1, 2]

# Uniform PMF for each position
pmfs = {pos: np.ones(n_actions) / n_actions for pos in positions}

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for pos, ax in zip(positions, axes):
    ax.bar(np.arange(n_actions), pmfs[pos], color="steelblue")
    ax.set_ylabel(f"P(action) @ pos {pos}")
    ax.set_ylim(0, 0.02)  # ~1/80 = 0.0125
axes[-1].set_xlabel("Action ID (0–79)")
fig.suptitle("Initial uniform probability mass functions (per position)")
plt.tight_layout()
plt.show()


# ### Round 1 

# In[ ]:


sampled = df.groupby("position", group_keys=False).apply(lambda x: x.sample(1, random_state=None))
sampled

# ---------- Setup ----------
n_actions = bf["n_actions"]     # K=80
positions = [0, 1, 2]

# Priors: Beta(1,1) everywhere (Round 0 state)
alpha = {pos: np.ones(n_actions, dtype=float) for pos in positions}
beta  =  {pos: np.ones(n_actions, dtype=float) for pos in positions}



# ---------- Round 0: before any updates ----------
pmf_round0 = {pos: np.full(n_actions, 1.0 / n_actions) for pos in positions}  # uniform
df_round0  = make_probs_df(round_num=0, alpha=alpha, beta=beta, pmf_by_pos=pmf_round0)

# ---------- Sample one observation per position from your log DF ----------
# NOTE: set random_state=<int> for reproducibility; None for fresh randomness.
sampled = (
    df.groupby("position", group_keys=False)
      .apply(lambda x: x.sample(1, random_state=None))
      .reset_index(drop=True)
)

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


# In[ ]:


print("Sampled (one row per position):")
print(sampled[["action","position","reward","pscore"]])

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False,   # <- don't share, so each row can show its ticks
    subplot_titles=[f"Position {p}" for p in positions],
    vertical_spacing=0.08
)

for i, pos in enumerate(positions, start=1):
    xs  = np.arange(n_actions)
    y0  = pmf_round0[pos]
    y1  = pmf_round1[pos]
    ymax = float(max(y0.max(), y1.max())) * 1.25

    fig.add_trace(
        go.Bar(
            x=xs, y=y0, name="Round 0 (uniform)",
            marker=dict(color="steelblue"),
            opacity=0.6, legendgroup="round0",
            showlegend=(i == 1),
            customdata=np.column_stack([np.full(n_actions, pos)]),
            hovertemplate="Position: %{customdata[0]}<br>Action: %{x}<br>Round: 0<br>P: %{y:.6f}<extra></extra>",
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Bar(
            x=xs, y=y1, name="Round 1 (updated)",
            marker=dict(color="darkorange"),
            opacity=0.8, legendgroup="round1",
            showlegend=(i == 1),
            customdata=np.column_stack([np.full(n_actions, pos)]),
            hovertemplate="Position: %{customdata[0]}<br>Action: %{x}<br>Round: 1<br>P: %{y:.6f}<extra></extra>",
        ),
        row=i, col=1
    )

    # per-row y range
    fig.update_yaxes(range=[0, ymax], row=i, col=1)

# force ticks 0..79 on EVERY subplot
tick_vals = np.arange(n_actions)
tick_txt  = [str(a) for a in tick_vals]
for i in range(1, 4):
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_txt,
        showticklabels=True,   # <- ensure labels are drawn
        ticks="outside",
        row=i, col=1
    )

# axis titles
fig.update_xaxes(title_text="Action ID (0–79)", row=3, col=1)
for i in range(1, 4):
    fig.update_yaxes(title_text="P(action selected)", row=i, col=1)

fig.update_layout(
    title="Selection probabilities by position: Round 0 vs Round 1",
    barmode="group",
    bargap=0.15,
    height=900,
    template="plotly_white",
)

fig.show()


# In[17]:


get_ipython().system('jupyter nbconvert --to script bandit.ipynb')


# In[ ]:




