import numpy as np
import pandas as pd

def make_probs_df(
    round_num: int,
    alpha: dict[int, np.ndarray],
    beta: dict[int, np.ndarray],
    pmf_by_pos: dict[int, np.ndarray],
    n_actions: int,
    positions: list[int],
) -> pd.DataFrame:
    """
    Build a wide DataFrame for a given round with selection probs and α/β per position.

    Parameters
    ----------
    round_num : int
        Round number identifier.
    alpha, beta : dict[int, np.ndarray]
        Dictionaries mapping position -> alpha/beta vectors for each action.
    pmf_by_pos : dict[int, np.ndarray]
        Dictionaries mapping position -> probability mass function over actions.
    n_actions : int
        Total number of actions.
    positions : list[int]
        List of positions (bandit keys).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: round, action, p_pos{pos}, alpha_pos{pos}, beta_pos{pos}
    """
    rows = []
    for a in range(n_actions):
        row = {"round": round_num, "action": a}
        for pos in positions:
            row[f"p_pos{pos}"]     = pmf_by_pos[pos][a]
            row[f"alpha_pos{pos}"] = alpha[pos][a]
            row[f"beta_pos{pos}"]  = beta[pos][a]
        rows.append(row)
    return pd.DataFrame(rows)

def pmf_history_to_df(tracker):
    """
    Flatten tracker.pmfs (dict: round -> {key -> np.array(n_actions)})
    into a tidy DataFrame with columns:
      round, bandit_key, action, p_select
    """
    recs = []
    for r, pmf_dict in tracker.pmfs.items():
        for key, vec in pmf_dict.items():
            for a, p in enumerate(np.asarray(vec)):
                recs.append({"round": r, "bandit_key": key, "action": a, "p_select": float(p)})
    return pd.DataFrame(recs)

def picks_history_to_df(tracker):
    """
    Flatten tracker.samples (dict: round -> DataFrame of sampled rows)
    into a tidy DataFrame with columns:
      round, bandit_key, action, reward, position, pscore
    For vanilla TS, bandit_key is 'all'.
    For factorized mode, bandit_key equals position.
    """
    recs = []
    for r, df_s in tracker.samples.items():
        if df_s is None or df_s.empty or r == 0:
            continue
        if tracker.factorize:
            for _, row in df_s.iterrows():
                recs.append({
                    "round": r,
                    "bandit_key": int(row["position"]),
                    "position": int(row["position"]),
                    "action": int(row["action"]),
                    "reward": int(row["reward"]),
                    "pscore": float(row["pscore"]),
                })
        else:
            row = df_s.iloc[0]
            recs.append({
                "round": r,
                "bandit_key": "all",
                "position": int(row["position"]),
                "action": int(row["action"]),
                "reward": int(row["reward"]),
                "pscore": float(row["pscore"]),
            })
    return pd.DataFrame(recs)


