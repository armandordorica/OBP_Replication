#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from obp.dataset import OpenBanditDataset
import numpy as np
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()
print(bf.keys())  # expect: n_rounds, n_actions, action, position, reward, pscore, context, action_context


# # Basic shape sanity

# In[ ]:


n = bf["n_rounds"]
print("n_rounds:", n, "| n_actions:", bf["n_actions"], "| len_list:", ds.len_list)

for k in ["action", "position", "reward", "pscore"]:
    arr = bf[k]
    print(f"{k}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', type(arr))}")

# --- Assertions (fail fast if something’s off) ---
assert isinstance(n, int) and n > 0
for k in ["action", "position", "reward", "pscore"]:
    assert len(bf[k]) == n, f"{k} length != n_rounds"

assert bf["pscore"].min() > 0, "Found zero/negative propensities"
assert set(np.unique(bf["position"])) <= {0,1,2}, "Positions must be 0-based {0,1,2}"
assert ds.len_list == 3, "OBD should have 3 slots"


# In[22]:


# Contexts can be None in some versions; print if present
ctx = bf.get("context", None)
actx = bf.get("action_context", None)
print("context:", None if ctx is None else ctx.shape)
print("action_context:", None if actx is None else actx.shape)


# In[12]:


df = pd.DataFrame({
    "action": bf["action"],
    "position": bf["position"],
    "reward": bf["reward"],
    "pscore": bf["pscore"],
})
df.head()


# In[20]:


print("\nHead:\n", df.head())
print("DF shape:", df.shape)
print("CTR (overall):", df["reward"].mean())
print("CTR by slot (0-based):\n", df.groupby("position")["reward"].mean())


df.shape


# In[14]:


bf.keys()


# ### Making Sure the CTRs match using CSV vs OpenBanditDataset class 

# In[15]:


# CSV (Random/all)
csv = pd.read_csv("../zr-obp/obd/random/all/all.csv", index_col=0)
csv["pos0"] = csv["position"] - 1
ctr_csv = csv.groupby("pos0")["click"].mean().rename("ctr_csv")

# OBP loader (Random/all)
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()
ctr_bf = (pd.DataFrame({"pos0": bf["position"], "click": bf["reward"]})
          .groupby("pos0")["click"].mean().rename("ctr_bf"))

print(pd.concat([ctr_csv, ctr_bf], axis=1))


# In[ ]:




