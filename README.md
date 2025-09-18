# OBP_Replication

Lightweight repo to replicate the **Open Bandit Dataset & Pipeline (OBD/OBP)** paper and then extend toward multi‑session credit assignment (per my qual proposal).


### Sample import of a random behavior policy using OpenBanditDataset


```python 
# pick a campaign + behavior policy
d = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = d.obtain_batch_bandit_feedback()

# Start with scalar arrays (1D)
df = pd.DataFrame({
    "round_id": range(bf["n_rounds"]),
    "action": bf["action"],       # which item was chosen
    "position": bf["position"],   # slot index (0,1,2)
    "reward": bf["reward"],       # click (0/1)
    "pscore": bf["pscore"],       # logging policy prob
})

df.head()
```


---

## Research Journal

### 2025‑09‑13

* Understand that it was only a 7 day horizon
* I encoded the hash from the features into more readable categories, i.e. `user_feature_{1,2,..}_category_{A,B,C}`
* Cardinality is low for user features and there are only 4. 
* As expected, propensity is the most important feature in predicted clickthrough rate (that's literally what it means as it's the baseline probability)
* A lot of the categories within the user features only have 0 or 1 click so the data is very sparse and confidence intervals very wide. 


### To do Next 
* Compute correlations for each feature with target variable 
* Compute AUC, F1, etc of random forest and try to get a sense of how this compares vs the OBP pipeline and articulate why one is better than the other. 

**What I did**
- Created repo `OBP_Replication`.
- (Initially) made a Python `venv`:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Installed **Miniconda** and initialized conda:
  ```bash
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o Miniconda3.sh
  bash Miniconda3.sh -b -p $HOME/miniconda
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.zshrc
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda init zsh
  deactivate && exec zsh   # leave venv; reload shell so conda hooks load
  ```
- Created and activated a dedicated conda env:
  ```bash
  conda create -n obp_replication python=3.10 -y
  conda activate obp_replication
  ```
- Installed baseline packages (pinning to OBP‑compatible stack):
  ```bash
  pip install --upgrade pip
  pip install obp jupyter notebook matplotlib seaborn scikit-learn pandas
  python -c "import obp; print('OBP:', obp.__version__)"
  ```

**Issue(s) hit**
- Importing `obp` raised:
  ```
  AttributeError: module 'matplotlib.cm' has no attribute 'register_cmap'
  ```
  This is a **version mismatch** between `seaborn` and `matplotlib` (OBP 0.5.7 pins an older seaborn in some places; pip resolved a newer matplotlib).  

**Quick fix (next step)**
- Align plotting deps (either downgrade matplotlib or upgrade seaborn consistently). Options:
  - Safer pin for OBP 0.5.7:
    ```bash
    pip install "matplotlib<3.6" "seaborn==0.11.2"
    ```
  - Or modernize seaborn and check OBP import paths:
    ```bash
    pip install --upgrade "seaborn>=0.12,<0.14"
    ```
- After adjustment, re‑run:
  ```bash
  python -c "import seaborn, matplotlib; print(seaborn.__version__, matplotlib.__version__)"
  python -c "import obp; print('OBP:', obp.__version__)"
  ```

**Repo scaffold created**
```
data/
  raw/
  cache/
notebooks/
scripts/
results/
logs/
```


**Why this matters**
- Having a locked env is prerequisite for reproducing OBP tables (RMSE/REE) and comparing against on‑policy ground truth.

**Progress update (same day)**
- ✅ Successfully imported `obp` after resolving the seaborn/matplotlib version clash.
- ✅ Loaded the **small-sized OBD sample** via:
  ```bash
  python - <<'PY'
  from obp.dataset import OpenBanditDataset
  d = OpenBanditDataset(behavior_policy="random", campaign="all")
  bf = d.obtain_batch_bandit_feedback()
  print("Keys:", sorted(bf.keys()))
  for k in ["n_rounds","action","position","reward","pscore"]:
      v = bf.get(k)
      print(k, getattr(v, "shape", type(v)))
  PY
  ```
  Output matched expectations:
  ```
  Keys: ['action', 'action_context', 'context', 'n_actions', 'n_rounds', 'position', 'pscore', 'reward']
  n_rounds <class 'int'>
  action (10000,)
  position (10000,)
  reward (10000,)
  pscore (10000,)
  ```
- ✅ Saved environment snapshot:
  ```bash
  pip freeze > requirements.txt
  ```

**Note on dataset size**
- When `data_path` is **omitted**, `OpenBanditDataset` downloads a small demo split (~10k rounds) for quick tests.
- To use the **full logs**, download the campaign/policy CSVs and pass a local folder via `data_path=...`, e.g.:
  ```python
  d = OpenBanditDataset(
      behavior_policy="random",
      campaign="all",
      data_path="/path/to/obd_full"  # folder containing extracted OBD CSVs
  )
  ```

---

### 2025‑09‑14
**Goal:** Explore dataset schema, understand feature structure, and replicate basic OPE.  

**What I did**  
- Downloaded full `zr-obp` repo (with Git LFS) and accessed raw CSVs (`log.csv`, `context.csv`, `item_context.csv`).  
- Loaded log data into Pandas, explored schema (columns: `timestamp`, `item_id`, `position`, `click`, `propensity_score`, hashed `user_feature_*`, `user-item-affinity_*`).  
- Verified ~26M rows are from a **7-day A/B test** across 3 campaigns (“All”, “Men’s”, “Women’s”) with Random vs BTS policies.  
- Created readable versions of categorical hashes:  
  - Renamed `user_feature_*` values as `A0, B1, C2, …` for interpretability.  
- Analyzed cardinality of each user feature → found surprisingly low unique values (3–9).  
- Visualized **CTR per category**:  
  - Barplots per feature value (alphabetical order).  
  - Error bar plots with Wilson confidence intervals.  
  - Gridded charts with shared y-axis scales for comparability.  
  - Blue vs gray coloring depending on minimum click counts (≥5 vs <5).  
- Computed **pairwise joint CTR heatmaps** (e.g. `user_feature_1 × user_feature_2`).  
- Built a one-hot encoded feature matrix (`X`) + target (`y`).  
- Fit a **Random Forest**:  
  - Baseline AUC ≈ [fill in].  
  - Found that `propensity_score` dominated importance → concluded it should be dropped as it leaks logging policy information.  
  - Next step: refit RF without `propensity_score` to see which features drive clicks.  

**Why this matters**  
- Confirmed that the raw logs expose the structure OBP estimators rely on (pscore, actions, rewards).  
- Early CTR plots show heterogeneity across hashed user features → suggests real signal to capture.  
- Random Forest sanity check shows that metadata can leak, so careful feature selection is critical before moving to sequence / credit-assignment experiments.  


## How to Reproduce My Env

From project root:
```bash
conda create -n obp_replication python=3.10 -y
conda activate obp_replication
pip install --upgrade pip
pip install obp jupyter notebook "matplotlib<3.6" "seaborn==0.11.2" scikit-learn pandas
```
Optional: save the spec
```bash
conda env export --no-builds > environment.yml
```

---

## Next Actions
1. Fix plotting stack; confirm `import obp` works.
2. Register Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name obp_replication --display-name "Python (obp_replication)"
   ```
3. Notebook `notebooks/00_quickstart.ipynb`: load `OpenBanditDataset` (`random/all`) and cache bandit feedback.
4. Add `scripts/evaluate.py` to run IPS/SNIPS/DM/DR (smoke test).
5. Start a download plan for full OBD; use sample CSVs if servers are slow.

---

## Useful Commands
```bash
# activate / deactivate
conda activate obp_replication
conda deactivate

# check versions
python -c "import obp, seaborn, matplotlib; print(obp.__version__, seaborn.__version__, matplotlib.__version__)"

# jupyter
jupyter notebook
```