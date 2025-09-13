# OBP_Replication

Lightweight repo to replicate the **Open Bandit Dataset & Pipeline (OBD/OBP)** paper and then extend toward multi‑session credit assignment (per my qual proposal).

---

## Research Journal

### 2025‑09‑13
**Goal:** Stand up a clean, reproducible environment and smoke‑test `obp`.

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

---

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