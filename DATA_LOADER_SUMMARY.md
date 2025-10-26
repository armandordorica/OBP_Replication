# Data Loader Module - Summary

## 📦 Files Created

### 1. `data_loader.py` - Main Module
**Location**: `/OBP_Replication/data_loader.py`

A comprehensive Python module for loading the Open Bandit Dataset with the following features:

#### **Core Function: `load_data()`**
```python
load_data(behavior_policy, campaign, dataset_type)
```

**Parameters**:
- `behavior_policy`: `'random'` or `'bts'` (Bernoulli Thompson Sampling)
- `campaign`: `'all'`, `'men'`, or `'women'`
- `dataset_type`: `'sample'` (10k via dataloader) or `'full'` (from CSV)

**Returns**: pandas DataFrame with columns:
- `action`: Item/action ID
- `position`: Position in slate (0, 1, or 2)
- `reward`: Binary reward (1=click, 0=no click)
- `pscore`: Propensity score
- Additional columns in full datasets

#### **Helper Functions**:

1. **`compute_ctr(df)`** - Calculate click-through rate
2. **`get_dataset_stats(df)`** - Get comprehensive statistics
3. **`load_all_campaigns(policy, dataset_type)`** - Load all 3 campaigns at once
4. **`load_all_policies(campaign, dataset_type)`** - Load both policies at once

#### **Features**:
- ✅ Parameter validation with helpful error messages
- ✅ Automatic column name standardization
- ✅ Support for both sample and full datasets
- ✅ Informative loading messages
- ✅ File existence checking
- ✅ Comprehensive docstrings
- ✅ Clean, reusable code

---

### 2. `data_loader_README.md` - Documentation
**Location**: `/OBP_Replication/data_loader_README.md`

Complete documentation including:
- Installation instructions
- Quick start guide
- Parameter descriptions
- 9 usage examples
- API reference
- Expected statistics table
- Error handling guide

---

### 3. `data_loader_examples.py` - Example Script
**Location**: `/OBP_Replication/data_loader_examples.py`

Standalone Python script with 9 comprehensive examples:

1. **Basic Loading** - Simple data loading
2. **Compute CTR** - Calculate CTR for multiple configs
3. **Dataset Stats** - Get comprehensive statistics
4. **Compare Policies** - Random vs BTS comparison
5. **Load All Campaigns** - Load all campaigns at once
6. **Load All Policies** - Load both policies at once
7. **Full Dataset** - Load and analyze full dataset
8. **Position Analysis** - Analyze CTR by position
9. **Action Analysis** - Find top performing actions

---

## 🚀 Quick Start

### Import in Jupyter Notebook:
```python
from data_loader import load_data, compute_ctr, get_dataset_stats

# Load sample data
df = load_data('random', 'all', 'sample')

# Compute CTR
ctr = compute_ctr(df)
print(f"CTR: {ctr:.4f} ({ctr*100:.2f}%)")

# Get statistics
stats = get_dataset_stats(df)
print(f"Records: {stats['n_records']:,}")
```

### Load Full Dataset:
```python
# Load full dataset from CSV
df_full = load_data('bts', 'all', 'full')

# Compare policies
df_random = load_data('random', 'all', 'full')
df_bts = load_data('bts', 'all', 'full')

lift = ((compute_ctr(df_bts) / compute_ctr(df_random)) - 1) * 100
print(f"BTS Lift: {lift:.2f}%")
```

---

## 📊 All Supported Combinations

### Behavior Policies:
- `random` - Random policy
- `bts` - Bernoulli Thompson Sampling

### Campaigns:
- `all` - All campaigns combined (80 items, 84 dimensions)
- `men` - Men's fashion (34 items, 38 dimensions)
- `women` - Women's fashion (46 items, 50 dimensions)

### Dataset Types:
- `sample` - 10,000 records (via OpenBanditDataset)
- `full` - Full dataset from CSV files

**Total Combinations**: 2 policies × 3 campaigns × 2 types = **12 configurations**

---

## ✅ Tested & Validated

The module has been tested in the notebook and successfully:

✅ Loads sample datasets for all combinations  
✅ Computes accurate CTR values  
✅ Provides comprehensive statistics  
✅ Loads all campaigns simultaneously  
✅ Loads all policies simultaneously  
✅ Handles missing files gracefully  
✅ Validates parameters  
✅ Standardizes column names  

### Example Output:
```
✅ Loaded 10,000 records from sample dataset
   Policy: RANDOM, Campaign: ALL
   CTR: 0.0038 (0.38%)

✅ Loaded 10,000 records from sample dataset
   Policy: BTS, Campaign: ALL
   CTR: 0.0042 (0.42%)
```

---

## 📁 Expected Directory Structure

For full datasets, the module expects:
```
OBP_Replication/
├── data_loader.py              # Main module
├── data_loader_README.md       # Documentation
├── data_loader_examples.py     # Examples
└── zr-obp/
    └── full_dataset/
        ├── random/
        │   ├── all/all.csv     (1,374,327 records)
        │   ├── men/men.csv     (452,949 records)
        │   └── women/women.csv (864,585 records)
        └── bts/
            ├── all/all.csv     (12,168,084 records)
            ├── men/men.csv     (4,077,727 records)
            └── women/women.csv (7,765,497 records)
```

---

## 📈 Expected Statistics (from Paper)

| Policy | Campaign | #Data      | CTR   | Relative CTR |
|--------|----------|------------|-------|--------------|
| Random | all      | 1,374,327  | 0.35% | 1.00         |
| BTS    | all      | 12,168,084 | 0.50% | 1.43         |
| Random | men      | 452,949    | 0.51% | 1.48         |
| BTS    | men      | 4,077,727  | 0.67% | 1.94         |
| Random | women    | 864,585    | 0.48% | 1.39         |
| BTS    | women    | 7,765,497  | 0.64% | 1.84         |

---

## 🎯 Benefits

### Before (inline code):
```python
# Load random data
ds = OpenBanditDataset(behavior_policy="random", campaign="all")
bf = ds.obtain_batch_bandit_feedback()
df = pd.DataFrame({
    "action": bf["action"],
    "position": bf["position"],
    "reward": bf["reward"]
})
```

### After (module):
```python
df = load_data('random', 'all', 'sample')
```

**Advantages**:
- ✅ 1 line instead of 5+
- ✅ Consistent interface for sample and full datasets
- ✅ Parameter validation
- ✅ Automatic column standardization
- ✅ Reusable across notebooks and scripts
- ✅ Well-documented and tested
- ✅ Error handling included

---

## 🔄 Module Version

- **Version**: 1.0.0
- **Author**: Armando Ordorica
- **Date**: October 26, 2025
- **Python**: 3.7+
- **Dependencies**: pandas, obp (Open Bandit Pipeline)

---

## 📚 References

- **Paper**: [Open Bandit Dataset and Pipeline](https://arxiv.org/abs/2008.07146) (NeurIPS 2021)
- **Repository**: [st-tech/zr-obp](https://github.com/st-tech/zr-obp)
- **Documentation**: [OBP Docs](https://zr-obp.readthedocs.io/)

---

## 🎉 Summary

Successfully created a professional, production-ready data loader module that:

1. ✅ Provides clean API for loading Open Bandit Dataset
2. ✅ Supports all policy × campaign × dataset type combinations
3. ✅ Includes comprehensive documentation
4. ✅ Provides working examples
5. ✅ Tested and validated in notebook
6. ✅ Ready for import in other projects

**Usage is now as simple as**:
```python
from data_loader import load_data
df = load_data('bts', 'men', 'full')
```
