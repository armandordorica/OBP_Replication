# OBP Utils - Open Bandit Pipeline Utilities

A comprehensive Python package containing utility modules for working with the Open Bandit Dataset.

## 📦 Package Structure

```
obp_utils/
├── __init__.py              # Package initialization
├── data_loader.py          # Data loading utilities
├── stats.py                # Statistical analysis functions
├── visualizations.py       # Plotting and visualization tools
└── README.md              # This file
```

## 🚀 Installation

Simply add the `obp_utils` folder to your Python path or import from the same directory:

```python
# Option 1: Import from package
from obp_utils import load_data, compute_ctr, plot_histogram_with_stats

# Option 2: Import specific modules
from obp_utils.data_loader import load_data
from obp_utils.stats import compute_manual_propensity
from obp_utils.visualizations import plot_bar_chart
```

## 📚 Modules

### 1. `data_loader.py` - Data Loading Interface

Unified interface for loading Open Bandit Dataset from multiple sources.

**Functions:**
- `load_data(behavior_policy, campaign, dataset_type)` - Load dataset
- `compute_ctr(df)` - Compute click-through rate
- `get_dataset_stats(df)` - Get comprehensive statistics
- `load_all_campaigns(policy, dataset_type)` - Load all campaigns
- `load_all_policies(campaign, dataset_type)` - Load all policies

**Example:**
```python
from obp_utils import load_data, compute_ctr

# Load full BTS dataset
df = load_data('bts', 'all', 'full')

# Compute CTR
ctr = compute_ctr(df)
print(f"CTR: {ctr:.4f}")
```

**Supported Combinations:**
- **Policies**: `random`, `bts`
- **Campaigns**: `all`, `men`, `women`
- **Dataset Types**: `sample` (10k), `full` (CSV)

---

### 2. `stats.py` - Statistical Analysis

Functions for computing statistics and distributions from the OBD dataset.

**Functions:**
- `calculate_distribution_stats(data)` - Compute distribution statistics
- `compute_feature_combinations(df, feature_cols)` - Get all feature combinations
- `compute_item_feature_distribution(df, item_id, feature_col)` - Item-feature distribution
- `compute_item_propensity_stats(df, item_id)` - Propensity score statistics
- `compute_manual_propensity(df, categorical_col)` - Manual propensity computation
- `compute_propensity_variance(df)` - Propensity score variance

**Example:**
```python
from obp_utils import load_data
from obp_utils.stats import compute_manual_propensity, compute_item_propensity_stats

df = load_data('random', 'all', 'sample')

# Compute manual propensities
manual_prop = compute_manual_propensity(df, 'user_feature_0')

# Get propensity stats for item 0
item_stats = compute_item_propensity_stats(df, item_id=0)
```

---

### 3. `visualizations.py` - Plotting Utilities

Functions for creating informative visualizations of OBD data.

**Functions:**
- `plot_bar_chart(data, x, y, title)` - Create bar charts
- `plot_boxplot_with_stats(data, column, title)` - Boxplot with statistics
- `plot_distribution_with_cdf(data, column, title)` - Distribution + CDF
- `plot_histogram_with_stats(data, column, title)` - Histogram with stats

**Example:**
```python
from obp_utils import load_data
from obp_utils.visualizations import plot_histogram_with_stats

df = load_data('bts', 'all', 'full')

# Plot propensity score distribution
plot_histogram_with_stats(
    df, 
    'propensity_score',
    'Propensity Score Distribution'
)
```

---

## 🎯 Quick Start Examples

### Example 1: Load Data and Compute Statistics
```python
from obp_utils import load_data, get_dataset_stats

# Load data
df = load_data('random', 'all', 'sample')

# Get comprehensive stats
stats = get_dataset_stats(df)

print(f"Records: {stats['n_records']:,}")
print(f"CTR: {stats['ctr']:.4f}")
print(f"Unique Actions: {stats['n_unique_actions']}")
```

### Example 2: Compare Policies
```python
from obp_utils import load_all_policies, compute_ctr

# Load both policies
policies = load_all_policies('all', 'full')

# Compare CTRs
for policy, df in policies.items():
    if df is not None:
        ctr = compute_ctr(df)
        print(f"{policy.upper()}: CTR = {ctr:.4f}")
```

### Example 3: Feature Analysis
```python
from obp_utils import load_data
from obp_utils.stats import compute_feature_combinations, compute_manual_propensity

df = load_data('bts', 'men', 'sample')

# Get feature combinations
user_features = [c for c in df.columns if c.startswith('user_feature_')]
unique_vals, total_combos = compute_feature_combinations(df, user_features)

print(f"Total possible combinations: {total_combos:,}")

# Compute propensities by feature
prop_df = compute_manual_propensity(df, 'user_feature_0')
print(prop_df.head())
```

### Example 4: Visualization Pipeline
```python
from obp_utils import load_data
from obp_utils.visualizations import (
    plot_histogram_with_stats,
    plot_boxplot_with_stats
)

df = load_data('random', 'all', 'sample')

# Visualize propensity scores
plot_histogram_with_stats(df, 'propensity_score', 'Propensity Scores')
plot_boxplot_with_stats(df, 'propensity_score', 'Propensity Score Distribution')
```

---

## 📊 Expected Dataset Statistics

From the Open Bandit Dataset paper (Table 1):

| Policy | Campaign | #Data | CTR | Relative CTR |
|--------|----------|-------|-----|--------------|
| Random | all | 1,374,327 | 0.35% | 1.00 |
| BTS | all | 12,168,084 | 0.50% | 1.43 |
| Random | men | 452,949 | 0.51% | 1.48 |
| BTS | men | 4,077,727 | 0.67% | 1.94 |
| Random | women | 864,585 | 0.48% | 1.39 |
| BTS | women | 7,765,497 | 0.64% | 1.84 |

---

## 🔧 Dependencies

```
pandas >= 1.0.0
numpy >= 1.18.0
matplotlib >= 3.0.0
seaborn >= 0.11.0
plotly >= 4.14.0
obp >= 0.5.0  # Open Bandit Pipeline
scikit-learn >= 0.23.0
```

---

## 📁 Usage in Notebooks

### From Parent Directory
```python
from obp_utils import load_data, compute_ctr
```

### From Subdirectory (e.g., zr-obp/)
```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from obp_utils import load_data, compute_ctr
```

---

## 🎨 Module Organization

The package is organized into three main areas:

1. **Data Management** (`data_loader.py`)
   - Loading datasets
   - Computing basic metrics
   - Bulk operations

2. **Analysis** (`stats.py`)
   - Statistical computations
   - Feature analysis
   - Propensity score calculations

3. **Visualization** (`visualizations.py`)
   - Plotting functions
   - Visual analysis tools
   - Interactive charts

---

## 🤝 Contributing

This is a research utility package. To add new functions:

1. Add the function to the appropriate module
2. Update the module's docstrings
3. Export it in `__init__.py`
4. Update this README

---

## 📄 License

This package is part of a research replication project for the Open Bandit Dataset paper.

---

## 📚 References

- **Paper**: [Open Bandit Dataset and Pipeline](https://arxiv.org/abs/2008.07146) (NeurIPS 2021)
- **Code**: [Open Bandit Pipeline GitHub](https://github.com/st-tech/zr-obp)
- **Documentation**: [OBP Docs](https://zr-obp.readthedocs.io/)

---

## 📝 Version History

**v1.0.0** (October 26, 2025)
- Initial release
- `data_loader.py`: Unified data loading
- `stats.py`: Statistical analysis functions  
- `visualizations.py`: Plotting utilities
- Package structure with `__init__.py`

---

## 💡 Tips

1. **Import at package level** for cleaner code:
   ```python
   from obp_utils import load_data  # Good
   from obp_utils.data_loader import load_data  # Also fine
   ```

2. **Use bulk loaders** for efficiency:
   ```python
   # Instead of multiple calls
   campaigns = load_all_campaigns('random', 'full')
   ```

3. **Check function docstrings** for detailed usage:
   ```python
   help(load_data)
   help(compute_manual_propensity)
   ```

---

**Author**: Armando Ordorica  
**Version**: 1.0.0  
**Date**: October 26, 2025
