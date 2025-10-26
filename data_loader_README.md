# Open Bandit Dataset Data Loader Module

A unified Python module for loading the Open Bandit Dataset with support for both sample and full datasets.

## Installation

No installation required. Simply import the module from the same directory:

```python
from data_loader import load_data, compute_ctr, get_dataset_stats
```

## Dependencies

- `pandas`
- `obp` (Open Bandit Pipeline)

Install dependencies:
```bash
pip install pandas obp
```

## Quick Start

### Basic Usage

```python
from data_loader import load_data

# Load sample dataset (10k records)
df = load_data(behavior_policy='random', campaign='all', dataset_type='sample')

# Load full dataset from CSV
df_full = load_data(behavior_policy='bts', campaign='men', dataset_type='full')
```

### Parameters

#### `load_data(behavior_policy, campaign, dataset_type)`

**behavior_policy** (str):
- `'random'`: Random policy
- `'bts'`: Bernoulli Thompson Sampling policy

**campaign** (str):
- `'all'`: All campaigns combined
- `'men'`: Men's fashion campaign
- `'women'`: Women's fashion campaign

**dataset_type** (str):
- `'sample'`: Small sample dataset (10k records per policy/campaign) via OpenBanditDataset
- `'full'`: Full dataset loaded from CSV files

### Returns

A pandas DataFrame with columns:
- `action`: Item/action ID
- `position`: Position in the slate (0, 1, or 2)
- `reward`: Binary reward (1 = click, 0 = no click)
- `pscore`: Propensity score (probability of action selection)
- Additional columns in full datasets

## Examples

### Example 1: Load and Compute CTR

```python
from data_loader import load_data, compute_ctr

# Load data
df = load_data('random', 'all', 'sample')

# Compute CTR
ctr = compute_ctr(df)
print(f"CTR: {ctr:.4f} ({ctr*100:.2f}%)")
```

### Example 2: Get Dataset Statistics

```python
from data_loader import load_data, get_dataset_stats

df = load_data('bts', 'all', 'full')
stats = get_dataset_stats(df)

print(f"Records: {stats['n_records']:,}")
print(f"Clicks: {stats['n_clicks']:,}")
print(f"CTR: {stats['ctr']:.4f}")
print(f"Unique Actions: {stats['n_unique_actions']}")
```

### Example 3: Load All Campaigns

```python
from data_loader import load_all_campaigns

# Load all campaigns for random policy
campaigns = load_all_campaigns('random', 'sample')

for name, df in campaigns.items():
    if df is not None:
        print(f"{name.upper()}: {len(df):,} records")
```

### Example 4: Load All Policies

```python
from data_loader import load_all_policies

# Load both policies for 'all' campaign
policies = load_all_policies('all', 'full')

for name, df in policies.items():
    if df is not None:
        ctr = compute_ctr(df)
        print(f"{name.upper()}: CTR = {ctr:.4f}")
```

### Example 5: Compare Policies

```python
from data_loader import load_data, compute_ctr

# Load both policies
df_random = load_data('random', 'all', 'full')
df_bts = load_data('bts', 'all', 'full')

# Compare CTRs
ctr_random = compute_ctr(df_random)
ctr_bts = compute_ctr(df_bts)

lift = ((ctr_bts / ctr_random) - 1) * 100

print(f"Random CTR: {ctr_random:.4f}")
print(f"BTS CTR: {ctr_bts:.4f}")
print(f"Lift: {lift:.2f}%")
```

## Dataset Structure

### Sample Dataset
- Downloaded automatically via `OpenBanditDataset`
- 10,000 records per policy/campaign combination
- Stratified sample from full dataset

### Full Dataset
Expected directory structure:
```
zr-obp/
  full_dataset/
    random/
      all/
        all.csv
      men/
        men.csv
      women/
        women.csv
    bts/
      all/
        all.csv
      men/
        men.csv
      women/
        women.csv
```

## Expected Statistics (Table 1 from Paper)

| Policy | Campaign | #Data | CTR | Relative CTR |
|--------|----------|-------|-----|--------------|
| Random | all | 1,374,327 | 0.35% | 1.00 |
| BTS | all | 12,168,084 | 0.50% | 1.43 |
| Random | men | 452,949 | 0.51% | 1.48 |
| BTS | men | 4,077,727 | 0.67% | 1.94 |
| Random | women | 864,585 | 0.48% | 1.39 |
| BTS | women | 7,765,497 | 0.64% | 1.84 |

## API Reference

### Functions

#### `load_data(behavior_policy, campaign, dataset_type)`
Load dataset with specified parameters.

#### `compute_ctr(df)`
Compute Click-Through Rate from DataFrame.

#### `get_dataset_stats(df)`
Get comprehensive statistics for a dataset.

#### `load_all_campaigns(behavior_policy, dataset_type)`
Load all three campaigns for a given policy.

#### `load_all_policies(campaign, dataset_type)`
Load both policies for a given campaign.

## Error Handling

```python
from data_loader import load_data

try:
    df = load_data('bts', 'all', 'full')
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

## Module Info

- **Version**: 1.0.0
- **Author**: Armando Ordorica
- **Date**: October 26, 2025

## References

- **Paper**: [Open Bandit Dataset and Pipeline (NeurIPS 2021)](https://arxiv.org/abs/2008.07146)
- **Code**: [Open Bandit Pipeline GitHub](https://github.com/st-tech/zr-obp)

## License

This module is part of a research replication project for the Open Bandit Dataset paper.
