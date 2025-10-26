# Data Loader Module - Quick Reference

## Import
```python
from data_loader import load_data, compute_ctr, get_dataset_stats
```

## Core Function
```python
load_data(behavior_policy, campaign, dataset_type)
```

## Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `behavior_policy` | `'random'`, `'bts'` | Data collection policy |
| `campaign` | `'all'`, `'men'`, `'women'` | Fashion campaign type |
| `dataset_type` | `'sample'`, `'full'` | 10k sample or full CSV |

## Quick Examples

### Load Sample Data
```python
df = load_data('random', 'all', 'sample')
```

### Load Full Dataset
```python
df = load_data('bts', 'men', 'full')
```

### Compute CTR
```python
ctr = compute_ctr(df)
print(f"CTR: {ctr:.4f} ({ctr*100:.2f}%)")
```

### Get Statistics
```python
stats = get_dataset_stats(df)
print(f"Records: {stats['n_records']:,}")
print(f"CTR: {stats['ctr']:.4f}")
```

### Load All Campaigns
```python
campaigns = load_all_campaigns('random', 'sample')
# Returns: {'all': df1, 'men': df2, 'women': df3}
```

### Load All Policies
```python
policies = load_all_policies('all', 'full')
# Returns: {'random': df1, 'bts': df2}
```

## All 12 Combinations

| # | Policy | Campaign | Type | Expected Records |
|---|--------|----------|------|-----------------|
| 1 | random | all | sample | 10,000 |
| 2 | random | all | full | 1,374,327 |
| 3 | random | men | sample | 10,000 |
| 4 | random | men | full | 452,949 |
| 5 | random | women | sample | 10,000 |
| 6 | random | women | full | 864,585 |
| 7 | bts | all | sample | 10,000 |
| 8 | bts | all | full | 12,168,084 |
| 9 | bts | men | sample | 10,000 |
| 10 | bts | men | full | 4,077,727 |
| 11 | bts | women | sample | 10,000 |
| 12 | bts | women | full | 7,765,497 |

## DataFrame Columns

All returned DataFrames have these columns:

- `action` - Item/action ID
- `position` - Position in slate (0, 1, or 2)
- `reward` - Binary (1=click, 0=no click)
- `pscore` - Propensity score

Full datasets may have additional columns (context features, timestamps, etc.)

## Common Patterns

### Compare Policies
```python
df_random = load_data('random', 'all', 'full')
df_bts = load_data('bts', 'all', 'full')

ctr_random = compute_ctr(df_random)
ctr_bts = compute_ctr(df_bts)

lift = ((ctr_bts / ctr_random) - 1) * 100
print(f"Lift: {lift:.2f}%")
```

### Analyze by Position
```python
df = load_data('bts', 'all', 'sample')

for pos in sorted(df['position'].unique()):
    pos_df = df[df['position'] == pos]
    ctr = compute_ctr(pos_df)
    print(f"Position {pos}: CTR = {ctr:.4f}")
```

### Top Actions
```python
df = load_data('random', 'all', 'sample')

action_ctr = df.groupby('action')['reward'].mean()
top_5 = action_ctr.nlargest(5)

for action_id, ctr in top_5.items():
    print(f"Action {action_id}: CTR = {ctr:.4f}")
```

## Error Handling
```python
try:
    df = load_data('bts', 'all', 'full')
except FileNotFoundError:
    print("Full dataset not found, using sample instead")
    df = load_data('bts', 'all', 'sample')
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

## Expected CTR Values (from Paper)

| Policy | Campaign | CTR | Relative CTR |
|--------|----------|-----|--------------|
| random | all | 0.35% | 1.00 |
| bts | all | 0.50% | 1.43 |
| random | men | 0.51% | 1.48 |
| bts | men | 0.67% | 1.94 |
| random | women | 0.48% | 1.39 |
| bts | women | 0.64% | 1.84 |

---

**Module Version**: 1.0.0  
**Author**: Armando Ordorica  
**Date**: October 26, 2025  

**Files**:
- `data_loader.py` - Module
- `data_loader_README.md` - Full documentation
- `data_loader_examples.py` - 9 examples
- `DATA_LOADER_SUMMARY.md` - Detailed summary
- `DATA_LOADER_QUICKREF.md` - This quick reference
