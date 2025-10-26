"""
Statistical and propensity analysis utilities for offline bandit evaluation.

This module contains functions for computing propensity scores, distributions,
and statistical summaries from bandit log data.
"""

import pandas as pd
import numpy as np
from itertools import product


def compute_item_feature_distribution(df, item_id, feature_col, item_col='item_id'):
    """
    Compute the distribution of a feature for a specific item.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe with the log data
    item_id : int
        The item ID to filter for
    feature_col : str
        The feature column to analyze (e.g., 'user_feature_0')
    item_col : str, optional
        The name of the item column (default: 'item_id')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: feature_col, count, total_occurrences, proportion
    """
    result = df[df[item_col] == item_id].groupby(feature_col).size().reset_index(name='count').sort_values(by='count', ascending=False)
    result['total_occurrences'] = result['count'].sum()
    result['proportion'] = result['count'] / result['total_occurrences']
    return result


def compute_manual_propensity(df, item_col='item_id', categorical_col='user_feature_0'):
    """
    Compute manual propensity scores based on item and a categorical feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the bandit log data
    item_col : str, optional
        The name of the item column (default: 'item_id')
    categorical_col : str
        The name of the categorical feature column (e.g., 'user_feature_0', 'position')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - item_col: The item IDs
        - categorical_col: The categorical feature values
        - count_of_occurrences: Number of occurrences for each (item, feature) pair
        - total: Total count across all pairs (same for all rows)
        - manual_propensity: count_of_occurrences / total
    
    Example:
    --------
    >>> result = compute_manual_propensity(log_df_readable, categorical_col='user_feature_0')
    >>> print(result.head())
       item_id user_feature_0  count_of_occurrences  total  manual_propensity
    0        0             A0                    94  10000             0.0094
    1        0             B0                    31  10000             0.0031
    """
    # Group by item and categorical feature, count occurrences
    result = df.groupby([item_col, categorical_col]).size().reset_index()
    result.columns = [item_col, categorical_col, 'count_of_occurrences']
    
    # Add total and manual propensity
    result['total'] = result['count_of_occurrences'].sum()
    result['manual_propensity'] = result['count_of_occurrences'] / result['total']
    
    return result


def compute_item_propensity_stats(df, item_id, item_col='item_id', propensity_col='propensity_score'):
    """
    Compute propensity statistics for a specific item.
    
    This function calculates:
    - count_of_occurrences: How many times the item appears in the log
    - total: Total number of records in the entire dataset
    - manual_propensity: count_of_occurrences / total (empirical probability)
    - mean_propensity_score: Average of the logged propensity scores for this item
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the bandit log data
    item_id : int or str
        The specific item ID to analyze
    item_col : str, optional
        The name of the item column (default: 'item_id')
    propensity_col : str, optional
        The name of the propensity score column (default: 'propensity_score')
    
    Returns:
    --------
    pd.DataFrame
        Single-row DataFrame with columns:
        - item_id: The item ID
        - count_of_occurrences: Number of times item appears
        - total: Total records in dataset
        - manual_propensity: Empirical probability (count/total)
        - mean_propensity_score: Average logged propensity score for this item
    
    Example:
    --------
    >>> stats = compute_item_propensity_stats(log_df_readable, item_id=0)
    >>> print(stats)
       item_id  count_of_occurrences  total  manual_propensity  mean_propensity_score
    0        0                   125  10000             0.0125               0.012500
    """
    # Compute count of occurrences for all items
    counts_df = df.groupby([item_col]).size().to_frame()
    counts_df.columns = ['count_of_occurrences']
    counts_df['total'] = counts_df['count_of_occurrences'].sum()
    counts_df['manual_propensity'] = counts_df['count_of_occurrences'] / counts_df['total']
    
    # Filter for the specific item
    item_counts = counts_df[counts_df.index == item_id].reset_index()
    
    # Compute mean propensity score from logged values
    mean_propensity = df.groupby([item_col])[propensity_col].mean().reset_index()
    mean_propensity.columns = [item_col, 'mean_propensity_score']
    mean_propensity_item = mean_propensity[mean_propensity[item_col] == item_id]
    
    # Merge the results
    result = pd.merge(item_counts, mean_propensity_item, on=item_col)
    
    return result


def calculate_distribution_stats(df, value_col, groupby_col, groupby_col_as_string=True):
    """
    Calculate comprehensive distribution statistics for a value column grouped by another column.
    
    This function computes summary statistics (mean, std, min, max, quartiles, p99) for each
    unique value in the groupby column and prepares the data for plotting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
    value_col : str
        The column containing values to analyze (e.g., 'propensity_score')
    groupby_col : str
        The column to group by (e.g., 'item_id', 'position')
    groupby_col_as_string : bool, optional
        Whether to create a string version of the groupby column for categorical plotting (default: True)
    
    Returns:
    --------
    tuple: (plot_df, stats_df)
        - plot_df: DataFrame with all individual values and a string version of groupby_col
        - stats_df: DataFrame with summary statistics (mean, std, min, max, p25, p50, p75, p99) per group
    
    Example:
    --------
    >>> plot_df, stats_df = calculate_distribution_stats(log_df, 'propensity_score', 'item_id')
    >>> print(stats_df.head())
       item_id  mean      std       min       max       p25       p50       p75       p99
    0        0  0.05    0.02      0.01      0.10      0.03      0.05      0.07      0.09
    """
    # Create a copy with string version of groupby column for categorical plotting
    groupby_col_str = f"{groupby_col}_str"
    plot_df = df.copy()
    
    if groupby_col_as_string:
        plot_df[groupby_col_str] = plot_df[groupby_col].astype(str)
    else:
        groupby_col_str = groupby_col
    
    # Calculate summary statistics
    stats_df = df.groupby(groupby_col)[value_col].agg([
        'mean', 'std', 'min', 'max',
        ('p25', lambda x: x.quantile(0.25)),
        ('p50', lambda x: x.quantile(0.50)),
        ('p75', lambda x: x.quantile(0.75)),
        ('p99', lambda x: x.quantile(0.99))
    ]).reset_index()
    
    # Add string version to stats_df
    if groupby_col_as_string:
        stats_df[groupby_col_str] = stats_df[groupby_col].astype(str)
    
    return plot_df, stats_df


def compute_propensity_variance(df, groupby_col):
    """
    Compute the variance of propensity scores when grouped by a specific column.
    
    This function calculates how much of the variance in propensity scores
    can be explained by grouping on a particular variable (e.g., item_id, position).
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing propensity scores
    groupby_col : str
        The column to group by (e.g., 'item_id', 'position', 'user_feature_0')
    
    Returns:
    --------
    float
        The variance of the mean propensity scores across groups
    
    Example:
    --------
    >>> var_item = compute_propensity_variance(log_df, 'item_id')
    >>> var_position = compute_propensity_variance(log_df, 'position')
    >>> print(f"Variance by item_id: {var_item:.6f}")
    >>> print(f"Variance by position: {var_position:.6f}")
    """
    # Calculate mean propensity score for each group
    group_means = df.groupby(groupby_col)['propensity_score'].mean()
    
    # Calculate variance of these means
    variance = group_means.var()
    
    return variance


def compute_feature_combinations(df, feature_cols, verbose=True):
    """
    Compute unique values for each feature column and calculate total possible combinations.
    
    This function analyzes categorical feature columns to determine:
    1. The unique values present in each feature column
    2. The total number of possible feature combinations (Cartesian product)
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the feature columns
    feature_cols : list of str
        List of feature column names to analyze (e.g., ['user_feature_0', 'user_feature_1'])
    verbose : bool, optional
        Whether to print detailed output (default: True)
    
    Returns:
    --------
    tuple: (unique_values, total_combinations)
        - unique_values: dict mapping column names to sorted lists of unique values
        - total_combinations: int representing the total number of possible combinations
    
    Example:
    --------
    >>> feature_cols = ['user_feature_0', 'user_feature_1', 'user_feature_2', 'user_feature_3']
    >>> unique_vals, total_combos = compute_feature_combinations(log_df, feature_cols)
    >>> print(f"Total possible combinations: {total_combos:,}")
    """
    # Get unique values for each feature
    unique_values = {}
    for col in feature_cols:
        unique_values[col] = sorted(df[col].dropna().unique())
    
    if verbose:
        print("Unique values per feature:")
        for col, vals in unique_values.items():
            print(f"  {col}: {vals} (n={len(vals)})")
    
    # Calculate total possible combinations (product of all cardinalities)
    total_combinations = 1
    for col, vals in unique_values.items():
        total_combinations *= len(vals)
    
    if verbose:
        print(f"\nTotal possible feature combinations: {total_combinations:,}")
    
    return unique_values, total_combinations


def find_missing_combinations(df, feature_cols, item_id, item_col='item_id', verbose=False):
    """
    Find feature combinations that are missing (not present) in the dataset for a specific item.
    
    This function determines which combinations of feature values do not appear in the data
    for the specified item_id. It's useful for understanding data sparsity and coverage gaps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the bandit log data
    feature_cols : list of str
        List of column names to analyze for combinations
    item_id : int or str
        The item ID to filter for
    item_col : str, optional
        The name of the item column (default: 'item_id')
    verbose : bool, optional
        If True, prints summary statistics (default: False)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all missing feature combinations, with one column per feature.
        Empty if all combinations are present.
    
    Example:
    --------
    >>> feature_cols = ['user_feature_0', 'user_feature_1', 'position']
    >>> missing_df = find_missing_combinations(log_df, feature_cols, item_id=0)
    >>> print(f"Missing combinations for item 0: {len(missing_df)}")
    """
    # Filter for the specific item
    item_df = df[df[item_col] == item_id]
    
    if verbose:
        print(f"Analyzing item_id={item_id}")
        print(f"Total records for this item: {len(item_df)}")
    
    # Get unique values for each feature
    unique_values = {}
    for col in feature_cols:
        unique_values[col] = sorted(df[col].dropna().unique())
    
    # Generate all possible combinations
    all_combinations = list(product(*[unique_values[col] for col in feature_cols]))
    total_possible = len(all_combinations)
    
    if verbose:
        print(f"Total possible combinations: {total_possible:,}")
    
    # Get existing combinations for this item
    existing_combinations = set(
        tuple(row) for row in item_df[feature_cols].drop_duplicates().values
    )
    
    if verbose:
        print(f"Existing combinations for item {item_id}: {len(existing_combinations)}")
    
    # Find missing combinations
    missing_combinations = [
        combo for combo in all_combinations 
        if combo not in existing_combinations
    ]
    
    # Convert to DataFrame
    missing_df = pd.DataFrame(missing_combinations, columns=feature_cols)
    
    if verbose:
        print(f"Missing combinations: {len(missing_df)}")
        print(f"Coverage: {len(existing_combinations)/total_possible*100:.2f}%")
    
    return missing_df
