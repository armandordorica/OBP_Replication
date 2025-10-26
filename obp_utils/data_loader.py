"""
Open Bandit Dataset Data Loader Module

This module provides a unified interface to load Open Bandit Dataset from either:
- Sample datasets (via OpenBanditDataset dataloader)
- Full datasets (from CSV files)

Author: Armando Ordorica
Date: October 26, 2025
"""

import os
import pandas as pd
from obp.dataset import OpenBanditDataset


def load_data(behavior_policy="random", campaign="all", dataset_type="sample"):
    """
    Unified data loader that handles both sample (dataloader) and full (CSV) datasets.
    
    Args:
        behavior_policy (str): Policy used for data collection
            - 'random': Random policy
            - 'bts': Bernoulli Thompson Sampling policy
        campaign (str): Campaign type
            - 'all': All campaigns combined
            - 'men': Men's fashion campaign
            - 'women': Women's fashion campaign
        dataset_type (str): Type of dataset to load
            - 'sample': Uses OpenBanditDataset (10k records per policy/campaign)
            - 'full': Uses CSV files (full dataset from disk)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - action: Item/action ID
            - position: Position in the slate (0, 1, or 2)
            - reward: Binary reward (1 = click, 0 = no click)
            - pscore: Propensity score (probability of action selection)
            - Additional columns may be present in full datasets
    
    Raises:
        FileNotFoundError: If dataset_type='full' and CSV file not found
        ValueError: If invalid parameter values provided
    
    Examples:
        >>> # Load sample dataset with random policy
        >>> df = load_data('random', 'all', 'sample')
        
        >>> # Load full BTS dataset for men's campaign
        >>> df = load_data('bts', 'men', 'full')
        
        >>> # Load women's campaign with random policy (sample)
        >>> df = load_data('random', 'women', 'sample')
    
    References:
        Paper: "Open Bandit Dataset and Pipeline" (NeurIPS 2021)
        URL: https://arxiv.org/abs/2008.07146
    """
    # Validate parameters
    valid_policies = ['random', 'bts']
    valid_campaigns = ['all', 'men', 'women']
    valid_dataset_types = ['sample', 'full']
    
    if behavior_policy not in valid_policies:
        raise ValueError(
            f"behavior_policy must be one of {valid_policies}, got: {behavior_policy}"
        )
    
    if campaign not in valid_campaigns:
        raise ValueError(
            f"campaign must be one of {valid_campaigns}, got: {campaign}"
        )
    
    if dataset_type not in valid_dataset_types:
        raise ValueError(
            f"dataset_type must be one of {valid_dataset_types}, got: {dataset_type}"
        )
    
    if dataset_type == "sample":
        # Use OpenBanditDataset for sample data
        ds = OpenBanditDataset(behavior_policy=behavior_policy, campaign=campaign)
        bf = ds.obtain_batch_bandit_feedback()
        
        # Create DataFrame with core columns
        df = pd.DataFrame({
            "action": bf["action"],
            "position": bf["position"],
            "reward": bf["reward"]
        })
        
        # Add propensity score if available
        if "pscore" in bf:
            df["pscore"] = bf["pscore"]
            
    elif dataset_type == "full":
        # Load from CSV files
        base_path = "zr-obp/full_dataset"
        csv_path = os.path.join(base_path, behavior_policy, campaign, f"{campaign}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Full dataset not found at: {csv_path}\n"
                f"Expected structure: zr-obp/full_dataset/{{policy}}/{{campaign}}/{{campaign}}.csv\n"
                f"Please ensure the full dataset is downloaded and extracted to the correct location."
            )
        
        # Load CSV
        df = pd.read_csv(csv_path, index_col=0)
        
        # Standardize column names to match sample dataset format
        column_mapping = {
            'item_id': 'action',
            'click': 'reward',
            'propensity_score': 'pscore'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['action', 'position', 'reward']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
    
    # Print loading confirmation
    print(f"✅ Loaded {len(df):,} records from {dataset_type} dataset")
    print(f"   Policy: {behavior_policy.upper()}, Campaign: {campaign.upper()}")
    
    return df


def compute_ctr(df):
    """
    Compute Click-Through Rate (CTR) from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with 'reward' column (binary: 1=click, 0=no click)
    
    Returns:
        float: CTR as a decimal (e.g., 0.035 for 3.5%)
    
    Examples:
        >>> df = load_data('random', 'all', 'sample')
        >>> ctr = compute_ctr(df)
        >>> print(f"CTR: {ctr:.4f} ({ctr*100:.2f}%)")
    """
    if 'reward' not in df.columns:
        raise ValueError("DataFrame must contain 'reward' column to compute CTR")
    
    return df['reward'].mean()


def get_dataset_stats(df):
    """
    Get comprehensive statistics for a loaded dataset.
    
    Args:
        df (pd.DataFrame): DataFrame returned by load_data()
    
    Returns:
        dict: Dictionary containing dataset statistics:
            - n_records: Total number of records
            - n_clicks: Total number of clicks
            - n_impressions: Total impressions (same as n_records)
            - ctr: Overall click-through rate
            - n_unique_actions: Number of unique actions/items
            - n_positions: Number of unique positions
            - actions_per_position: Average actions per position
    
    Examples:
        >>> df = load_data('bts', 'all', 'full')
        >>> stats = get_dataset_stats(df)
        >>> print(f"CTR: {stats['ctr']:.4f}")
    """
    stats = {
        'n_records': len(df),
        'n_clicks': int(df['reward'].sum()),
        'n_impressions': len(df),
        'ctr': df['reward'].mean(),
        'n_unique_actions': df['action'].nunique(),
        'n_positions': df['position'].nunique(),
    }
    
    # Compute actions per position
    if stats['n_positions'] > 0:
        stats['actions_per_position'] = stats['n_records'] / stats['n_positions']
    else:
        stats['actions_per_position'] = 0
    
    return stats


def load_all_campaigns(behavior_policy="random", dataset_type="sample"):
    """
    Load all three campaigns (all, men, women) for a given policy.
    
    Args:
        behavior_policy (str): 'random' or 'bts'
        dataset_type (str): 'sample' or 'full'
    
    Returns:
        dict: Dictionary with keys 'all', 'men', 'women' mapping to DataFrames
    
    Examples:
        >>> campaigns = load_all_campaigns('random', 'sample')
        >>> print(f"All: {len(campaigns['all'])} records")
        >>> print(f"Men: {len(campaigns['men'])} records")
        >>> print(f"Women: {len(campaigns['women'])} records")
    """
    campaigns = {}
    for campaign in ['all', 'men', 'women']:
        try:
            campaigns[campaign] = load_data(behavior_policy, campaign, dataset_type)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {campaign}: {e}")
            campaigns[campaign] = None
    
    return campaigns


def load_all_policies(campaign="all", dataset_type="sample"):
    """
    Load both policies (random, bts) for a given campaign.
    
    Args:
        campaign (str): 'all', 'men', or 'women'
        dataset_type (str): 'sample' or 'full'
    
    Returns:
        dict: Dictionary with keys 'random', 'bts' mapping to DataFrames
    
    Examples:
        >>> policies = load_all_policies('all', 'full')
        >>> print(f"Random: {len(policies['random'])} records")
        >>> print(f"BTS: {len(policies['bts'])} records")
    """
    policies = {}
    for policy in ['random', 'bts']:
        try:
            policies[policy] = load_data(policy, campaign, dataset_type)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {policy}: {e}")
            policies[policy] = None
    
    return policies


# Module metadata
__version__ = "1.0.0"
__author__ = "Armando Ordorica"
__all__ = [
    'load_data',
    'compute_ctr',
    'get_dataset_stats',
    'load_all_campaigns',
    'load_all_policies'
]
