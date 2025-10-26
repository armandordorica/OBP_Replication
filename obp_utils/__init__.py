"""
OBP Utils - Utility Package for Open Bandit Pipeline Analysis

A comprehensive collection of utility modules for analyzing the Open Bandit Dataset,
including data loading, statistical analysis, visualization, and data quality checks.

Modules:
    - data_loader: Load and preprocess Open Bandit Dataset
    - correlation: Correlation analysis utilities
    - distributions: Distribution analysis and visualization
    - formatter: Data formatting utilities
    - nulls: Missing data analysis and handling
    - plotter: Advanced plotting utilities
    - timestamps: Time series analysis utilities

Author: Armando Ordorica
Date: October 26, 2025
Version: 1.0.0
"""

from .data_loader import (
    load_data,
    compute_ctr,
    get_dataset_stats,
    load_all_campaigns,
    load_all_policies
)

from .formatter import (
    comma_separator,
    format_thousands,
    remap_user_features
)

__version__ = "1.0.0"
__author__ = "Armando Ordorica"

__all__ = [
    # Data loader functions
    'load_data',
    'compute_ctr',
    'get_dataset_stats',
    'load_all_campaigns',
    'load_all_policies',
    # Formatter functions
    'comma_separator',
    'format_thousands',
    'remap_user_features',
    # Module names (import as: from obp_utils import correlation)
    'data_loader',
    'correlation',
    'distributions',
    'formatter',
    'nulls',
    'plotter',
    'timestamps',
]

def get_available_modules():
    """Return a list of available utility modules."""
    return [
        'data_loader',
        'correlation',
        'distributions',
        'formatter',
        'nulls',
        'plotter',
        'timestamps',
    ]

def print_module_info():
    """Print information about the package and available modules."""
    print("=" * 80)
    print("OBP Utils - Open Bandit Pipeline Utility Package")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("\nAvailable Modules:")
    for module in get_available_modules():
        print(f"  - {module}")
    print("\nQuick Start:")
    print("  from obp_utils import load_data")
    print("  df = load_data('random', 'all', 'sample')")
    print("\nFor module-specific imports:")
    print("  from obp_utils import correlation")
    print("  from obp_utils import plotter")
    print("=" * 80)
