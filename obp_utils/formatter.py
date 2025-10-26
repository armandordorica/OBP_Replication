import numpy as np
import pandas as pd

def comma_separator(x):  
    if isinstance(x, int):  
        return f"{x:,}"  
    return x  
  
def format_thousands(x):  
    return "{:,}".format(x)  

def remap_user_features(df, user_feature_columns=None):
    """
    Remap user feature hash values to readable codes (A0, B0, C0, etc.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing user features with hash values
    user_feature_columns : list, optional
        List of column names to remap. If None, will auto-detect columns
        starting with 'user_feature_'
    
    Returns:
    --------
    tuple: (remapped_df, mappings_dict)
        - remapped_df: DataFrame with remapped values
        - mappings_dict: Dictionary of {column: {original_value: new_code}}
    
    Examples:
    ---------
    >>> df_remapped, mappings = remap_user_features(df)
    >>> df_remapped, mappings = remap_user_features(df, ['user_feature_0', 'user_feature_1'])
    """
    df_copy = df.copy()
    
    # Auto-detect user feature columns if not provided
    if user_feature_columns is None:
        user_feature_columns = [col for col in df.columns if col.startswith('user_feature_')]
    
    # Letters for feature values
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    mappings = {}
    
    for col in user_feature_columns:
        if col not in df.columns:
            continue
            
        # Get unique values sorted
        unique_vals = sorted(df[col].unique())
        
        # Create mapping: hash -> letter + index
        mapping = {}
        for idx, val in enumerate(unique_vals):
            letter_idx = idx // 10  # Which letter (A, B, C, etc.)
            num_idx = idx % 10      # Which number (0-9)
            
            if letter_idx < len(letters):
                code = f"{letters[letter_idx]}{num_idx}"
            else:
                # If we run out of letters, use numeric codes
                code = f"X{idx}"
            
            mapping[val] = code
        
        # Apply mapping
        df_copy[col] = df_copy[col].map(mapping)
        mappings[col] = mapping
    
    return df_copy, mappings