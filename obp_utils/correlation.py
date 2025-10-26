import pandas as pd
import numpy as np
import seaborn as sns


def compute_multiple_pairwise_corr(dataframe, variables, target_variable):  
    """Compute pairwise correlation coefficients between each variable in a list and the target variable   
    in a pandas dataframe.  
      
    Args:   
        dataframe (pandas.DataFrame): The input dataframe containing the variables and target_variable columns.  
        variables (list of str): A list of variable names to be correlated with the target variable.  
        target_variable (str): The name of the target variable in the dataframe.  
          
    Returns:  
        pandas.DataFrame: A dataframe with two columns 'variable' and 'correlation',   
                          showing the correlation coefficient between each variable and the target variable.  
    """  
    correlations = {'variable': [], f'correlation_with_{target_variable}': []}  
      
    for variable in variables:  
        corr_df = dataframe[[variable, target_variable]].corr(method='pearson')  
        corr_coefficient = corr_df.iloc[0, 1]  
        correlations['variable'].append(variable)  
        correlations[f'correlation_with_{target_variable}'].append(corr_coefficient)  
  
    corr_dataframe = pd.DataFrame(correlations)  
    return corr_dataframe  

def get_categorical_rate_df(df, var_name):
    """
    Given a dataframe and a variable name to specify a column in that dataframe, it will return a new dataframe
    with number of scans for each of the values of the variable name as well as percentage over total.
    """
    col_name = df.groupby([var_name]).count().columns[0]

    temp_df = df.groupby([var_name])[[col_name]].count()
    temp_df['total_scans'] = temp_df[col_name].sum()
    temp_df['pct_over_total'] = temp_df[col_name] / temp_df['total_scans']

    temp_df.rename(columns={temp_df.columns[0]: "num_entries"}, inplace=True)

    # col_names = list(temp_df.columns)
    # col_names[0] = 'num_entries'
    # temp_df.columns = col_names
    temp_df.sort_values(by='pct_over_total', ascending=False, inplace=True)

    return temp_df


def compute_correlation(df, target_column='X', feature_columns=['A', 'B', 'C', 'D', 'E']):  
    correlations = df[feature_columns + [target_column]].corr()  
    target_correlations = correlations[target_column].drop(target_column)  
    return target_correlations  
  

def get_important_fields_df(df, var_name, threshold=100):
    """
    Given a dataframe and a variable name corresponding to a column name, it returns a subset of the dataframe
    where each of the entries belong to a popular variable that occurs more than `threshold` times.
    """
    
    temp_df = df.groupby([var_name]).size().reset_index(name='count')
    temp_df2 = temp_df[temp_df['count'] > threshold]

    main_values = list(temp_df2[var_name])

    return df[df[var_name].isin(main_values)]


def get_correlation_df(df, categorical_var_name, target_var, min_correlation=None):
    # One-hot encode the categorical column
    one_hot_encoded_df = pd.get_dummies(df[categorical_var_name], prefix=categorical_var_name)
    
    # Add the target variable column back to the DataFrame
    one_hot_encoded_df[target_var] = df[target_var]
    
    # Compute the correlation matrix
    correlation = one_hot_encoded_df.corr()
    
    # Extract correlations of one-hot encoded variables with the target variable
    correlations_with_target = correlation[target_var].drop(target_var)
    
    # Convert to DataFrame for nicer formatting
    result_df = correlations_with_target.to_frame()
    result_df.columns = ['Correlation with ' + target_var]
    
    # Reset index to extract categorical variable names and values
    result_df.reset_index(inplace=True)
    
    # Extract variable name and value from the index and add them as columns
    result_df['var_name'] = categorical_var_name
    result_df['values'] = result_df['index'].apply(lambda x: '_'.join(x.split(categorical_var_name)[-1].split('_')[1:]))
    
    # Sort the DataFrame based on the correlation values in descending order
    result_df.sort_values(by='Correlation with ' + target_var, ascending=False, inplace=True)

    # Apply minimum threshold filter based on the absolute value of correlation if specified
    if min_correlation is not None:
        result_df = result_df[abs(result_df['Correlation with ' + target_var]) > min_correlation]
    
    # Set the formatted index
    result_df.set_index('index', inplace=True)
    
    return result_df

def get_most_important_variables(df, potential_important_vars, threshold_count=100, min_correlation=0.08,
                                 target_var='fraud'):
    """
    Inputs:
    * Base Dataframe
    * List with potential_important_vars

    Output:
    * A dataframe showing the correlation of each of the values from potential_important_vars
    with fraud (target_var) above a threshold (min_correlation) and
    making sure that that value shows up at least X times (threshold_count).
        * Index of output is the value, i.e. country_VNM
        * Columns are `Correlation with fraud`, `var_name`, and `values`.

    Dependent functions:
    * get_important_fields_df()
    * get_correlation_df()
    """
    var_name = potential_important_vars[0]
    df_new_index = df.reset_index()
    
    temp_df = get_important_fields_df(df_new_index, var_name, threshold=threshold_count)

    high_corr_df = get_correlation_df(temp_df, var_name, target_var=target_var,
                                           min_correlation=min_correlation)

    for i in range(1, len(potential_important_vars)):
        var_name = potential_important_vars[i]
        reduced_df = get_important_fields_df(df_new_index, var_name, threshold=threshold_count)
        high_corr_temp_df = get_correlation_df(reduced_df, var_name, target_var=target_var,
                                                    min_correlation=min_correlation)
        # high_corr_df = high_corr_df.append(high_corr_temp_df)
        high_corr_df = pd.concat([high_corr_df, high_corr_temp_df])
        print(var_name)
        print(high_corr_temp_df)

    return high_corr_df


def get_corr_matrix_multiple_actions(input_df, categorical_var, actions, min_correlation=0.005):

    i=0
    a = get_correlation_df(input_df, categorical_var, target_var=f'sum_{actions[i]}',
                                               min_correlation=min_correlation)
    a.sort_values(by=f'Correlation with sum_{actions[i]}', ascending = False, inplace=True) 
    a = a[['values', f'Correlation with sum_{actions[i]}']].reset_index().drop(columns='index')
    
    for i in range(1,len(actions)): 
        b = get_correlation_df(input_df, categorical_var, target_var=f'sum_{actions[i]}',
                                                   min_correlation=min_correlation)
        b.sort_values(by=f'Correlation with sum_{actions[i]}', ascending = False, inplace=True) 
        b = b[['values', f'Correlation with sum_{actions[i]}']].reset_index().drop(columns='index')
        a = a.merge(b, how='outer', on='values')
    
    column_names = [f'Correlation with sum_{x}' for x in actions]
    a.sort_values(by=column_names, ascending = [False, False, True, True], inplace=True)
    
    return a.style.bar(align='mid', color=['red', 'lightgreen'])


def get_user_feature_correlations(df, target_var='reward', min_correlation=None):
    """
    Compute univariate correlations between each user feature value and the target variable.
    
    This function one-hot encodes each user feature column and computes the correlation
    between each encoded feature value and the target variable. Results are sorted by
    absolute correlation strength.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with user features (readable codes or hash values) and target variable
    target_var : str
        Name of the target variable column (default: 'reward')
    min_correlation : float, optional
        Minimum absolute correlation threshold to include in results.
        If None, all correlations are returned.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - 'user_feature': name of the user feature column
        - 'value': specific value of the user feature
        - 'correlation_with_{target_var}': correlation coefficient with target
        
        Sorted by absolute correlation in descending order.
    
    Examples:
    ---------
    >>> # Compute all correlations
    >>> corr_df = get_user_feature_correlations(df, target_var='reward')
    >>> 
    >>> # Filter weak correlations
    >>> corr_df = get_user_feature_correlations(df, target_var='reward', min_correlation=0.01)
    >>> print(corr_df.head(10))
    """
    # Get all user feature columns
    user_feature_cols = [col for col in df.columns if col.startswith('user_feature_')]
    
    all_correlations = []
    
    for col in user_feature_cols:
        # One-hot encode the user feature column
        one_hot_encoded = pd.get_dummies(df[col], prefix=col)
        
        # Add the target variable
        one_hot_encoded[target_var] = df[target_var].values
        
        # Compute correlations
        correlations = one_hot_encoded.corr()[target_var].drop(target_var)
        
        # Create result dataframe for this feature
        for feature_value, corr in correlations.items():
            feature_name = col
            value = feature_value.replace(f'{col}_', '')
            
            all_correlations.append({
                'user_feature': feature_name,
                'value': value,
                f'correlation_with_{target_var}': corr
            })
    
    # Create final dataframe
    result_df = pd.DataFrame(all_correlations)
    
    # Apply minimum correlation filter if specified
    if min_correlation is not None:
        result_df = result_df[abs(result_df[f'correlation_with_{target_var}']) > min_correlation]
    
    # Sort by absolute correlation
    result_df = result_df.sort_values(
        by=f'correlation_with_{target_var}', 
        key=abs, 
        ascending=False
    )
    
    return result_df

def analyze_all_variables_for_reward(df, target_var='reward', min_correlation=0.01, threshold_count=100):
    """
    Comprehensive analysis of all categorical variables and their correlation with reward.
    Analyzes action, position, user features, and item features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with categorical variables and target variable
    target_var : str
        Name of the target variable (default: 'reward')
    min_correlation : float
        Minimum absolute correlation threshold (default: 0.01)
    threshold_count : int
        Minimum number of occurrences for a value to be included (default: 100)
    
    Returns:
    --------
    dict: Dictionary with keys for each variable type containing correlation DataFrames
        - 'action': Correlations for action variable
        - 'position': Correlations for position variable  
        - 'user_features': Combined correlations for all user features
        - 'item_features': Combined correlations for all item features
        - 'summary': Summary statistics across all variables
    """
    results = {}
    all_correlations = []
    
    # Analyze action if present
    if 'action' in df.columns:
        print(f"Analyzing 'action' variable...")
        try:
            action_df = get_important_fields_df(df, 'action', threshold=threshold_count)
            action_corr = get_correlation_df(action_df, 'action', target_var=target_var, 
                                            min_correlation=min_correlation)
            results['action'] = action_corr
            all_correlations.append(action_corr)
            print(f"  Found {len(action_corr)} significant correlations")
        except Exception as e:
            print(f"  Error analyzing action: {e}")
            results['action'] = None
    
    # Analyze position if present
    if 'position' in df.columns:
        print(f"Analyzing 'position' variable...")
        try:
            position_df = get_important_fields_df(df, 'position', threshold=threshold_count)
            position_corr = get_correlation_df(position_df, 'position', target_var=target_var,
                                              min_correlation=min_correlation)
            results['position'] = position_corr
            all_correlations.append(position_corr)
            print(f"  Found {len(position_corr)} significant correlations")
        except Exception as e:
            print(f"  Error analyzing position: {e}")
            results['position'] = None
    
    # Analyze all user features
    user_feature_cols = [col for col in df.columns if col.startswith('user_feature_')]
    if user_feature_cols:
        print(f"\nAnalyzing {len(user_feature_cols)} user features...")
        user_feature_results = []
        for col in user_feature_cols:
            try:
                uf_df = get_important_fields_df(df, col, threshold=threshold_count)
                uf_corr = get_correlation_df(uf_df, col, target_var=target_var,
                                            min_correlation=min_correlation)
                user_feature_results.append(uf_corr)
                print(f"  {col}: {len(uf_corr)} significant correlations")
            except Exception as e:
                print(f"  Error analyzing {col}: {e}")
        
        if user_feature_results:
            results['user_features'] = pd.concat(user_feature_results)
            all_correlations.append(results['user_features'])
    
    # Analyze item features (user-item affinities)
    item_feature_cols = [col for col in df.columns if 'item_feature' in col.lower() and col not in user_feature_cols]
    if item_feature_cols:
        print(f"\nAnalyzing {len(item_feature_cols)} item features...")
        item_feature_results = []
        for col in item_feature_cols:
            try:
                if_df = get_important_fields_df(df, col, threshold=threshold_count)
                if_corr = get_correlation_df(if_df, col, target_var=target_var,
                                            min_correlation=min_correlation)
                item_feature_results.append(if_corr)
                print(f"  {col}: {len(if_corr)} significant correlations")
            except Exception as e:
                print(f"  Error analyzing {col}: {e}")
        
        if item_feature_results:
            results['item_features'] = pd.concat(item_feature_results)
            all_correlations.append(results['item_features'])
    
    # Create summary statistics
    if all_correlations:
        combined_df = pd.concat(all_correlations)
        corr_col = f'Correlation with {target_var}'
        
        summary = {
            'total_variables_analyzed': len([k for k in results.keys() if k != 'summary']),
            'total_significant_values': len(combined_df),
            'max_positive_correlation': combined_df[corr_col].max(),
            'max_negative_correlation': combined_df[corr_col].min(),
            'mean_abs_correlation': combined_df[corr_col].abs().mean(),
            'top_10_positive': combined_df.nlargest(10, corr_col),
            'top_10_negative': combined_df.nsmallest(10, corr_col)
        }
        results['summary'] = summary
    
    return results
