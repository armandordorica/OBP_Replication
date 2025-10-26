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
