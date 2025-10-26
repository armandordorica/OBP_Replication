import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
  
    
    
import pandas as pd

# Set pandas display options (if these are global preferences, they should be set once)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


import plotly.graph_objects as go



try:
    # Attempt to import and configure PrestoClient for a specific environment
    from data_clients.presto.presto_client import PrestoClient

    presto_client = PrestoClient(
        'presto_devapp_access_username',  # Placeholder for Knox key, replace with actual credentials
        'presto_devapp_access_password',  # Placeholder for Knox key, replace with actual credentials
        'datahub',
        user='svc-ts-model-evaluation',
        hostname='presto-gateway.pinadmin.com',
        port='443',
        protocol='https',
    )

    def query_to_df(query):
        cur = presto_client.execute_query(query)
        output_df = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
        return output_df

except:
    # Fallback for a different environment where the first import might not be available
    print("Importing from jupyter pinadmin")
    from p.v1 import *
    import p.v1 as v1

    dir(v1)

    import plotly.graph_objects as go

    def query_to_df(query):
        output_df = presto(query, use_cache=False)
        return output_df



def create_folder(folder_name): 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return 

def get_column_categorization_df(input_df): 
    col_types = []
    for col_name in input_df.columns: 
        col_types.append((input_df[col_name].dtypes.name, col_name))

    col_types_df = pd.DataFrame({'col_type':[],'col_name':[]})

    for col_type in col_types: 
        temp_df = pd.DataFrame({'col_type':[col_type[0]],'col_name':[col_type[1]]})
        col_types_df = col_types_df.append(temp_df)
    col_types_df.sort_values(by=['col_type', 'col_name'])

    return col_types_df

def replace_nan_with_null_string( x):
    """
    Given a value specified by a row and column of a dataframe, replace NaNs with
    the 'NULLVALUE' string.
    """
    try:  # value is a number
        if math.isnan(x):
            return 'NULLVALUE'
        else:
            return x
    except:  # value is not a number
        if x is None:
            return 'NULLVALUE'
        else:
            return x


def get_null_rate(df, col_name):
    """
    Returns the null rate in a column of a dataframe.
    """
    null_rate = len(df[df[col_name].isna()]) / len(df[col_name]) * 100
    return null_rate


def get_null_rates_df(df):
    """
    Outputs a dataframe with the null rates of each of the columns in the input dataframe.
    """
    variables = list(df.columns)[1:]

    null_rates = []
    for var in variables:
        null_rate = get_null_rate(df, var)
        null_rates.append(null_rate)

    all_null_rates_df = pd.DataFrame({'Variable': variables, 'Null Rate': null_rates}).sort_values(by='Null Rate',
                                                                                                   ascending=False)

    return all_null_rates_df



def get_sql_table_null_rate(table_name): 
    
    cols= list(query_to_df(f"""select * from {table_name} limit 1""").columns)

    query = ""
    for i in range(0,len(cols)):
        col = cols[i]
        query+= f""" select '{col}' as column_name, 
        cast((COUNT(*) - COUNT({col})) as DECIMAL) as null_numerator,
        cast(COUNT(*) as DECIMAL) as all_rows,
        cast((COUNT(*) - COUNT({col})) as decimal)/COUNT(*)  AS null_rate
        from {table_name}
        """
        if len(cols)-i>=2:
            query+="""UNION ALL"""
            query+="""\n"""

    null_rate_df = query_to_df(query)    
    null_rate_df['null_rate'] = null_rate_df['null_numerator'].astype(float)/null_rate_df['all_rows'].astype(float)
    null_rate_df.sort_values(by='null_rate', ascending=False, inplace=True)
    return null_rate_df

def get_cols_low_null_rate(df, threshold_pct=99):
    """
    Given a dataframe and a specified threshold, it returns a list of the columns that have their fields populated
    in more than X% of the time, where X is the corresponding threshold.
    """
    all_null_rates_df = get_null_rates_df(df)

    non_null_cols = list(all_null_rates_df[all_null_rates_df['Null Rate'] < threshold_pct]['Variable'])
    return non_null_cols


def get_redundant_variables(df):
    """
    Given a dataframe, it returns a list of variables that have the same value across all rows.
    """
    
    list_of_cols = list(df.columns)
    error_vars = []
    vars_with_no_change = []

    for i in range(0, len(list_of_cols)):
        print("Analyzing " + list_of_cols[i])
        try: 
            if len(df[list_of_cols[i]].unique()) <= 1:
                vars_with_no_change.append(list_of_cols[i])
        except: 
            print(f"Error with variable {list_of_cols[i]} ")
            error_vars.append(list_of_cols[i])
            pass
        
    print(f"Variables that caused errors: {error_vars}")

    return vars_with_no_change


def get_count_variables(list_of_vars):
    """
    Given a list of variables, it extracts a subset of 'cnt' variables

    Output: List
    """
    count_vars = [x for x in list_of_vars if 'cnt' in x]
    list_of_lists = [x.split('_')[:-1] for x in count_vars]
    a = list(set(['_'.join(x) for x in list_of_lists]))
    a.sort()
    return a


def replace_null_value_with_float(x):
    if x == 'NULLVALUE':
        return 0
    else:
        return float(x)


def get_most_similar_value_to_null(df, var):
    #     print("Variable:{}".format(var))
    """
    Input: Expects a dataframe with a column that has null values that show as "NULLVALUE" (no NaNs but actual strings).

    The function looks at the fraud rates of all possible values of that variable and selects the one that most ressembles
    the fraud rate to the null values.

    Output:
    Returs the key within the specified column of the dataframe with the most similar fraud rate behaviour to the rows
    marked as NULLVALUE. This will be used to impute the nulls.
    """

    temp_df = df.groupby(var)[[target_var]].mean()
    #     print(temp_df)

    try:
        null_value_fraud_rate = temp_df.loc['NULLVALUE'].iloc[0]
        temp_df.drop(['NULLVALUE'], axis=0, inplace=True)

    except:
        print("Can't find nulls for variable {}".format(var))
        return

    print("Finding replacement for nulls for variable: {}".format(var))
    keys = list(temp_df.index)
    values = list(temp_df[target_var])

    closest_distance = np.abs(null_value_fraud_rate - values[0])
    closest_key = temp_df[temp_df[target_var] == values[0]].index[0]

    for i in range(1, len(values)):
        current_distance = np.abs(null_value_fraud_rate - values[i])
        current_key = temp_df[temp_df[target_var] == values[i]].index[0]

        if current_distance < closest_distance:
            closest_distance = current_distance
            closest_key = current_key

    closest_fraud_rate = temp_df.loc[closest_key]

    return closest_key


def get_most_similar_value_to_null_numeric_var(input_df, var_name, target_var='fraud'):
    temp_df = input_df.copy(deep=True)

    if 'NULLVALUE' not in temp_df[var_name].unique():
        print("There are no nulls for input DataFrame for variable {}".format(var_name))
        return

    ntile = 10

    var_ntile = var_name + '_ntile'

    ### Create a df with the variable name, fraud, and corresponding ntile
    temp_no_nulls_df = temp_df[temp_df[var_name] != 'NULLVALUE'][[var_name, target_var]]
    try:
        temp_no_nulls_df[var_ntile] = pd.qcut(temp_no_nulls_df[var_name], ntile, labels=False)

        ### Create a df with each of the ntiles and the corresponding fraud rate
        ntile_fraud_rates = temp_no_nulls_df.groupby(var_ntile)[[target_var]].mean()
        print("ntile_fraud_rates:{}".format(ntile_fraud_rates))

        ### Calculate the fraud rate for the null values
        null_fraud_rate = temp_df[temp_df[var_name] == 'NULLVALUE'].groupby(var_name)[[target_var]].mean()[target_var].iloc[0]

        print("Null Fraud Rate: {}".format(null_fraud_rate))

        ### Create a distance column that stores the diff between the fraud rate of null values and each of the ntiles
        ntile_fraud_rates['distance'] = np.abs(ntile_fraud_rates[target_var] - null_fraud_rate)

        ### Obtain ntile that behaes the most similar to nullvalues
        closest_ntile = ntile_fraud_rates[ntile_fraud_rates['distance'] == ntile_fraud_rates['distance'].min()].index[0]

        print("Closest ntile:{}".format(closest_ntile))

        ### Value to impute will be the mean value of the ntile that behaves the closest to nulls
        value_to_impute = temp_no_nulls_df[temp_no_nulls_df[var_ntile] == closest_ntile][var_name].mean()

        print("value_to_impute:{}".format(value_to_impute))
        return value_to_impute

    except:
        value_to_impute = get_most_similar_value_to_null(input_df, var_name)
        print("value_to_impute:{}".format(value_to_impute))
        return value_to_impute

    
def get_null_rate_over_time(input_df, var_name, primary_key='scanReference', date_key='fetch_date'): 

    temp_df = input_df[[primary_key,date_key, var_name]].copy(deep=True)

    var_name_null = var_name+'_null'
    temp_df[var_name_null] = np.where(temp_df[var_name].isna(), 1,0)

    temp_df2 = temp_df.groupby([date_key])[[var_name_null]].mean()

    temp_df2.reset_index(inplace=True)
    temp_df2[date_key] = pd.to_datetime(temp_df2[date_key])
    temp_df2.columns =['date', var_name+'_null_rate']
    
    return temp_df2


def plot_null_rate_over_time(input_df, var_name, date_key='date'): 
    temp_df = get_null_rate_over_time(input_df, var_name)

    plt.figure(figsize=(20,10))
    plt.plot(temp_df[date_key], temp_df[var_name+'_null_rate'])
    plt.title("Null Rate for {}".format(var_name), fontsize=24)
    plt.show()


def get_summary_null_rates_over_time_df(input_df, list_of_vars, null_rates_df): 
    i=0
    var_name = list_of_vars[i]
    null_rates_over_time = get_null_rate_over_time(input_df, var_name)

    for i in range(1,len(list_of_vars)): 
        var_name = list_of_vars[i]
        if null_rates_df[null_rates_df['Variable']==var_name]['Null Rate'].iloc[0] > 0: 
            null_rates_over_time[var_name+'_null_rate'] = get_null_rate_over_time(input_df, var_name)[var_name+'_null_rate']
        else:
            print("Variable:{} has no nulls".format(var_name))

    return null_rates_over_time


def get_num_high_null_days_over_past_month(input_df, var_name, threshold = 0.99,date_key='fetch_date'): 
    null_rates_over_time = get_null_rate_over_time(input_df, var_name)
    temp_df =null_rates_over_time[[date_key, var_name+'_null_rate']][::-1]
    temp_df.reset_index(inplace=True)
    temp_df['all_null'] = np.where(temp_df[[var_name+'_null_rate']]>threshold,1,0)
    temp_df['cumsum'] = temp_df['all_null'].cumsum()
    num_null_days_last_30_days = temp_df.iloc[0:30]['cumsum'].iloc[0]
    
    return num_null_days_last_30_days

def trendline(index,data, order=1):
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)


def get_null_rate_stats(null_rates_over_time, col_name):
    sns.distplot(null_rates_over_time[col_name])
    plt.title(f"Distribution of Null Rates for {col_name}")
    plt.show()
    
    mean = null_rates_over_time[col_name].mean()
    lbound = mean - 3*null_rates_over_time[col_name].std()
    hbound = mean+ 3*null_rates_over_time[col_name].std()

    num_items_outside_bounds = len(null_rates_over_time[(null_rates_over_time[col_name] < lbound) | (null_rates_over_time[col_name] > hbound)])

    pct_outside_bounds = num_items_outside_bounds/len(null_rates_over_time)

    return mean, lbound, hbound, pct_outside_bounds

def replace_numeric_null_with_numeric_value(x, replace_value=999999999 ):
    if x is None: 
        return replace_value
    elif np.isnan(x): # for Float types 
        return replace_value 
    else: 
        return x 
    
    
   
def get_most_similar_value_to_null_numeric_int_var(input_df, var_name, target_var = 'is_approved',  null_value=999999999): 
    temp_df = input_df.copy(deep=True)
    temp_no_nulls_df = temp_df[temp_df[var_name]<null_value][[var_name, target_var]]

    fraud_rates_df = temp_no_nulls_df.groupby(var_name)[[target_var]].mean()

    null_fraud_df = temp_df[temp_df[var_name]>=null_value].groupby(target_var).count()
    null_fraud_df.reset_index(inplace=True)

    if set(null_fraud_df[target_var].unique())==set([0,1]): 
        numerator = null_fraud_df[null_fraud_df[target_var]==1]['scan_reference'].iloc[0]
    else: 
        print("No fraud instances found for {}".format(var_name))
        numerator = 0
    denominator = null_fraud_df['scan_reference'].sum()

    null_fraud_rate = numerator/denominator

    fraud_rates_df['null_fraud_rate']=null_fraud_rate
    fraud_rates_df['distance_to_fraud_rate']=np.abs(fraud_rates_df[target_var] - fraud_rates_df['null_fraud_rate'])

    value_to_impute = fraud_rates_df[fraud_rates_df['distance_to_fraud_rate']==\
                                     fraud_rates_df['distance_to_fraud_rate'].min()].index[0]
    
    print(fraud_rates_df)
    return value_to_impute


def get_most_similar_value_to_null_float_var(input_df, var_name, target_var = 'is_approved'): 
    temp_no_nulls_df = input_df[input_df[var_name]<=1][[var_name, target_var]]
    temp_nulls_df = input_df[input_df[var_name]>1][[var_name, target_var]]

    var_name_approx = var_name+'_approx'
    var_ntile = var_name + '_ntile'
    ntile = 10 
    max_decimal_places = 10
    num_decimal_places = 1

    while num_decimal_places < max_decimal_places:
        print("Num of Decimal Places: {}".format(num_decimal_places))
        temp_no_nulls_df[var_name_approx] = np.round(temp_no_nulls_df[var_name],num_decimal_places)
        try: 
            temp_no_nulls_df[var_ntile] = pd.qcut(temp_no_nulls_df[var_name_approx], ntile,  labels=False)
            temp_no_nulls_df['ntile_range'] = pd.qcut(temp_no_nulls_df[var_name_approx], ntile)
            break 
        except: 
            pass 
        num_decimal_places+=1 

    ntile_ranges = temp_no_nulls_df[[var_ntile, 'ntile_range']].sort_values(by=var_ntile, ascending=True)['ntile_range'].unique()

    ### Create a df with each of the ntiles and the corresponding fraud rate
    ntile_fraud_rates = temp_no_nulls_df.groupby(var_ntile)[[target_var]].mean()
    ntile_fraud_rates['ntile_ranges']=ntile_ranges

    null_fraud_rate = temp_nulls_df.groupby(var_name)[[target_var]].mean()[target_var].iloc[0]
    print("Null Fraud Rate: {}".format(null_fraud_rate))

    ### Create a distance column that stores the diff between the fraud rate of null values and each of the ntiles
    ntile_fraud_rates['distance'] = np.abs(ntile_fraud_rates[target_var] - null_fraud_rate)
    print("Ntile fraud rates:{}".format(ntile_fraud_rates))

    ### Obtain ntile that behaes the most similar to nullvalues
    closest_ntile = ntile_fraud_rates[ntile_fraud_rates['distance'] == ntile_fraud_rates['distance'].min()].index[0]
    print("Closest ntile:{}".format(closest_ntile))

    ### Value to impute will be the mean value of the ntile that behaves the closest to nulls
    value_to_impute = temp_no_nulls_df[temp_no_nulls_df[var_ntile] == closest_ntile][var_name].mean()
    print("value_to_impute:{}".format(value_to_impute))
    
    return value_to_impute 




