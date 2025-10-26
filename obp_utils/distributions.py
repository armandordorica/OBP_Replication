import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
import seaborn as sns


def plot_bar_and_cumulative_distribution(
    df,
    category_col,
    bar_col,
    cumulative_col,
    title="Prevalence and Cumulative Distribution",
    xlabel="Category",
    ylabel="Percentage (%)",
    figsize=(20, 10),
    bar_color="blue",
    line_color="red",
    annotate=True
):
    """
    Plots a bar chart with a cumulative distribution line overlay.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - category_col (str): Column name for the categories on the x-axis.
    - bar_col (str): Column name for the bar values.
    - cumulative_col (str): Column name for the cumulative line values.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Figure size (width, height).
    - bar_color (str): Color for the bars.
    - line_color (str): Color for the cumulative line.
    - annotate (bool): Whether to annotate the bars and line points with values.
    
    Returns:
    - None
    """
    # Plotting
    plt.figure(figsize=figsize)

    # Bar Plot for the specified bar column
    bars = plt.bar(df[category_col], df[bar_col], color=bar_color, alpha=0.7, label="Percentage Over Total")

    # Line Plot for the cumulative column
    line, = plt.plot(df[category_col], df[cumulative_col], marker="o", color=line_color, label="Cumulative Distribution")

    # Annotate the Bar Plot
    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.2f}", ha="center", fontsize=8)

        # Annotate the Cumulative Line
        for i, y in enumerate(df[cumulative_col]):
            plt.text(i, y + 0.5, f"{y:.2f}", ha="center", color=line_color, fontsize=8)

    # Formatting the Plot
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()
    




def calculate_ci_ecdf(series, confidence=0.95):
    series_clean = series.dropna()
    mean = round(series_clean.mean(), 4)

    sorted_series = np.sort(series_clean)
    ecdf = np.arange(1, len(sorted_series) + 1) / len(sorted_series)

    alpha = 1 - confidence
    lower_bound_index = np.searchsorted(ecdf, alpha / 2)
    upper_bound_index = np.searchsorted(ecdf, 1 - alpha / 2)

    ci_lower = round(sorted_series[max(0, lower_bound_index - 1)], 4)
    ci_upper = round(sorted_series[min(len(sorted_series) - 1, upper_bound_index)], 4)

    return mean, ci_lower, ci_upper


def get_cdf_df(input_df, categorical_var): 
    temp_df = input_df.groupby(categorical_var).size().reset_index(name='count')
    temp_df.sort_values(by=categorical_var, ascending=True, inplace=True)
    temp_df['cum_sum'] = temp_df['count'].cumsum()
    temp_df['total'] = temp_df['count'].sum()
    temp_df['pct_over_total'] = temp_df['count']/temp_df['total']
    temp_df['cum_pct'] = temp_df['cum_sum']/temp_df['total']
    return temp_df


def get_capped_df_by_categorical_cdf(input_df, category_var,cap_pct=0.90): 
    """
    Summary: Given a dataframe with a categorical variable of interest, return a dataframe with categories that explain `cap_pct` percentage of the total rows 
    
    Input: 
        input_df - dataframe containing the categorical variable, i.e. pin_l2_interest
        category_var - a string describing the name of the categorical variable 
        cap_pct - pct where you want to cap the categories, i.e. a cap_pct of 0.90 
                    will give you a dataframe with only the categories that make up 
                    the first 90% of the entities 
        
    Output: 
        A dataframe with only the categories that make up 
                    the first 90% of the entities 
        
    """
    
    temp_df= input_df.groupby(category_var).size().reset_index(name='count')
    temp_df.sort_values(by='count', ascending=False, inplace=True)
    temp_df['total_count'] = temp_df['count'].sum()
    temp_df['cum_sum'] = temp_df['count'].cumsum()
    temp_df['cum_sum_pct']= temp_df['cum_sum']/temp_df['total_count']
    temp_df = temp_df[temp_df['cum_sum_pct']<cap_pct]
    
    return input_df[input_df[category_var].isin(list(temp_df[category_var].unique()))]



def get_confidence_interval(input_df, variable_name, conf_interval=95): 
    cdf_df = get_cdf_df(input_df, variable_name)
    mean = np.mean(input_df[variable_name])

    conf_interval = 95
    tails_width = 100-conf_interval

    tail_width = tails_width/2
    
    low_bound = cdf_df[cdf_df['cum_pct']<=tail_width/100][variable_name].max()
    
    high_bound = cdf_df[cdf_df['cum_pct']<=(100-tail_width)/100][variable_name].max()
    
    plt.figure(figsize=(20,10))
    sns.distplot(input_df[variable_name])
    plt.axvline(x = np.round(low_bound,4), color = 'b', ls='--', label = f'Low bound {np.round(low_bound,4)}')   
    plt.axvline(x = np.round(mean,4), color = 'r', ls='--', label = f'Mean {np.round(mean,4)}') 

    plt.axvline(x = np.round(high_bound,4), color = 'b', ls='--', label = f'High bound {np.round(high_bound,4)}') 
    plt.xlabel(f"Values of {variable_name}")
    plt.title(f"Distribution of {variable_name} with confidence interval bounds of {conf_interval}%", fontsize=20)
    plt.legend()
    plt.show()
    
    
    return low_bound, mean, high_bound
    


def get_statistical_confidence_interval(mean, variance, sample_size, confidence_level=0.95):
    """
    Computes the confidence interval given the mean, variance, sample size, and confidence level.
    
    Args:
    mean (float): The mean of the sample.
    variance (float): The variance of the sample.
    sample_size (int): The size of the sample.
    confidence_level (float): The desired confidence level, expressed as a decimal.
    
    Returns:
    tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    z = abs(stats.norm.ppf((1 - confidence_level) / 2))
    std_err = math.sqrt(variance / sample_size)
    interval = z * std_err
    lower_bound = mean - interval
    upper_bound = mean + interval
    return (lower_bound, upper_bound)
