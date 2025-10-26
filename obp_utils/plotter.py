### Plotting Functions

import matplotlib.pyplot as plt
import PADS_Toolset.correlation as corr
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo


import seaborn as sns


from matplotlib.ticker import FuncFormatter
import pandas as pd


from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np
import yaml 

from datetime import datetime
import os


import random


def add_jitter(value, jitter=0.01):
    return value + random.uniform(-jitter, jitter)

try: 

    from p.v1 import *
    import p.v1 as v1

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dir(v1)

    import plotly.graph_objects as go

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    def query_to_df(query):
        output_df = presto(query, use_cache=False)
        return output_df
    
except: 
    from data_clients.presto.presto_client import PrestoClient
    presto_client = PrestoClient(
    'presto_devapp_access_username', # Get/create your own knox key specific to your team/org
    'presto_devapp_access_password', # Get/create your own knox key specific to your team/org
    'datahub',
    user='svc-ts-model-evaluation',
    hostname='presto-gateway.pinadmin.com',
    port='443',
    protocol='https',
    )
    def query_to_df(query):
        cur = presto_client.execute_query(query)
        output_df  = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
        return output_df

def read_yaml(file_name): 
    with open(file_name) as file:
        yaml_data= yaml.safe_load(file)
    return yaml_data

##################


def plot_waterfall(input_cdf_df, input_categorical_var, input_title, fig_width=650, fig_height=500, max_pct=0.95): 
    top_pct = input_cdf_df[input_cdf_df['cum_pct']<=max_pct]

    measure = ['relative'] *len(top_pct)
    measure.append("total")

    x = list(top_pct[input_categorical_var])
    x.append("TOTAL")

    text = list(top_pct['cum_pct'].diff())
    text[0] = top_pct['cum_pct'].iloc[0]
    text.append(top_pct['cum_pct'].iloc[-1])
    text = [str(np.round(x,3)) for x in text]

    y = list(top_pct['cum_pct'].diff())
    y[0] = top_pct['cum_pct'].iloc[0]
    y.append(0)


    fig = go.Figure(go.Waterfall(
    name = "", orientation = "v",
    measure = measure, 
    x = x,
    textposition = "outside",
    text =  text,
    y = y,
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title = input_title ,
        showlegend = True
    )
    fig.update_xaxes(title_text='{}'.format(input_categorical_var))
    fig.update_yaxes(title_text='Pct over total')


    fig.show()


def plot_waterfall_breakdown(input_categorical_var, input_action,input_sql_table, title, fig_width=650, fig_height=500, max_pct=0.95): 
    
    input_categorical_var_counts = query_to_df(f"select {input_categorical_var}, sum(sum_{input_action}) as num_instances from {input_sql_table} group by 1 ")
    input_categorical_var_counts.sort_values(by='num_instances', ascending=False, inplace=True)
    input_categorical_var_counts['pct_over_total'] = input_categorical_var_counts['num_instances']/input_categorical_var_counts['num_instances'].sum()
    input_categorical_var_counts['cum_pct'] = input_categorical_var_counts['pct_over_total'].cumsum()
    top_pct = input_categorical_var_counts[input_categorical_var_counts['cum_pct']<=max_pct]

    measure = ['relative'] *len(top_pct)
    measure.append("total")
    
    x = list(top_pct[input_categorical_var])
    x.append("TOTAL")

    text = list(top_pct['cum_pct'].diff())
    text[0] = top_pct['cum_pct'].iloc[0]
    text.append(top_pct['cum_pct'].iloc[-1])
    text = [str(np.round(x,3)) for x in text]

    y = list(top_pct['cum_pct'].diff())
    y[0] = top_pct['cum_pct'].iloc[0]
    y.append(0)


    fig = go.Figure(go.Waterfall(
        name = "", orientation = "v",
        measure = measure, 
        x = x,
        textposition = "outside",
        text =  text,
        y = y,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
            width=fig_width,
            height=fig_height,
            title = title ,
            showlegend = True
    )
    fig.update_xaxes(title_text='{}'.format(input_categorical_var))
    fig.update_yaxes(title_text='Pct over total')


    fig.show()
    
    return input_categorical_var_counts


def plot_lineplot(x, y, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    # plt.show()

def plot_scatterplot(x, y, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    plt.show()

def plot_correlation(df, var_name, target_var='fraud', min_correlation=0.05):
    """
    Generates a plot between the values of a column in a dataframe and the target variable.
    It filters out irrelevant values to only keep those above a min_correlation (absolute value).
    """
    title = f"Correlation between {var_name} and {target_var}"
    df3 = corr.get_correlation_df(df, var_name, target_var, min_correlation)

    df3.plot.bar(
        figsize=(20, 10), title="Correlation with {}".format(var_name), fontsize=15, grid=True)
    plt.title(title, fontsize=20)
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel("Correlation with {}".format(target_var), fontsize=20)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    plt.show()
    
    
    
def plot_plotly_scatterplot(x_list, y_list, fig_width=800, fig_height=600, fig_title='', x_axis_text='', y_axis_text='', xtick_angle=90): 
    fig = px.scatter(x=x_list, y=y_list)
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title = fig_title ,
        showlegend = True
    )
    fig.update_xaxes(title_text=x_axis_text, tickangle=90)
    fig.update_yaxes(title_text=y_axis_text)
    fig.show()
    
    
def plot_plotly_lineplot(x_list, y_list, fig_width=800, fig_height=600, fig_title='', x_axis_text='', y_axis_text='', xtick_angle=90): 
    fig = px.line(x=x_list, y=y_list)
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title = fig_title ,
        showlegend = True
    )
    fig.update_xaxes(title_text=x_axis_text, tickangle=90)
    fig.update_yaxes(title_text=y_axis_text)
    fig.show()
    
    
    


    
#### User Sequences 



def plot_user_actions(input_transitions_df, user_id, session_num, actions_dict, order_feedviews, figsize=(20,10), title='User Session Visualization with User Actions'):
    sample_sequence_df = input_transitions_df[(input_transitions_df['user_id'] == user_id) & (input_transitions_df['session_num'] == session_num)]
    sample_sequence_df['feedview_label_num'] = sample_sequence_df['feedview_type'].map(order_feedviews)

    x = np.array(pd.to_datetime(sample_sequence_df['a_fv_start_time_stamp_ms_datetime']))
    y = np.array(sample_sequence_df['feedview_label_num'])

    fig, ax = plt.subplots(figsize=figsize)
    plt.step(x, y)

    for action, properties in actions_dict.items():
        ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c=properties['color'], s=sample_sequence_df[action]*properties['multiplier'], marker=properties['marker'], label=properties['label'], zorder=properties['zorder'], alpha=properties['alpha'])

    x_padding = 0.01
    y_padding = 0.1

    ax.set_xlim(min(x) - x_padding * (max(x) - min(x)), max(x) + x_padding * (max(x) - min(x)))
    ax.set_ylim(min(y) - y_padding, max(y) + y_padding)

    colors = ['papayawhip', 'white']
    n_stripes = len(order_feedviews)

    ymin, ymax = ax.get_ylim()
    stripe_height = (ymax - ymin) / n_stripes

    for i in range(n_stripes):
        ax.add_patch(
            Rectangle(
                (min(x), ymin + i * stripe_height),
                max(x) - min(x),
                stripe_height,
                color=colors[i % len(colors)],
                zorder=-1
            )
        )

    initial_time = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].min().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].min(), color = 'b', ls='--', label = f'Initial Timestamp:{initial_time}', alpha = 0.2)    

    final_time = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].max().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].max(), color = 'b', ls='--', label = f'Final Timestamp:{final_time}', alpha = 0.2)    

    plt.yticks(list(sample_sequence_df['feedview_label_num']), list(sample_sequence_df['feedview_type']))
    plt.xticks(rotation=90)
    plt.title(f"{title}, user:{user_id}")
    plt.legend(loc='upper right', title='User Actions', bbox_to_anchor=(1.22, 1.0))
    plt.xlabel("Feedview timestamp")
    plt.ylabel("Feedview (Surface)")

    plt.show()
    
    
def plot_user_sequence(input_transitions_df, user_id, session_num): 
    
    """
    Plots a sample sequence of feedviews for a given user and session number, along with various user activity types.

    Parameters:
        input_transitions_df (pandas.DataFrame): Dataframe containing information about user feedview sequences.
        user_id (int): Unique ID of the user for whom the sequence is being plotted.
        session_num (int): Session number for the given user.

    Returns:
        None.

    Visualizations:
        Generates a plot of the feedview sequence with various user activity types.

    """
    order_feedviews = {'HOME_FEED': 1,
     'RELATED_PRODUCT_FEED': 2,
     'RELATED_STORIES_FEED': 3,
     'FLASHLIGHT': 4,
     'SHOPPING_LIST_FEED': 5,
     'RELATED_PIN_FEED': 6,
     'FLASHLIGHT_STELA_CAROUSEL': 7,
     'SEARCH_PINS': 8,
     'BOARD_FEED': 9}


    sample_sequence_df = input_transitions_df[(input_transitions_df['user_id']==user_id) & (input_transitions_df['session_num']==session_num)]
    sample_sequence_df['feedview_label_num'] = sample_sequence_df['feedview_type'].map(order_feedviews)
    sample_sequence_df.sort_values(by='feedview_num', ascending=True, inplace=True)
    order_feedviews = {'HOME_FEED': 1,
         'RELATED_PRODUCT_FEED': 2,
         'RELATED_STORIES_FEED': 3,
         'FLASHLIGHT': 4,
         'SHOPPING_LIST_FEED': 5,
         'RELATED_PIN_FEED': 6,
         'FLASHLIGHT_STELA_CAROUSEL': 7,
         'SEARCH_PINS': 8,
         'BOARD_FEED': 9}


    x = np.array(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'])
    y= np.array(sample_sequence_df['feedview_label_num'])

    fig, ax = plt.subplots()
    plt.step(x,y)


    ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c='yellow', s=sample_sequence_df['sum_impressions'], marker='s', label='impression activity', zorder=1)
    ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c='red', s=sample_sequence_df['sum_repins'], marker= 'd', label='repin activity', zorder=3)
    ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c='green', s=sample_sequence_df['sum_closeups'], marker= 'o', label='closeup activity', zorder=1)
    ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c='orange', s=sample_sequence_df['sum_clickthroughs'], marker= '+', label='clickthrough activity', zorder=3)
    ax.scatter(sample_sequence_df['a_fv_start_time_stamp_ms_datetime'], sample_sequence_df['feedview_label_num'], c='black', s=sample_sequence_df['sum_long_clickthroughs'], marker= 'x', label='long clickthrough activity', zorder=4)


    # ax.set_xlim(min(x), max(x))
    # ax.set_ylim(min(y), max(y))

    # Define alternating colors and the number of stripes

    x_padding = 0.01
    y_padding = 0.1

    # Set the plot limits with added padding
    ax.set_xlim(min(x) - x_padding * (max(x) - min(x)), max(x) + x_padding * (max(x) - min(x)))
    ax.set_ylim(min(y) - y_padding, max(y) + y_padding)

    colors = ['papayawhip', 'white']
    n_stripes = len(order_feedviews)

    # Calculate the height of each stripe
    ymin, ymax = ax.get_ylim()
    stripe_height = (ymax - ymin) / n_stripes

    # Add stripes to the plot
    for i in range(n_stripes):
        ax.add_patch(
            Rectangle(
                (min(x), ymin + i * stripe_height),
                max(x) - min(x),
                stripe_height,
                color=colors[i % len(colors)],
                zorder=-1
            )
        )

    initial_time = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].min().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].min(), color = 'b', ls='--', label = f'Initial Timestamp:{initial_time}', alpha = 0.2)    

    final_time = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].max().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['a_fv_start_time_stamp_ms_datetime'].max(), color = 'b', ls='--', label = f'Initial Timestamp:{final_time}', alpha = 0.2)    

    plt.yticks(list(sample_sequence_df['feedview_label_num']), list(sample_sequence_df['feedview_type']))
    plt.xticks(rotation=90)
    plt.title(f"Sample Rabbit Hole Visualization, user:{user_id}")
    plt.legend(loc='upper right', title='User Actions', bbox_to_anchor=(1.7, 1.0))
    plt.xlabel("Feedview timestamp")
    plt.ylabel("Feedview (Surface)")

    plt.show()
    
    
    
    
    




def plot_lx_interests_sequence(input_lx_transitions_df, user_id, session_num, subset_colors = ['red', 'green', 'blue', 'orange', 'yellow']):
    
    """
    Plots a visualization of the user's browsing sequence through different feedviews during a session,
    highlighting their top 5 interests.
    
    Args:
    input_lx_transitions_df (pd.DataFrame): A DataFrame containing user browsing behavior data, including user ID, session number, feedview types, and timestamps.
    user_id (int): The ID of the user whose behavior needs to be visualized.
    session_num (int): The session number of the user's browsing behavior to be visualized.
    subset_colors (list, optional): A list of colors to be used for highlighting the top 5 interests. Defaults to ['red', 'green', 'blue', 'orange', 'yellow'].

    Returns:
    None

    Displays:
    A step plot of the user's browsing sequence through different feedviews, with top 5 interests highlighted as colored squares.

    """
    
    order_feedviews = {'HOME_FEED': 1,
     'RELATED_PRODUCT_FEED': 2,
     'RELATED_STORIES_FEED': 3,
     'FLASHLIGHT': 4,
     'SHOPPING_LIST_FEED': 5,
     'RELATED_PIN_FEED': 6,
     'FLASHLIGHT_STELA_CAROUSEL': 7,
     'SEARCH_PINS': 8,
     'BOARD_FEED': 9}
    
    sample_sequence_df = input_lx_transitions_df[(input_lx_transitions_df['user_id']==user_id) & (input_lx_transitions_df['session_num']==session_num)]
    sample_sequence_df['feedview_label_num'] = sample_sequence_df['feedview_type'].map(order_feedviews)
    sample_sequence_df.sort_values(by='feedview_num', ascending=True, inplace=True)

    grouped_df = sample_sequence_df.groupby(['user_id', 'session_num'])[[x for x in list(input_lx_transitions_df.columns) if 'sum_pins_l1' in x]].sum()

    top_5_cols = grouped_df.sum().nlargest(5).index.tolist()

    order_feedviews = {'HOME_FEED': 1,
         'RELATED_PRODUCT_FEED': 2,
         'RELATED_STORIES_FEED': 3,
         'FLASHLIGHT': 4,
         'SHOPPING_LIST_FEED': 5,
         'RELATED_PIN_FEED': 6,
         'FLASHLIGHT_STELA_CAROUSEL': 7,
         'SEARCH_PINS': 8,
         'BOARD_FEED': 9}


    sample_sequence_df['feedview_start_datetime'] = pd.to_datetime(sample_sequence_df['feedview_start_datetime'])
    x = np.array(sample_sequence_df['feedview_start_datetime'])
    y= np.array(sample_sequence_df['feedview_label_num'])

    fig, ax = plt.subplots()
    plt.step(x,y)

    for i in range(0,len(top_5_cols)): 
        interest =top_5_cols[i]
        chosen_color =subset_colors[i]
        ax.scatter(sample_sequence_df['feedview_start_datetime'], sample_sequence_df['feedview_label_num'], c=chosen_color, s=sample_sequence_df[interest], marker='s', label=interest, zorder=i)

    x_padding = 0.01
    y_padding = 0.1

    # Set the plot limits with added padding
    ax.set_xlim(min(x) - x_padding * (max(x) - min(x)), max(x) + x_padding * (max(x) - min(x)))
    ax.set_ylim(min(y) - y_padding, max(y) + y_padding)

    colors = ['papayawhip', 'white']
    n_stripes = len(order_feedviews)

    # Calculate the height of each stripe
    ymin, ymax = ax.get_ylim()
    stripe_height = (ymax - ymin) / n_stripes

    # Add stripes to the plot
    for i in range(n_stripes):
        ax.add_patch(
            Rectangle(
                (min(x), ymin + i * stripe_height),
                max(x) - min(x),
                stripe_height,
                color=colors[i % len(colors)],
                zorder=-1
            )
        )

    initial_time = sample_sequence_df['feedview_start_datetime'].min().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['feedview_start_datetime'].min(), color = 'b', ls='--', label = f'Initial Timestamp:{initial_time}', alpha = 0.2)    

    final_time = sample_sequence_df['feedview_start_datetime'].max().strftime('%Y-%m-%d %H:%M')
    plt.axvline(x = sample_sequence_df['feedview_start_datetime'].max(), color = 'b', ls='--', label = f'Initial Timestamp:{final_time}', alpha = 0.2)    

    plt.yticks(list(sample_sequence_df['feedview_label_num']), list(sample_sequence_df['feedview_type']))
    plt.xticks(rotation=90)
    plt.title(f"User Browsing Sequence across surface highlighting activity by L1 interest, user:{user_id}")
    plt.legend(loc='upper right', title='Interests per Feedview', bbox_to_anchor=(1.7, 1.0))
    plt.xlabel("Feedview timestamp")
    plt.ylabel("Feedview (Surface)")

    plt.show()

def plotly_multiline_plot(df, columns, labels=None):
    """
    Creates a multiline plot using Plotly.

    Args:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of column names to plot.
    labels (list, optional): A list of labels corresponding to the columns. Default is None, in which case column names are used.

    Returns:
    plotly.graph_objects.Figure: The multiline plot figure.
    """

    if labels is None:
        labels = columns

    assert len(columns) == len(labels), "The number of columns and labels must be equal."

    fig = go.Figure()

    for col, label in zip(columns, labels):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, mode='lines'))

    fig.update_layout(title='Multiline Plot', xaxis_title='Date', yaxis_title='Values')

    return f



def plot_plotly_annotated_scatterplot(x, y, labels, title='', xaxis_title='', yaxis_title='', width=1200, height=800, textangle=-20, annotation_font_size=8, output_html_filename='scatterplot_with_annotations.html'): 

    trace = go.Scatter(x=x, y=y, mode='markers')

    annotations = [dict(x=add_jitter(x[i]), y=add_jitter(y[i]), text=labels[i], showarrow=False, font=dict(size=annotation_font_size), textangle=textangle) for i in range(len(x))]
    # Create layout with annotations
    layout = go.Layout(annotations=annotations, 

        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width= width,
        height =height)

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

    pyo.plot(fig, filename=output_html_filename)
    
    
def plot_plotly_annotated_scatterplot_with_error_bars(x, y, x_lower_error, x_upper_error, labels, title='', xaxis_title='', yaxis_title='', width=1200, height=800, textangle=-20, annotation_font_size=10, label_offset=-0.04, output_html_filename='scatterplot_with_annotations.html' ):
    error_x = go.ErrorX(
        array=x_upper_error,
        arrayminus=x_lower_error,
        visible=True
    )
    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        error_x=error_x
    )
    annotations = [dict(x=add_jitter(x[i]), y=y[i] + label_offset, text=labels[i], showarrow=False, font=dict(size=annotation_font_size), textangle=textangle) for i in range(len(x))]
    layout = go.Layout(
        annotations=annotations,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    pyo.plot(fig, filename=output_html_filename)



def create_multi_line_timeseries_bokeh(df, date_col, y_col, cat_col, width, height, legend_title, xlabel, ylabel, title):
    """
    Creates a multi-line time series plot using Bokeh.

    Parameters:
    - df: pandas DataFrame containing the data.
    - date_col: String name of the column in df that contains date information.
    - y_col: String name of the column in df that contains y values.
    - cat_col: String name of the categorical column for different lines.
    - width: Integer for the width of the plot.
    - height: Integer for the height of the plot.
    - legend_title: String title for the legend.
    - xlabel: String label for the x-axis.
    - ylabel: String label for the y-axis.
    - title: String title for the plot.
    """
    
    # Ensure date_col is datetime type
    df[date_col] = pd.to_datetime(df[date_col])

    # Creating a color iterator from a palette
    colors = Category10[10]

    p = figure(width=width, height=height, x_axis_type="datetime", title=title)
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    # Group by categorical column
    grouped = df.groupby(cat_col)
    
    for (name, group), color in zip(grouped, colors):
        source = ColumnDataSource(group)
        # Convert name to string explicitly
        legend_name = str(name)
        p.line(x=date_col, y=y_col, source=source, legend_label=legend_name, line_width=2, color=color, alpha=0.8)


    p.legend.title = legend_title
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

        # Adding hover tool
    hover = HoverTool()
    hover.tooltips = [
        (xlabel, '@{' + date_col + '}{%F}'), # Assuming the date is formatted as desired; customize the format as needed
        (ylabel, '@' + y_col), # Direct reference to y_col's values
        (legend_title, '$name') # Use the special field $name to display the legend item's label
    ]
    hover.formatters = {
        '@{' + date_col + '}': 'datetime', # If your date_col needs a special format, uncomment this line
    }
    p.add_tools(hover)

    show(p)



def plot_multiline_plot_matplotlib(df, date_col, value_cols, labels, plot_title='Data Plot over Time', x_label='Date', y_label='', plot_title_font_size=20, fig_width=20, fig_height=10):
    # Function to format the tick labels with commas for thousands
    def with_commas(x, pos):
        return f'{int(x):,}'

    plt.figure(figsize=(fig_width, fig_height))
    
    # Plotting the data for each column in value_cols
    for col, label in zip(value_cols, labels):
        plt.plot(df[date_col], df[col], label=label)

    # Setting the x-ticks to rotate for better readability
    plt.xticks(rotation=90)

    # Using FuncFormatter to format the y-axis with commas
    plt.gca().yaxis.set_major_formatter(FuncFormatter(with_commas))

    # Adding a legend to the plot
    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title, fontsize=plot_title_font_size)
    plt.show()




def plot_distribution_with_annotations(data, title='Distribution of Ads Engagement Score'):
    """
    Plots the distribution of a pandas Series with annotations for min, max,
    and the 25th, 50th, and 75th percentiles.

    Parameters:
        data (pd.Series): The pandas Series to be plotted.
    """
    # Calculate statistics
    min_val = data.min()
    max_val = data.max()
    percentiles = data.quantile([0.25, 0.5, 0.75])

    # Plotting
    sns.set(style="whitegrid")  # Setting the seaborn style
    plt.figure(figsize=(10, 6))
    ax = sns.distplot(data, kde=True, hist_kws={'density': True, 'linewidth': 0})

    # Adding vertical lines
    ax.axvline(min_val, color='r', linestyle='--', label=f'Min: {min_val:.2f}')
    ax.axvline(max_val, color='g', linestyle='--', label=f'Max: {max_val:.2f}')
    ax.axvline(percentiles[0.25], color='b', linestyle=':', label=f'25th Percentile: {percentiles[0.25]:.2f}')
    ax.axvline(percentiles[0.5], color='b', linestyle='-', label=f'Median: {percentiles[0.5]:.2f}')
    ax.axvline(percentiles[0.75], color='b', linestyle='-.', label=f'75th Percentile: {percentiles[0.75]:.2f}')

    # Add legend and titles
    plt.legend()
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()
