"""
Visualization utilities for offline bandit policy evaluation.

This module contains reusable plotting functions for analyzing
and visualizing offline bandit data, including distribution plots,
bar charts, and cumulative distribution functions.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_bar_chart(data, x_col, y_col, x_title=None, y_title=None, title=None, 
                   width=None, height=500, show_values=True, value_format='{:,}'):
    """
    Create a bar chart with customizable parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing the data to plot
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    x_title : str, optional
        Label for x-axis (defaults to x_col if not provided)
    y_title : str, optional
        Label for y-axis (defaults to y_col if not provided)
    title : str, optional
        Chart title
    width : int, optional
        Chart width in pixels
    height : int, optional
        Chart height in pixels (default: 500)
    show_values : bool, optional
        Whether to show values on top of bars (default: True)
    value_format : str, optional
        Format string for values (default: '{:,}' for thousands separator)
    
    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The generated figure
    """
    # Set default labels if not provided
    x_title = x_title or x_col
    y_title = y_title or y_col
    
    # Create figure
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        text=y_col if show_values else None,
        title=title,
        labels={x_col: x_title, y_col: y_title},
        height=height,
        width=width
    )
    
    # Configure value display
    if show_values:
        # Convert format string to plotly template
        # Common formats: '{:,}' -> '%{text:,}', '{:.2f}' -> '%{text:.2f}'
        if ':,' in value_format:
            text_template = '%{text:,}'
        elif ':.2f' in value_format:
            text_template = '%{text:.2f}'
        elif ':.4f' in value_format:
            text_template = '%{text:.4f}'
        elif ':.6f' in value_format:
            text_template = '%{text:.6f}'
        else:
            text_template = '%{text}'
            
        fig.update_traces(
            texttemplate=text_template,
            textposition='outside',
            cliponaxis=False
        )
        
        # Add headroom so labels don't get clipped
        max_y = data[y_col].max()
        fig.update_layout(yaxis_range=[0, max(1, max_y * 1.1)])
    
    return fig


def plot_distribution_with_cdf(data, column_name, title=None, height=500, width=None, sort_by='count'):
    """
    Plot distribution (PDF) and cumulative distribution (CDF) for a categorical column.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing the data
    column_name : str
        Name of the column to analyze
    title : str, optional
        Chart title (defaults to column name)
    height : int, optional
        Chart height in pixels (default: 500)
    width : int, optional
        Chart width in pixels
    sort_by : str, optional
        How to sort the x-axis: 'count' (descending), 'value' (alphabetical), or 'none'
    
    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The generated figure with dual y-axes
    """
    # Count occurrences
    counts = data[column_name].value_counts(dropna=False).reset_index()
    counts.columns = [column_name, 'count']
    
    # Sort based on parameter
    if sort_by == 'count':
        counts = counts.sort_values('count', ascending=False)
    elif sort_by == 'value':
        counts = counts.sort_values(column_name)
    # if 'none', keep value_counts order
    
    # Calculate percentages and cumulative sum
    total = counts['count'].sum()
    counts['percentage'] = (counts['count'] / total) * 100
    counts['cumulative_pct'] = counts['percentage'].cumsum()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart (PDF - percentage)
    fig.add_trace(
        go.Bar(
            x=counts[column_name].astype(str),
            y=counts['percentage'],
            name='Distribution (PDF)',
            text=counts['percentage'],
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_color='#3498db',
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>' +
                         'Count: %{customdata[0]:,}<br>' +
                         'Percentage: %{y:.2f}%<br>' +
                         '<extra></extra>',
            customdata=counts[['count']].values
        ),
        secondary_y=False
    )
    
    # Add line chart (CDF)
    fig.add_trace(
        go.Scatter(
            x=counts[column_name].astype(str),
            y=counts['cumulative_pct'],
            name='Cumulative (CDF)',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3, dash='dot'),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>%{x}</b><br>' +
                         'Cumulative: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Set titles
    title = title or f'Distribution of {column_name}'
    fig.update_layout(
        title=title,
        xaxis_title=column_name,
        hovermode='x unified',
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Percentage (%)", secondary_y=False, range=[0, max(counts['percentage'].max() * 1.1, 1)])
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, range=[0, 105])
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Distribution Summary: {column_name}")
    print(f"{'='*60}")
    print(f"Total observations: {total:,}")
    print(f"Unique values: {len(counts)}")
    print(f"\nTop 5 values:")
    print(counts.head().to_string(index=False))
    print(f"{'='*60}\n")
    
    return fig
