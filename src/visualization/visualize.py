import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

def set_plotting_style():
    """
    Set consistent style for matplotlib visualizations
    """
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14

def plot_lead_time_distribution(df, lead_time_col='lead_time', 
                               group_col=None, save_path=None):
    """
    Plot distribution of lead times
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with lead time data
    lead_time_col : str
        Column name for lead times
    group_col : str, optional
        Column name for grouping (e.g., 'region')
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    set_plotting_style()
    
    if group_col and group_col in df.columns:
        fig, ax = plt.subplots()
        for group, group_data in df.groupby(group_col):
            sns.kdeplot(group_data[lead_time_col], label=group, ax=ax)
        
        plt.title(f'Lead Time Distribution by {group_col}')
        plt.xlabel('Lead Time (days)')
        plt.ylabel('Density')
        plt.legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        sns.histplot(df[lead_time_col], kde=True, ax=axes[0])
        axes[0].set_title('Lead Time Distribution')
        axes[0].set_xlabel('Lead Time (days)')
        
        # Box plot
        sns.boxplot(y=df[lead_time_col], ax=axes[1])
        axes[1].set_title('Lead Time Box Plot')
        axes[1].set_ylabel('Lead Time (days)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_time_series_plot(df, date_col, value_col, group_col=None,
                          title=None, rolling_window=None, save_path=None):
    """
    Create interactive time series plot using Plotly
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    date_col : str
        Column name for dates
    value_col : str
        Column name for the value to plot
    group_col : str, optional
        Column name for grouping
    title : str, optional
        Plot title
    rolling_window : int, optional
        Window size for moving average
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if group_col and group_col in df.columns:
        fig = px.line(
            df, x=date_col, y=value_col, color=group_col,
            title=title or f'{value_col} Over Time by {group_col}'
        )
        
        if rolling_window:
            # Add rolling average for each group
            for group in df[group_col].unique():
                group_data = df[df[group_col] == group]
                group_data_rolled = group_data[value_col].rolling(window=rolling_window).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=group_data[date_col],
                        y=group_data_rolled,
                        mode='lines',
                        line=dict(width=3, dash='dash'),
                        name=f'{group} ({rolling_window}-day MA)'
                    )
                )
    else:
        fig = px.line(
            df, x=date_col, y=value_col,
            title=title or f'{value_col} Over Time'
        )
        
        if rolling_window:
            # Add rolling average
            rolling_avg = df[value_col].rolling(window=rolling_window).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=rolling_avg,
                    mode='lines',
                    line=dict(width=3, dash='dash'),
                    name=f'{rolling_window}-day Moving Average'
                )
            )
    
    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=value_col,
        legend_title_text='Legend',
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

def create_supply_chain_network(df, source_col, target_col, weight_col=None,
                               title="Supply Chain Network", save_path=None):
    """
    Create network visualization of supply chain
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with network edges
    source_col : str
        Column name for source nodes
    target_col : str
        Column name for target nodes
    weight_col : str, optional
        Column name for edge weights
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Create network graph
    G = nx.DiGraph()
    
    # Add edges from dataframe
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        
        weight = 1.0
        if weight_col and weight_col in df.columns:
            weight = row[weight_col]
        
        if G.has_edge(source, target):
            # If edge exists, update weight
            G[source][target]['weight'] += weight
        else:
            # Add new edge
            G.add_edge(source, target, weight=weight)
    
    # Compute node positions using layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Get edge weights for width
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [3 * (w / max_weight) for w in edge_weights]
    
    # Draw the network components
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(
        G, pos, width=normalized_weights,
        alpha=0.5, edge_color='navy',
        arrowsize=20, arrowstyle='->'
    )
    
    plt.title(title, fontsize=18)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf() 