import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx

def extract_datetime_features(df, date_column):
    """
    Extract datetime features from a date column
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with date column
    date_column : str
        Name of the date column
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted datetime features
    """
    df = df.copy()
    
    # Convert to datetime if not already
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract basic date components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    # Extract boolean flags
    df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start
    df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end
    df[f'{date_column}_is_quarter_end'] = df[date_column].dt.is_quarter_end
    
    # Cyclical encoding for day of week and month
    df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
    df[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
    
    return df

def calculate_order_statistics(df, order_id_col, group_cols=None):
    """
    Calculate order statistics for each group
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with order data
    order_id_col : str
        Column name for order ID
    group_cols : list, optional
        List of columns to group by
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with order statistics
    """
    df = df.copy()
    
    if group_cols is None:
        # Default grouping at product level
        if 'product_id' in df.columns:
            group_cols = ['product_id']
        else:
            return df
    
    # Calculate order statistics
    order_stats = df.groupby(group_cols).agg({
        order_id_col: 'count',
        'quantity': ['sum', 'mean', 'std', 'median'] if 'quantity' in df.columns else 'count',
        'price': ['sum', 'mean', 'std', 'median'] if 'price' in df.columns else None
    }).reset_index()
    
    # Flatten multi-level column names
    order_stats.columns = ['_'.join(col).strip('_') for col in order_stats.columns.values]
    
    # Rename count column
    order_stats = order_stats.rename(columns={f'{order_id_col}_count': 'order_count'})
    
    # Merge with original dataframe
    df = df.merge(order_stats, on=group_cols)
    
    return df

def calculate_network_features(df, source_col, target_col, value_col=None):
    """
    Calculate network features for supply chain entities
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with network edges
    source_col : str
        Column name for source nodes (e.g., supplier_id)
    target_col : str
        Column name for target nodes (e.g., distributor_id)
    value_col : str, optional
        Column name for edge values (e.g., quantity)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with network features
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges from dataframe
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        
        weight = 1.0
        if value_col and value_col in df.columns:
            weight = row[value_col]
        
        if G.has_edge(source, target):
            # Update weight
            G[source][target]['weight'] += weight
        else:
            # Add new edge
            G.add_edge(source, target, weight=weight)
    
    # Calculate network metrics
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Add weighted degree if weight is used
    if value_col:
        weighted_in_degree = dict(G.in_degree(weight='weight'))
        weighted_out_degree = dict(G.out_degree(weight='weight'))
    
    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    # Create feature dataframe for sources
    source_features = pd.DataFrame({
        source_col: list(set(df[source_col])),
        f'{source_col}_out_degree': [out_degree.get(node, 0) for node in set(df[source_col])],
        f'{source_col}_betweenness': [betweenness.get(node, 0) for node in set(df[source_col])],
        f'{source_col}_closeness': [closeness.get(node, 0) for node in set(df[source_col])]
    })
    
    if value_col:
        source_features[f'{source_col}_weighted_out'] = [weighted_out_degree.get(node, 0) for node in set(df[source_col])]
    
    # Create feature dataframe for targets
    target_features = pd.DataFrame({
        target_col: list(set(df[target_col])),
        f'{target_col}_in_degree': [in_degree.get(node, 0) for node in set(df[target_col])],
        f'{target_col}_betweenness': [betweenness.get(node, 0) for node in set(df[target_col])],
        f'{target_col}_closeness': [closeness.get(node, 0) for node in set(df[target_col])]
    })
    
    if value_col:
        target_features[f'{target_col}_weighted_in'] = [weighted_in_degree.get(node, 0) for node in set(df[target_col])]
    
    # Merge network features with original dataframe
    df = df.merge(source_features, on=source_col, how='left')
    df = df.merge(target_features, on=target_col, how='left')
    
    return df

def create_lag_features(df, date_col, target_col, group_col=None, lags=[1, 7, 14, 30]):
    """
    Create lagged features for time series analysis
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    date_col : str
        Column name for date
    target_col : str
        Column name for target variable
    group_col : str, optional
        Column name for grouping
    lags : list, optional
        List of lag periods to create
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with lag features
    """
    df = df.copy()
    
    # Ensure date column is datetime and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col])
    
    if group_col and group_col in df.columns:
        # Create lags for each group
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    else:
        # Create lags for entire dataframe
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Calculate rolling statistics
    for window in [7, 14, 30]:
        if window <= max(lags):
            continue
            
        if group_col and group_col in df.columns:
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    
    return df 