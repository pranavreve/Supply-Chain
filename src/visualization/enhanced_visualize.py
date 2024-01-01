import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_time_series_decomposition(decomposition, title='Time Series Decomposition'):
    """
    Plot time series decomposition components
    
    Parameters
    ----------
    decomposition : dict
        Dictionary with decomposition results
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The decomposition plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    components = ['observed', 'trend', 'seasonal', 'residual']
    
    for i, component in enumerate(components):
        axes[i].plot(decomposition[component])
        axes[i].set_title(component.capitalize())
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    
    return fig

def plot_inventory_levels(df, product_col, inventory_col, reorder_point_col, 
                          date_col, products=None):
    """
    Plot inventory levels with reorder points
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with inventory data
    product_col : str
        Column name for product identifier
    inventory_col : str
        Column name for inventory level
    reorder_point_col : str
        Column name for reorder point
    date_col : str
        Column name for date
    products : list, optional
        List of products to plot (defaults to first 5)
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive inventory plot
    """
    if products is None:
        # Default to first 5 products
        products = df[product_col].unique()[:5]
    
    # Filter data
    plot_df = df[df[product_col].isin(products)]
    
    # Create figure
    fig = px.line(
        plot_df, 
        x=date_col, 
        y=inventory_col, 
        color=product_col,
        title='Inventory Levels Over Time'
    )
    
    # Add reorder points
    for product in products:
        product_df = plot_df[plot_df[product_col] == product]
        fig.add_scatter(
            x=product_df[date_col],
            y=product_df[reorder_point_col],
            mode='lines',
            line=dict(dash='dash'),
            name=f'{product} - Reorder Point'
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Inventory Level',
        legend_title='Product',
        height=600
    )
    
    return fig

def plot_supply_chain_metrics_dashboard(df, date_col, metrics, groupby_col=None):
    """
    Create a dashboard of supply chain metrics
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with supply chain data
    date_col : str
        Column name for date
    metrics : dict
        Dictionary mapping metric names to column names
    groupby_col : str, optional
        Column to group by (e.g., region, product)
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive dashboard figure
    """
    # Determine number of metrics
    n_metrics = len(metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=n_metrics, 
        cols=1,
        subplot_titles=list(metrics.keys()),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add traces for each metric
    for i, (metric_name, col_name) in enumerate(metrics.items(), 1):
        if groupby_col is not None:
            # Group by the specified column
            for group in df[groupby_col].unique():
                group_df = df[df[groupby_col] == group]
                fig.add_trace(
                    go.Scatter(
                        x=group_df[date_col],
                        y=group_df[col_name],
                        name=f'{group} - {metric_name}',
                        mode='lines'
                    ),
                    row=i, col=1
                )
        else:
            # No grouping
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[col_name],
                    name=metric_name,
                    mode='lines'
                ),
                row=i, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 * n_metrics,
        title_text='Supply Chain Performance Metrics',
        showlegend=True
    )
    
    return fig

def plot_model_feature_importance(feature_importances, top_n=20, title='Feature Importance'):
    """
    Plot feature importance from machine learning models
    
    Parameters
    ----------
    feature_importances : dict
        Dictionary mapping feature names to importance scores
    top_n : int
        Number of top features to display
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Feature importance plot
    """
    # Convert to DataFrame for sorting
    importance_df = pd.DataFrame({
        'Feature': list(feature_importances.keys()),
        'Importance': list(feature_importances.values())
    })
    
    # Sort and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    return fig

def plot_forecast_with_confidence(forecast, date_col='ds', forecast_col='yhat', 
                                 lower_bound_col='yhat_lower', upper_bound_col='yhat_upper',
                                 actual_col=None, actual_data=None):
    """
    Plot forecast with confidence intervals
    
    Parameters
    ----------
    forecast : pandas.DataFrame
        Prophet forecast results
    date_col : str
        Column name for dates
    forecast_col : str
        Column name for forecast values
    lower_bound_col : str
        Column name for lower confidence bound
    upper_bound_col : str
        Column name for upper confidence bound
    actual_col : str, optional
        Column name for actual values in the forecast DataFrame
    actual_data : pandas.DataFrame, optional
        DataFrame with actual data to plot alongside forecast
        
    Returns
    -------
    plotly.graph_objects.Figure
        Forecast plot with confidence intervals
    """
    # Create figure
    fig = go.Figure()
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast[date_col],
            y=forecast[forecast_col],
            mode='lines',
            name='Forecast',
            line=dict(color='royalblue')
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast[date_col], forecast[date_col].iloc[::-1]]),
            y=pd.concat([forecast[upper_bound_col], forecast[lower_bound_col].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.2)',
            line=dict(color='rgba(65, 105, 225, 0)'),
            name='95% Confidence Interval'
        )
    )
    
    # Add actual data if provided
    if actual_col is not None and actual_col in forecast.columns:
        # Actual data is in the forecast DataFrame
        mask = ~forecast[actual_col].isna()
        fig.add_trace(
            go.Scatter(
                x=forecast.loc[mask, date_col],
                y=forecast.loc[mask, actual_col],
                mode='markers',
                name='Actual',
                marker=dict(color='black')
            )
        )
    elif actual_data is not None and date_col in actual_data.columns:
        # Actual data is provided separately
        actual_col = [col for col in actual_data.columns if col != date_col][0]
        fig.add_trace(
            go.Scatter(
                x=actual_data[date_col],
                y=actual_data[actual_col],
                mode='markers',
                name='Actual',
                marker=dict(color='black')
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Time Series Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Legend',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_anomaly_detection(ts_data, anomalies, date_col, value_col, 
                         title='Anomaly Detection in Time Series'):
    """
    Plot time series with highlighted anomalies
    
    Parameters
    ----------
    ts_data : pandas.Series or pandas.DataFrame
        Time series data with date index
    anomalies : pandas.Series
        Boolean series indicating anomalies
    date_col : str
        Column name for dates (if ts_data is DataFrame)
    value_col : str
        Column name for values (if ts_data is DataFrame)
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Time series plot with anomalies highlighted
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data
    if isinstance(ts_data, pd.DataFrame):
        dates = ts_data[date_col]
        values = ts_data[value_col]
    else:
        # Assume Series with DatetimeIndex
        dates = ts_data.index
        values = ts_data.values
    
    # Plot the time series
    ax.plot(dates, values, label='Original Time Series')
    
    # Find anomaly points
    if len(anomalies) == len(values):
        anomaly_points = anomalies
    else:
        # Assume anomalies is indexed by the same dates
        anomaly_points = anomalies.reindex(dates, fill_value=False)
    
    # Plot anomalies
    anomaly_dates = dates[anomaly_points]
    anomaly_values = values[anomaly_points]
    ax.scatter(anomaly_dates, anomaly_values, color='red', s=50, label='Anomalies')
    
    # Format plot
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    return fig 