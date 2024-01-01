import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import prophet

def decompose_time_series(df, target_column, date_column, period=7):
    """
    Decompose time series into trend, seasonal, and residual components
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    target_column : str
        Column name of the metric to decompose
    date_column : str
        Column name containing dates
    period : int
        Period for seasonal decomposition
        
    Returns
    -------
    dict
        Dictionary containing the decomposition results
    """
    # Ensure the data is sorted by date
    df = df.sort_values(by=date_column)
    
    # Set the date as index for decomposition
    df_indexed = df.set_index(date_column)
    
    # Perform decomposition
    decomposition = seasonal_decompose(df_indexed[target_column], period=period)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    }

def detect_anomalies(decomposition, threshold=3):
    """
    Detect anomalies in time series data based on residuals
    
    Parameters
    ----------
    decomposition : dict
        Dictionary with decomposition results
    threshold : float
        Z-score threshold for anomaly detection
        
    Returns
    -------
    pandas.Series
        Boolean series indicating anomalies
    """
    # Calculate z-scores for residuals
    residuals = decomposition['residual'].dropna()
    mean = residuals.mean()
    std = residuals.std()
    z_scores = np.abs((residuals - mean) / std)
    
    # Identify anomalies
    anomalies = z_scores > threshold
    
    return anomalies

def forecast_with_prophet(df, target_column, date_column, forecast_periods=30):
    """
    Generate forecasts using Facebook Prophet
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    target_column : str
        Column name of the metric to forecast
    date_column : str
        Column name containing dates
    forecast_periods : int
        Number of periods to forecast
        
    Returns
    -------
    tuple
        (Prophet model, forecast DataFrame)
    """
    # Prepare data for Prophet
    prophet_df = df[[date_column, target_column]].rename(
        columns={date_column: 'ds', target_column: 'y'}
    )
    
    # Initialize and fit Prophet model
    model = prophet.Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_df)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=forecast_periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast 