import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean and preprocess the supply chain data
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw supply chain data
        
    Returns
    -------
    pandas.DataFrame
        Cleaned data
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Convert date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    return df

def calculate_lead_times(df):
    """
    Calculate lead times between order placement and fulfillment
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with order data
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with lead time metrics
    """
    # Verify that required date columns exist
    if 'order_date' in df.columns and 'delivery_date' in df.columns:
        # Calculate lead time in days
        df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
        
        # Flag negative lead times as errors
        df['lead_time_error'] = df['lead_time'] < 0
        
        # Calculate additional metrics
        if 'region' in df.columns:
            region_metrics = df.groupby('region')['lead_time'].agg(['mean', 'median', 'std']).reset_index()
            df = df.merge(region_metrics, on='region', suffixes=('', '_region_avg'))
            
            # Calculate z-score for lead time within each region
            df['lead_time_z_score'] = df.apply(
                lambda x: (x['lead_time'] - x['mean']) / x['std'] if x['std'] > 0 else 0, 
                axis=1
            )
            
            # Flag anomalies based on z-score
            df['lead_time_anomaly'] = abs(df['lead_time_z_score']) > 2.5
    
    return df

def aggregate_metrics(df):
    """
    Aggregate supply chain metrics for analysis
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with order and lead time data
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated metrics
    """
    # Daily aggregation
    if 'order_date' in df.columns:
        daily_metrics = df.groupby(df['order_date'].dt.date).agg({
            'lead_time': ['mean', 'median', 'std', 'count'],
            'quantity': ['sum', 'mean'],
            'price': ['sum', 'mean']
        }).reset_index()
        
        # Flatten multi-level columns
        daily_metrics.columns = ['_'.join(col).strip('_') for col in daily_metrics.columns.values]
        
        return daily_metrics
    
    return pd.DataFrame() 