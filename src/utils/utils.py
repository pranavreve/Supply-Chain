import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Set up logging
def setup_logger(log_file=None):
    """
    Set up logger for the project
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger('supply_chain')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger

# Load and save data
def load_data(filepath):
    """
    Load data from file
    
    Parameters
    ----------
    filepath : str
        Path to data file
        
    Returns
    -------
    pandas.DataFrame
        Loaded dataframe
    """
    # Check file extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.csv':
        return pd.read_csv(filepath)
    elif ext.lower() == '.xlsx' or ext.lower() == '.xls':
        return pd.read_excel(filepath)
    elif ext.lower() == '.parquet':
        return pd.read_parquet(filepath)
    elif ext.lower() == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_data(df, filepath, index=False):
    """
    Save data to file
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    filepath : str
        Path to save file
    index : bool, optional
        Whether to save with index
        
    Returns
    -------
    None
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Check file extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.csv':
        df.to_csv(filepath, index=index)
    elif ext.lower() == '.xlsx' or ext.lower() == '.xls':
        df.to_excel(filepath, index=index)
    elif ext.lower() == '.parquet':
        df.to_parquet(filepath, index=index)
    elif ext.lower() == '.json':
        df.to_json(filepath, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Config handling
def load_config(config_path):
    """
    Load configuration from JSON file
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config, config_path):
    """
    Save configuration to JSON file
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save configuration
        
    Returns
    -------
    None
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Date utilities
def get_date_range(start_date, end_date, freq='D'):
    """
    Get date range for analysis
    
    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    freq : str, optional
        Frequency for date range
        
    Returns
    -------
    pandas.DatetimeIndex
        Date range
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def convert_to_fiscal_year(date, fiscal_start_month=4):
    """
    Convert date to fiscal year
    
    Parameters
    ----------
    date : datetime or str
        Date to convert
    fiscal_start_month : int, optional
        Starting month of fiscal year (default is 4 for April)
        
    Returns
    -------
    int
        Fiscal year
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    if date.month >= fiscal_start_month:
        return date.year + 1
    else:
        return date.year

# Statistical utilities
def detect_outliers(series, n_sigmas=3):
    """
    Detect outliers in a series using z-score
    
    Parameters
    ----------
    series : pandas.Series
        Series to detect outliers
    n_sigmas : int, optional
        Number of standard deviations to use as threshold
        
    Returns
    -------
    pandas.Series
        Boolean series indicating outliers
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series(False, index=series.index)
    
    z_scores = (series - mean) / std
    
    return abs(z_scores) > n_sigmas 