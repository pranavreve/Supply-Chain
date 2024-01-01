import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from pmdarima import auto_arima
import logging
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """A class for analyzing and forecasting time series data in supply chains"""
    
    def __init__(self):
        """Initialize the TimeSeriesAnalyzer with default parameters"""
        self.data = None
        self.freq = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.decomposition = None
    
    def load_data(self, data: pd.DataFrame, date_column: str, value_column: str, 
                 freq: str = None, group_columns: List[str] = None) -> None:
        """
        Load time series data into the analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing time series data
        date_column : str
            Name of the column containing datetime values
        value_column : str
            Name of the column containing values to forecast
        freq : str, optional
            Frequency of the time series (e.g., 'D' for daily, 'W' for weekly)
        group_columns : List[str], optional
            List of columns to group by (for multiple time series)
        """
        # Validate data
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not in DataFrame")
        if value_column not in data.columns:
            raise ValueError(f"Value column '{value_column}' not in DataFrame")
        
        # Ensure date column is datetime
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Store data with proper formatting
        if group_columns is not None:
            # For multiple time series, store as a dictionary of DataFrames
            self.data = {}
            for name, group in data.groupby(group_columns):
                # If groupby returns multiple keys, use tuple as key
                if isinstance(name, tuple):
                    key = name
                else:
                    key = (name,)
                    
                # Sort by date and set index
                ts = group.sort_values(date_column).set_index(date_column)[value_column]
                
                # Ensure the index has no duplicates
                if ts.index.has_duplicates:
                    logger.warning(f"Time series for {key} has duplicate dates. Using the mean of duplicate values.")
                    ts = ts.groupby(level=0).mean()
                
                # Set frequency if provided
                if freq is not None:
                    ts = ts.asfreq(freq)
                
                self.data[key] = ts
            
            logger.info(f"Loaded {len(self.data)} time series")
        else:
            # For a single time series
            ts = data.sort_values(date_column).set_index(date_column)[value_column]
            
            # Ensure the index has no duplicates
            if ts.index.has_duplicates:
                logger.warning("Time series has duplicate dates. Using the mean of duplicate values.")
                ts = ts.groupby(level=0).mean()
            
            # Set frequency if provided
            if freq is not None:
                ts = ts.asfreq(freq)
            
            self.data = ts
            logger.info("Loaded time series data")
        
        self.freq = freq
    
    def visualize_time_series(self, keys: List[Tuple] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize the loaded time series data
        
        Parameters:
        -----------
        keys : List[Tuple], optional
            For grouped data, keys to visualize. If None, visualizes all time series
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(self.data, dict):
            # For multiple time series
            if keys is None:
                keys = list(self.data.keys())
            
            for key in keys:
                if key not in self.data:
                    logger.warning(f"Key {key} not found in data")
                    continue
                
                label = " - ".join(str(k) for k in key)
                self.data[key].plot(ax=ax, label=label)
        else:
            # For a single time series
            self.data.plot(ax=ax)
        
        ax.set_title("Time Series Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if isinstance(self.data, dict) and len(keys) > 1:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def perform_decomposition(self, key: Tuple = None, model: str = 'additive', 
                             period: int = None) -> plt.Figure:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Parameters:
        -----------
        key : Tuple, optional
            For grouped data, key to decompose. If None, uses the first time series
        model : str
            Decomposition model type: 'additive' or 'multiplicative'
        period : int, optional
            Period of the seasonal component. If None, inferred from frequency
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with decomposition plots
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select the time series to decompose
        if isinstance(self.data, dict):
            if key is None:
                key = next(iter(self.data.keys()))
            
            if key not in self.data:
                raise ValueError(f"Key {key} not found in data")
            
            ts = self.data[key]
        else:
            ts = self.data
            key = None
        
        # Infer period if not provided
        if period is None:
            if self.freq == 'D':
                period = 7  # Weekly seasonality
            elif self.freq == 'W':
                period = 52  # Yearly seasonality
            elif self.freq == 'M':
                period = 12  # Yearly seasonality
            elif self.freq == 'Q':
                period = 4  # Yearly seasonality
            else:
                period = 12  # Default
                logger.warning(f"Could not infer period from frequency '{self.freq}'. Using default of {period}.")
        
        # Perform decomposition
        try:
            decomposition = seasonal_decompose(ts, model=model, period=period)
            self.decomposition = decomposition
            
            # Create plot
            fig = plt.figure(figsize=(12, 10))
            
            plt.subplot(411)
            plt.plot(ts, label='Original')
            plt.legend(loc='best')
            plt.title('Seasonal Decomposition')
            
            plt.subplot(412)
            plt.plot(decomposition.trend, label='Trend')
            plt.legend(loc='best')
            
            plt.subplot(413)
            plt.plot(decomposition.seasonal, label='Seasonality')
            plt.legend(loc='best')
            
            plt.subplot(414)
            plt.plot(decomposition.resid, label='Residuals')
            plt.legend(loc='best')
            
            plt.tight_layout()
            
            logger.info(f"Performed {model} decomposition with period {period}")
            
            return fig
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise
    
    def check_stationarity(self, key: Tuple = None) -> Dict:
        """
        Check stationarity of time series using Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        key : Tuple, optional
            For grouped data, key to check. If None, uses the first time series
            
        Returns:
        --------
        Dict
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select the time series to test
        if isinstance(self.data, dict):
            if key is None:
                key = next(iter(self.data.keys()))
            
            if key not in self.data:
                raise ValueError(f"Key {key} not found in data")
            
            ts = self.data[key]
        else:
            ts = self.data
            key = None
        
        # Perform ADF test
        result = adfuller(ts.dropna())
        
        # Create results dictionary
        adf_result = {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # Using 5% significance level
        }
        
        # Log results
        logger.info(f"ADF Test - Test Statistic: {adf_result['test_statistic']:.4f}, p-value: {adf_result['p_value']:.4f}")
        logger.info(f"Series {'is' if adf_result['is_stationary'] else 'is not'} stationary")
        
        return adf_result
    
    def plot_acf_pacf(self, key: Tuple = None, lags: int = 40, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
        
        Parameters:
        -----------
        key : Tuple, optional
            For grouped data, key to plot. If None, uses the first time series
        lags : int
            Number of lags to include in plots
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with ACF and PACF plots
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select the time series to plot
        if isinstance(self.data, dict):
            if key is None:
                key = next(iter(self.data.keys()))
            
            if key not in self.data:
                raise ValueError(f"Key {key} not found in data")
            
            ts = self.data[key]
            title_suffix = f" for {' - '.join(str(k) for k in key)}"
        else:
            ts = self.data
            title_suffix = ""
            key = None
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot ACF
        plot_acf(ts.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title(f"Autocorrelation Function{title_suffix}")
        
        # Plot PACF
        plot_pacf(ts.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title(f"Partial Autocorrelation Function{title_suffix}")
        
        plt.tight_layout()
        
        logger.info(f"Plotted ACF and PACF with {lags} lags")
        
        return fig
    
    def fit_arima(self, key: Tuple = None, order: Tuple[int, int, int] = None, 
                seasonal_order: Tuple[int, int, int, int] = None,
                auto: bool = True) -> Dict:
        """
        Fit ARIMA or SARIMA model to time series
        
        Parameters:
        -----------
        key : Tuple, optional
            For grouped data, key to model. If None, uses the first time series
        order : Tuple[int, int, int], optional
            ARIMA order (p, d, q). If None and auto=True, order is determined automatically
        seasonal_order : Tuple[int, int, int, int], optional
            Seasonal order (P, D, Q, s) for SARIMA
        auto : bool
            Whether to use auto_arima to automatically determine model parameters
            
        Returns:
        --------
        Dict
            Dictionary containing model summary and parameters
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select the time series to model
        if isinstance(self.data, dict):
            if key is None:
                key = next(iter(self.data.keys()))
            
            if key not in self.data:
                raise ValueError(f"Key {key} not found in data")
            
            ts = self.data[key]
        else:
            ts = self.data
            key = None
        
        # For storing model information
        model_info = {}
        
        # Automatically determine parameters if requested
        if auto and order is None:
            logger.info("Using auto_arima to determine model parameters")
            auto_model = auto_arima(
                ts, 
                seasonal=seasonal_order is not None,
                stepwise=True,
                trace=True,
                suppress_warnings=True
            )
            
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if hasattr(auto_model, 'seasonal_order') else None
            
            model_info['auto_model'] = auto_model
            logger.info(f"Auto-determined ARIMA order: {order}")
            if seasonal_order:
                logger.info(f"Auto-determined seasonal order: {seasonal_order}")
        
        # Fit the ARIMA model
        try:
            if seasonal_order:
                model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
                model_name = f"SARIMA{order}x{seasonal_order}"
            else:
                model = SARIMAX(ts, order=order)
                model_name = f"ARIMA{order}"
            
            fit_model = model.fit(disp=False)
            
            # Store model information
            model_info['model'] = fit_model
            model_info['order'] = order
            model_info['seasonal_order'] = seasonal_order
            model_info['aic'] = fit_model.aic
            model_info['bic'] = fit_model.bic
            
            # Store in class dictionary
            model_key = key if key is not None else "default"
            self.models[model_key] = model_info
            
            logger.info(f"Fit {model_name} model with AIC: {fit_model.aic:.2f}, BIC: {fit_model.bic:.2f}")
            
            return model_info
        except Exception as e:
            logger.error(f"ARIMA model fitting failed: {str(e)}")
            raise
    
    def fit_prophet(self, key: Tuple = None, yearly_seasonality: bool = True, 
                   weekly_seasonality: bool = True, daily_seasonality: bool = False,
                   changepoint_prior_scale: float = 0.05) -> Dict:
        """
        Fit Prophet model to time series
        
        Parameters:
        -----------
        key : Tuple, optional
            For grouped data, key to model. If None, uses the first time series
        yearly_seasonality : bool
            Whether to include yearly seasonality
        weekly_seasonality : bool
            Whether to include weekly seasonality
        daily_seasonality : bool
            Whether to include daily seasonality
        changepoint_prior_scale : float
            Controls flexibility of the trend
            
        Returns:
        --------
        Dict
            Dictionary containing model and parameters
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select the time series to model
        if isinstance(self.data, dict):
            if key is None:
                key = next(iter(self.data.keys()))
            
            if key not in self.data:
                raise ValueError(f"Key {key} not found in data")
            
            ts = self.data[key]
        else:
            ts = self.data
            key = None
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Fit Prophet model
        try:
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale
            )
            
            fit_model = model.fit(prophet_data)
            
            # Store model information
            model_info = {
                'model': fit_model,
                'parameters': {
                    'yearly_seasonality': yearly_seasonality,
                    'weekly_seasonality': weekly_seasonality,
                    'daily_seasonality': daily_seasonality,
                    'changepoint_prior_scale': changepoint_prior_scale
                }
            }
            
            # Store in class dictionary using 'prophet' prefix to distinguish from ARIMA
            model_key = f"prophet_{key}" if key is not None else "prophet_default"
            self.models[model_key] = model_info
            
            logger.info(f"Fit Prophet model with changepoint_prior_scale: {changepoint_prior_scale}")
            
            return model_info
        except Exception as e:
            logger.error(f"Prophet model fitting failed: {str(e)}")
            raise
    
    def forecast(self, model_key=None, periods: int = 30, alpha: float = 0.05, 
                include_history: bool = True) -> pd.DataFrame:
        """
        Generate forecasts from fitted model
        
        Parameters:
        -----------
        model_key : Tuple or str, optional
            Key for the model to use for forecasting. If None, uses the last fitted model
        periods : int
            Number of periods to forecast
        alpha : float
            Significance level for prediction intervals (default: 0.05 for 95% intervals)
        include_history : bool
            Whether to include historical data in the forecast
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing forecasts and prediction intervals
        """
        if not self.models:
            raise ValueError("No models fitted. Call fit_arima() or fit_prophet() first.")
        
        if model_key is None:
            model_key = next(reversed(self.models.keys()))
        
        if model_key not in self.models:
            raise ValueError(f"Model with key {model_key} not found")
        
        model_info = self.models[model_key]
        model = model_info['model']
        
        # Check if it's a Prophet model (based on model key or model type)
        is_prophet = isinstance(model_key, str) and model_key.startswith('prophet_')
        
        try:
            if is_prophet:
                # Prophet forecast
                future = model.make_future_dataframe(periods=periods, freq=self.freq)
                forecast = model.predict(future)
                
                # Add prediction intervals
                forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                
                # Set the index to datetime for consistency with ARIMA forecasts
                forecast_result = forecast[forecast_columns].copy()
                forecast_result = forecast_result.set_index('ds')
                
                # Rename columns to be consistent with ARIMA forecasts
                forecast_result = forecast_result.rename(columns={
                    'yhat': 'forecast',
                    'yhat_lower': 'lower_bound',
                    'yhat_upper': 'upper_bound'
                })
                
                # Filter to just future data if requested
                if not include_history:
                    if isinstance(self.data, dict):
                        historical_data = self.data[tuple(model_key.split('_')[1:])]
                    else:
                        historical_data = self.data
                    
                    forecast_result = forecast_result.loc[forecast_result.index > historical_data.index[-1]]
            else:
                # ARIMA forecast
                forecast = model.get_forecast(steps=periods)
                
                # Create forecast DataFrame
                forecast_result = pd.DataFrame({
                    'forecast': forecast.predicted_mean,
                    'lower_bound': forecast.conf_int(alpha=alpha).iloc[:, 0],
                    'upper_bound': forecast.conf_int(alpha=alpha).iloc[:, 1]
                })
                
                # Include historical data if requested
                if include_history:
                    if isinstance(self.data, dict):
                        historical_data = self.data[model_key if model_key != "default" else next(iter(self.data.keys()))]
                    else:
                        historical_data = self.data
                    
                    # Create a DataFrame with historical data
                    historical_df = pd.DataFrame({
                        'forecast': historical_data,
                        'lower_bound': np.nan,
                        'upper_bound': np.nan
                    })
                    
                    # Combine historical and forecast data
                    forecast_result = pd.concat([historical_df, forecast_result])
            
            # Store forecast
            self.forecasts[model_key] = forecast_result
            
            logger.info(f"Generated forecast for {periods} periods")
            
            return forecast_result
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            raise
    
    def plot_forecast(self, model_key=None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the forecast from a fitted model
        
        Parameters:
        -----------
        model_key : Tuple or str, optional
            Key for the forecast to plot. If None, uses the last generated forecast
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with forecast plot
        """
        if not self.forecasts:
            raise ValueError("No forecasts generated. Call forecast() first.")
        
        if model_key is None:
            model_key = next(reversed(self.forecasts.keys()))
        
        if model_key not in self.forecasts:
            raise ValueError(f"Forecast with key {model_key} not found")
        
        # Get the forecast data
        forecast = self.forecasts[model_key]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the forecast
        ax.plot(forecast.index, forecast['forecast'], label='Forecast', color='blue')
        
        # Plot the confidence intervals
        if not forecast['lower_bound'].isna().all():
            ax.fill_between(
                forecast.index,
                forecast['lower_bound'],
                forecast['upper_bound'],
                color='blue',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Plot the historical data if available
        non_nan_indices = ~forecast['forecast'].isna()
        if non_nan_indices.any():
            historical_part = forecast.loc[non_nan_indices]
            ax.plot(historical_part.index, historical_part['forecast'], label='Historical', color='black')
        
        # Format the plot
        ax.set_title("Time Series Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
    
    def evaluate_forecast(self, test_data: pd.Series, model_key=None) -> Dict:
        """
        Evaluate forecast performance against test data
        
        Parameters:
        -----------
        test_data : pd.Series
            Actual values to compare forecasts against
        model_key : Tuple or str, optional
            Key for the forecast to evaluate. If None, uses the last generated forecast
            
        Returns:
        --------
        Dict
            Dictionary containing evaluation metrics
        """
        if not self.forecasts:
            raise ValueError("No forecasts generated. Call forecast() first.")
        
        if model_key is None:
            model_key = next(reversed(self.forecasts.keys()))
        
        if model_key not in self.forecasts:
            raise ValueError(f"Forecast with key {model_key} not found")
        
        # Get the forecast data
        forecast = self.forecasts[model_key]
        
        # Align test data with forecast periods
        common_index = forecast.index.intersection(test_data.index)
        
        if len(common_index) == 0:
            raise ValueError("No overlap between forecast and test data")
        
        # Extract aligned data
        actual = test_data.loc[common_index]
        predicted = forecast.loc[common_index, 'forecast']
        
        # Calculate metrics
        n = len(actual)
        residuals = actual - predicted
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / actual)) * 100
        
        # Store metrics
        metrics = {
            'n': n,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        self.metrics[model_key] = metrics
        
        logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def compare_models(self, test_data: pd.Series, model_keys: List = None) -> pd.DataFrame:
        """
        Compare multiple forecast models
        
        Parameters:
        -----------
        test_data : pd.Series
            Actual values to compare forecasts against
        model_keys : List, optional
            List of model keys to compare. If None, compares all available models
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison metrics for each model
        """
        if not self.forecasts:
            raise ValueError("No forecasts generated. Call forecast() first.")
        
        if model_keys is None:
            model_keys = list(self.forecasts.keys())
        
        results = []
        
        for key in model_keys:
            if key not in self.forecasts:
                logger.warning(f"Forecast with key {key} not found. Skipping.")
                continue
            
            try:
                metrics = self.evaluate_forecast(test_data, key)
                
                # Add model key to metrics
                metrics['model_key'] = key
                
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Evaluation failed for model {key}: {str(e)}")
        
        if not results:
            raise ValueError("No models could be evaluated")
        
        # Create DataFrame from results
        comparison = pd.DataFrame(results)
        
        # Sort by RMSE (lower is better)
        comparison = comparison.sort_values('rmse')
        
        return comparison 