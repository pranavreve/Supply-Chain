import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """Time Series Forecaster for supply chain metrics."""
    
    def __init__(self, method='hw', **kwargs):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        method : str
            Forecasting method to use: 'hw' (Holt-Winters), 'arima', or 'sarima'
        kwargs : dict
            Additional parameters for the specific forecasting method
        """
        self.method = method
        self.params = kwargs
        self.model = None
        self.results = None
        self.fitted = False
        
        # Default parameters for each method
        self.default_params = {
            'hw': {
                'trend': 'add',
                'seasonal': 'add',
                'seasonal_periods': 7
            },
            'arima': {
                'order': (1, 1, 1)
            },
            'sarima': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 0, 1, 7)
            }
        }
        
        # Update default parameters with provided parameters
        self._update_params()
    
    def _update_params(self):
        """Update default parameters with provided ones."""
        if self.method in self.default_params:
            for key, value in self.default_params[self.method].items():
                if key not in self.params:
                    self.params[key] = value
    
    def fit(self, data, target_col):
        """
        Fit the time series model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data containing the time series
        target_col : str
            Column name of the target variable
        
        Returns:
        --------
        self : TimeSeriesForecaster
            Returns the fitted instance
        """
        try:
            self.target_col = target_col
            y = data[target_col].values
            
            if self.method == 'hw':
                self.model = ExponentialSmoothing(
                    y,
                    trend=self.params['trend'],
                    seasonal=self.params['seasonal'],
                    seasonal_periods=self.params['seasonal_periods']
                )
                
            elif self.method == 'arima':
                self.model = ARIMA(y, order=self.params['order'])
                
            elif self.method == 'sarima':
                self.model = SARIMAX(
                    y,
                    order=self.params['order'],
                    seasonal_order=self.params['seasonal_order']
                )
                
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.results = self.model.fit()
                
            self.fitted = True
            logger.info(f"Successfully fitted {self.method} model on {target_col}")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting {self.method} model: {str(e)}")
            raise
    
    def predict(self, steps=7):
        """
        Generate forecasts for future periods.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast ahead
            
        Returns:
        --------
        forecast : pandas.Series
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
            
        try:
            if self.method == 'hw':
                forecast = self.results.forecast(steps)
            else:
                forecast = self.results.forecast(steps)
                
            logger.info(f"Generated forecast for {steps} steps ahead")
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data to evaluate the model on
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
            
        try:
            actual = test_data[self.target_col].values
            steps = len(actual)
            forecast = self.predict(steps)
            
            mse = mean_squared_error(actual, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
            logger.info(f"Model evaluation metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def get_confidence_intervals(self, steps=7, alpha=0.05):
        """
        Generate confidence intervals for forecasts.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast ahead
        alpha : float
            Significance level (default: 0.05 for 95% confidence interval)
            
        Returns:
        --------
        lower_bound, upper_bound : tuple of numpy.ndarray
            Lower and upper bounds of the confidence interval
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
            
        try:
            forecast = self.predict(steps)
            
            if self.method in ['arima', 'sarima']:
                # For ARIMA and SARIMA models
                forecast_obj = self.results.get_forecast(steps)
                conf_int = forecast_obj.conf_int(alpha=alpha)
                lower_bound = conf_int[:, 0]
                upper_bound = conf_int[:, 1]
            else:
                # For Holt-Winters, use a simple approximation
                std_error = np.std(self.results.resid)
                z_value = 1.96  # Approximately for 95% confidence
                lower_bound = forecast - z_value * std_error
                upper_bound = forecast + z_value * std_error
            
            return lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise

def split_time_series(data, test_size=0.2, train_size=None):
    """
    Split time series data into training and testing sets.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data to split
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2)
    train_size : float, optional
        Proportion of data to use for training (default: None)
        
    Returns:
    --------
    train_data, test_data : tuple of pandas.DataFrame
        Training and testing datasets
    """
    if train_size is not None:
        train_end = int(len(data) * train_size)
    else:
        train_end = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[train_end:].copy()
    
    return train_data, test_data 