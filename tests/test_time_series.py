import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.time_series import decompose_time_series, detect_anomalies

class TestTimeSeriesAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create a simple time series DataFrame for testing
        dates = pd.date_range(start='2023-01-01', periods=100)
        trend = np.linspace(10, 30, 100)  # Upward trend
        seasonal = 5 * np.sin(np.linspace(0, 10 * np.pi, 100))  # Seasonal component
        random = np.random.normal(0, 1, 100)  # Random noise
        
        values = trend + seasonal + random
        
        self.time_series_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def test_decompose_time_series(self):
        """Test time series decomposition"""
        # Ensure the dataframe is properly set up
        self.assertEqual(len(self.time_series_df), 100)
        
        # Run decomposition
        decomposition = decompose_time_series(
            self.time_series_df, 
            target_column='value', 
            date_column='date',
            period=10  # Set a small period for testing
        )
        
        # Check that decomposition contains expected components
        self.assertIn('trend', decomposition)
        self.assertIn('seasonal', decomposition)
        self.assertIn('residual', decomposition)
        self.assertIn('observed', decomposition)
        
        # Check lengths of components
        # Note: statsmodels seasonal_decompose drops values at edges
        # So lengths can be smaller than original series
        self.assertTrue(len(decomposition['trend']) >= 90)
        self.assertTrue(len(decomposition['seasonal']) >= 90)
        self.assertTrue(len(decomposition['residual']) >= 90)
        self.assertTrue(len(decomposition['observed']) >= 90)
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Create decomposition with known anomalies
        n_points = 100
        observed = np.random.normal(0, 1, n_points)
        
        # Add some obvious anomalies
        anomaly_indices = [25, 50, 75]
        for idx in anomaly_indices:
            observed[idx] = 20  # Far outside normal range
        
        decomposition = {
            'observed': pd.Series(observed),
            'trend': pd.Series(np.zeros(n_points)),
            'seasonal': pd.Series(np.zeros(n_points)),
            'residual': pd.Series(observed)  # Residual equals observed since trend and seasonal are zero
        }
        
        # Detect anomalies
        anomalies = detect_anomalies(decomposition, threshold=3)
        
        # Check that anomalies were detected at the correct indices
        self.assertEqual(sum(anomalies), len(anomaly_indices))
        for idx in anomaly_indices:
            self.assertTrue(anomalies.iloc[idx])

if __name__ == '__main__':
    unittest.main() 