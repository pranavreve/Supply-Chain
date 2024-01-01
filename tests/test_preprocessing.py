import sys
import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import unittest

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import clean_data, calculate_lead_times, aggregate_metrics

# Sample data for testing
@pytest.fixture
def sample_data():
    """Create sample data for testing preprocessing functions"""
    np.random.seed(42)
    
    # Create sample dates
    base_date = datetime(2023, 1, 1)
    order_dates = [base_date + timedelta(days=i) for i in range(50)]
    delivery_dates = [order_date + timedelta(days=np.random.randint(1, 10)) 
                     for order_date in order_dates]
    
    # Add some missing values and duplicates
    delivery_dates[5] = None
    order_dates.append(order_dates[10])
    delivery_dates.append(delivery_dates[10])
    
    # Create the dataframe
    df = pd.DataFrame({
        'order_id': range(1000, 1000 + len(order_dates)),
        'product_id': np.random.randint(1, 20, len(order_dates)),
        'order_date': order_dates,
        'delivery_date': delivery_dates,
        'quantity': np.random.randint(1, 100, len(order_dates)),
        'price': np.random.uniform(10, 1000, len(order_dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(order_dates))
    })
    
    # Add some NaN values
    df.loc[3, 'quantity'] = None
    df.loc[7, 'region'] = None
    
    return df

def test_clean_data(sample_data):
    """Test the clean_data function"""
    df = sample_data.copy()
    cleaned_df = clean_data(df)
    
    # Check that duplicates were removed
    assert len(cleaned_df) < len(df)
    assert not cleaned_df.duplicated().any()
    
    # Check that date columns were converted to datetime
    assert pd.api.types.is_datetime64_dtype(cleaned_df['order_date'])
    assert pd.api.types.is_datetime64_dtype(cleaned_df['delivery_date'])
    
    # Check that missing values were handled
    assert not cleaned_df['quantity'].isna().any()
    assert not cleaned_df['region'].isna().any()

def test_calculate_lead_times(sample_data):
    """Test the calculate_lead_times function"""
    df = clean_data(sample_data)
    df_with_lead_times = calculate_lead_times(df)
    
    # Check that lead_time column was added
    assert 'lead_time' in df_with_lead_times.columns
    
    # Check lead time calculation
    for idx, row in df_with_lead_times.iterrows():
        if pd.notna(row['delivery_date']) and pd.notna(row['order_date']):
            expected_lead_time = (row['delivery_date'] - row['order_date']).days
            assert row['lead_time'] == expected_lead_time
    
    # Check that lead_time_error column was added
    assert 'lead_time_error' in df_with_lead_times.columns
    
    # Check region metrics
    if 'mean' in df_with_lead_times.columns:
        for region in df_with_lead_times['region'].unique():
            region_data = df_with_lead_times[df_with_lead_times['region'] == region]
            expected_mean = region_data['lead_time'].mean()
            assert abs(region_data['mean'].iloc[0] - expected_mean) < 1e-10

def test_aggregate_metrics(sample_data):
    """Test the aggregate_metrics function"""
    df = clean_data(sample_data)
    df = calculate_lead_times(df)
    
    # Test daily aggregation
    daily_metrics = aggregate_metrics(df)
    
    # Check that aggregation was performed
    assert len(daily_metrics) <= len(df['order_date'].dt.date.unique())
    
    # Check that expected columns exist
    assert 'lead_time_mean' in daily_metrics.columns
    assert 'quantity_sum' in daily_metrics.columns
    assert 'price_sum' in daily_metrics.columns
    
    # Check aggregation values for first day
    first_day = df['order_date'].dt.date.min()
    first_day_data = df[df['order_date'].dt.date == first_day]
    first_day_metrics = daily_metrics[daily_metrics['order_date'] == first_day]
    
    if not first_day_metrics.empty:
        assert abs(first_day_metrics['lead_time_mean'].iloc[0] - first_day_data['lead_time'].mean()) < 1e-10
        assert first_day_metrics['quantity_sum'].iloc[0] == first_day_data['quantity'].sum()

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'order_id': ['ORD123', 'ORD124', 'ORD125', 'ORD126', 'ORD127', 'ORD128'],
            'product_id': ['P001', 'P002', 'P003', 'P001', 'P002', 'P003'],
            'order_date': [
                '2023-01-01', '2023-01-02', '2023-01-03', 
                '2023-01-04', '2023-01-05', '2023-01-06'
            ],
            'delivery_date': [
                '2023-01-05', '2023-01-08', '2023-01-09', 
                '2023-01-10', '2023-01-12', None
            ],
            'quantity': [5, 10, 15, 20, 25, 30],
            'price': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            'supplier_id': ['S001', 'S002', 'S003', 'S001', 'S002', 'S003'],
            'distributor_id': ['D001', 'D001', 'D002', 'D002', 'D003', 'D003'],
            'retailer_id': ['R001', 'R002', 'R003', 'R001', 'R002', 'R003'],
            'region': ['North', 'South', 'East', 'West', 'North', 'South']
        })
        
        # Add duplicated row
        self.sample_data_with_duplicates = pd.concat([
            self.sample_data,
            pd.DataFrame([self.sample_data.iloc[0].to_dict()])
        ]).reset_index(drop=True)
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicate rows"""
        cleaned_df = clean_data(self.sample_data_with_duplicates)
        self.assertEqual(len(cleaned_df), len(self.sample_data))
    
    def test_clean_data_converts_dates(self):
        """Test that clean_data converts date columns to datetime"""
        cleaned_df = clean_data(self.sample_data)
        
        # Check that date columns are now datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(cleaned_df['order_date']))
        self.assertTrue(pd.api.types.is_datetime64_dtype(cleaned_df['delivery_date']))
    
    def test_calculate_lead_times(self):
        """Test that calculate_lead_times correctly calculates lead times"""
        cleaned_df = clean_data(self.sample_data)
        df_with_lead_times = calculate_lead_times(cleaned_df)
        
        # Check that lead_time column exists
        self.assertIn('lead_time', df_with_lead_times.columns)
        
        # Check calculated lead times for specific rows
        expected_lead_times = [4, 6, 6, 6, 7, np.nan]  # Last one is NaN due to missing delivery_date
        for i, expected_lead_time in enumerate(expected_lead_times):
            if pd.isna(expected_lead_time):
                self.assertTrue(pd.isna(df_with_lead_times.iloc[i]['lead_time']))
            else:
                self.assertEqual(df_with_lead_times.iloc[i]['lead_time'], expected_lead_time)

if __name__ == '__main__':
    # Run tests manually
    test_data = sample_data()
    test_clean_data(test_data)
    test_calculate_lead_times(test_data)
    test_aggregate_metrics(test_data)
    print("All tests passed!")
    unittest.main() 