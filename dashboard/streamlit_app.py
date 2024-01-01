import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
from io import StringIO
import sys
import os
import json
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

# Add the src directory to the path
sys.path.append(os.path.abspath('../src'))

from models.time_series import decompose_time_series, detect_anomalies, forecast_with_prophet
from models.inventory_optimization import calculate_safety_stock, calculate_reorder_point
from visualization.visualize import plot_inventory_levels, plot_supply_chain_metrics_dashboard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS configuration
def load_aws_config():
    """Load AWS configuration based on environment"""
    env = os.environ.get('ENV', 'dev')
    config_path = f'config/{env}.json'
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {
            'region': 'us-east-1',
            'bucket': f'supply-chain-analytics-{env}-analytics-zone',
            'api_endpoint': f'https://api.example.com/{env}'
        }

# Get AWS configuration
aws_config = load_aws_config()

# Function to load data from S3
@st.cache_data(ttl=3600)
def load_data_from_s3(bucket, key):
    """Load data from S3 bucket"""
    try:
        s3 = boto3.client('s3', region_name=aws_config['region'])
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        st.error(f"Error loading data from S3: {str(e)}")
        return pd.DataFrame()

# Function to generate sample data (for local development)
@st.cache_data
def generate_sample_data():
    """Generate sample data for local development"""
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=365)
    
    # Define parameters
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C']
    suppliers = ['Supplier 1', 'Supplier 2', 'Supplier 3', 'Supplier 4']
    
    # Generate data (same logic as in Dash app)
    # ... [Same data generation logic from the Dash app] ...
    
    data = []
    for date in dates:
        for region in regions:
            for product in products:
                # Basic metrics with some seasonality and trend
                day_of_year = date.dayofyear
                month = date.month
                
                # Add seasonality
                seasonal_factor = np.sin(day_of_year / 365 * 2 * np.pi) * 0.2 + 1
                
                # Add product-specific trend
                if product == 'Product A':
                    trend = 0.001 * (date - pd.Timestamp('2023-01-01')).days
                elif product == 'Product B':
                    trend = -0.0005 * (date - pd.Timestamp('2023-01-01')).days
                else:
                    trend = 0
                
                # Add region-specific effects
                if region == 'North':
                    region_effect = 0.1
                elif region == 'South':
                    region_effect = -0.05
                elif region == 'East':
                    region_effect = 0.15
                else:
                    region_effect = -0.1
                
                # Generate metrics with noise
                base_lead_time = 5 + region_effect
                lead_time = max(1, base_lead_time * seasonal_factor + trend + np.random.normal(0, 0.5))
                
                base_demand = 100 + region_effect * 200
                demand = max(10, base_demand * seasonal_factor + trend * 100 + np.random.normal(0, 10))
                
                base_fulfillment = 0.95 + region_effect * 0.05
                fulfillment_rate = min(1.0, max(0.7, base_fulfillment - trend * 10 + np.random.normal(0, 0.02)))
                
                base_stockout = 0.05 - region_effect * 0.02
                stockout_probability = min(0.3, max(0.01, base_stockout + trend * 5 + np.random.normal(0, 0.01)))
                
                # Inventory levels
                safety_stock = demand * 0.2
                reorder_point = safety_stock + (demand * lead_time / 30)
                inventory_level = max(0, reorder_point * (1.2 + np.sin(day_of_year / 30) * 0.3 + np.random.normal(0, 0.1)))
                
                # Supplier for this product-region combination
                supplier = np.random.choice(suppliers)
                
                # Costs
                unit_cost = {
                    'Product A': 50,
                    'Product B': 75,
                    'Product C': 120
                }[product] * (1 + region_effect * 0.1)
                
                shipping_cost = unit_cost * 0.1 * lead_time / 5
                holding_cost = inventory_level * unit_cost * 0.02
                stockout_cost = demand * stockout_probability * unit_cost * 0.5
                
                # Record
                data.append({
                    'date': date,
                    'region': region,
                    'product': product,
                    'supplier': supplier,
                    'lead_time': lead_time,
                    'demand': demand,
                    'fulfillment_rate': fulfillment_rate,
                    'stockout_probability': stockout_probability,
                    'inventory_level': inventory_level,
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'unit_cost': unit_cost,
                    'shipping_cost': shipping_cost,
                    'holding_cost': holding_cost,
                    'stockout_cost': stockout_cost,
                    'total_cost': shipping_cost + holding_cost + stockout_cost
                })
    
    return pd.DataFrame(data)

# Load data (either from S3 or generate sample)
use_sample_data = os.environ.get('USE_SAMPLE_DATA', 'true').lower() == 'true'

if use_sample_data:
    df = generate_sample_data()
else:
    df = load_data_from_s3(aws_config['bucket'], 'data/supply_chain_data.csv')

# App title and description
st.title('Supply Chain Analytics Dashboard')
st.markdown("""
This dashboard provides comprehensive analytics for supply chain management, 
including inventory optimization, lead time analysis, and cost management.
""")

# Sidebar filters
st.sidebar.header('Filters')

# Date range filter
min_date = df['date'].min()
max_date = df['date'].max()
default_start_date = max_date - timedelta(days=90)
default_end_date = max_date

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [default_start_date, default_end_date],
    min_value=min_date,
    max_value=max_date
)

# Region filter
all_regions = df['region'].unique()
selected_region = st.sidebar.selectbox('Select Region', all_regions)

# Product filter
all_products = df['product'].unique()
selected_product = st.sidebar.selectbox('Select Product', all_products)

# Supplier filter (with "All" option)
all_suppliers = df['supplier'].unique()
supplier_options = ['All'] + list(all_suppliers)
selected_supplier = st.sidebar.selectbox('Select Supplier', supplier_options)

# Apply filters
filtered_df = df[
    (df['date'] >= pd.Timestamp(start_date)) &
    (df['date'] <= pd.Timestamp(end_date)) &
    (df['region'] == selected_region) &
    (df['product'] == selected_product)
]

if selected_supplier != 'All':
    filtered_df = filtered_df[filtered_df['supplier'] == selected_supplier]

# Metric selection
selected_metrics = st.sidebar.multiselect(
    'Select Metrics to Display',
    ['lead_time', 'fulfillment_rate', 'stockout_probability', 'inventory_level'],
    default=['lead_time', 'fulfillment_rate']
)

# Dashboard sections
# KPI metrics
st.subheader('Key Performance Indicators')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Average Lead Time",
        value=f"{filtered_df['lead_time'].mean():.2f} days",
        delta=f"{filtered_df['lead_time'].mean() - df['lead_time'].mean():.2f} days"
    )

with col2:
    st.metric(
        label="Fulfillment Rate",
        value=f"{filtered_df['fulfillment_rate'].mean():.2%}",
        delta=f"{(filtered_df['fulfillment_rate'].mean() - df['fulfillment_rate'].mean()) * 100:.2f}%"
    )

with col3:
    st.metric(
        label="Stockout Probability",
        value=f"{filtered_df['stockout_probability'].mean():.2%}",
        delta=f"{(filtered_df['stockout_probability'].mean() - df['stockout_probability'].mean()) * 100:.2f}%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Total Cost",
        value=f"${filtered_df['total_cost'].sum():,.2f}",
        delta=f"${filtered_df['total_cost'].sum() - df['total_cost'].sum():,.2f}"
    )

# Time Series Analysis Tab
st.subheader('Time Series Analysis')

if not selected_metrics:
    st.warning('Please select at least one metric to display.')
else:
    # Create subplots for selected metrics
    fig = make_subplots(
        rows=len(selected_metrics), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[m.replace('_', ' ').title() for m in selected_metrics]
    )
    
    # Add traces for each metric
    for i, metric in enumerate(selected_metrics, 1):
        fig.add_trace(
            go.Scatter(
                x=filtered_df['date'],
                y=filtered_df[metric],
                mode='lines',
                name=metric.replace('_', ' ').title()
            ),
            row=i, col=1
        )
        
        # Add moving average
        ma_window = 7
        if len(filtered_df) > ma_window:
            ma = filtered_df[metric].rolling(window=ma_window).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['date'],
                    y=ma,
                    mode='lines',
                    line=dict(dash='dash'),
                    name=f"{metric.replace('_', ' ').title()} (7-day MA)"
                ),
                row=i, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 * len(selected_metrics),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Inventory Management Tab
st.subheader('Inventory Management')

inventory_fig = go.Figure()

# Add inventory level
inventory_fig.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['inventory_level'],
        mode='lines',
        name='Inventory Level',
        line=dict(color='royalblue')
    )
)

# Add reorder point
inventory_fig.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['reorder_point'],
        mode='lines',
        name='Reorder Point',
        line=dict(color='firebrick', dash='dash')
    )
)

# Add safety stock
inventory_fig.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['safety_stock'],
        mode='lines',
        name='Safety Stock',
        line=dict(color='green', dash='dot')
    )
)

# Add demand
inventory_fig.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['demand'],
        mode='lines',
        name='Demand',
        line=dict(color='purple', dash='dashdot')
    )
)

# Update layout
inventory_fig.update_layout(
    title=f"Inventory Management - {selected_product} in {selected_region}",
    xaxis_title="Date",
    yaxis_title="Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600
)

st.plotly_chart(inventory_fig, use_container_width=True)

# Cost Analysis Tab
st.subheader('Cost Analysis')

# Resample data monthly for cost breakdown
monthly_df = filtered_df.set_index('date').resample('M').sum().reset_index()

cost_fig = go.Figure()

# Add cost components
cost_fig.add_trace(
    go.Bar(
        x=monthly_df['date'],
        y=monthly_df['shipping_cost'],
        name='Shipping Cost',
        marker_color='#4285F4'
    )
)

cost_fig.add_trace(
    go.Bar(
        x=monthly_df['date'],
        y=monthly_df['holding_cost'],
        name='Holding Cost',
        marker_color='#EA4335'
    )
)

cost_fig.add_trace(
    go.Bar(
        x=monthly_df['date'],
        y=monthly_df['stockout_cost'],
        name='Stockout Cost',
        marker_color='#FBBC05'
    )
)

# Add total cost line
cost_fig.add_trace(
    go.Scatter(
        x=monthly_df['date'],
        y=monthly_df['total_cost'],
        mode='lines+markers',
        name='Total Cost',
        line=dict(color='black', width=2)
    )
)

# Update layout
cost_fig.update_layout(
    title=f"Cost Analysis - {selected_product} in {selected_region} (Monthly)",
    xaxis_title="Month",
    yaxis_title="Cost ($)",
    barmode='stack',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600
)

st.plotly_chart(cost_fig, use_container_width=True)

# Forecasting Section
st.subheader('Forecast & Predictions')

forecast_tab1, forecast_tab2 = st.tabs(["Demand Forecast", "Lead Time Forecast"])

with forecast_tab1:
    if st.button('Generate Demand Forecast'):
        st.info('Generating forecast...')
        
        # In a real implementation, use Prophet or another forecasting model
        # Here we use a simple moving average as a placeholder
        if len(filtered_df) > 0:
            # Use 80% of data for training
            cutoff_idx = int(len(filtered_df) * 0.8)
            train_df = filtered_df.iloc[:cutoff_idx]
            test_df = filtered_df.iloc[cutoff_idx:]
            
            # Simple forecast
            window = 14
            if len(train_df) > window:
                forecast_values = train_df['demand'].rolling(window=window).mean().iloc[-1]
                forecast_dates = test_df['date']
                
                # Create forecast figure
                forecast_fig = go.Figure()
                
                # Add historical data
                forecast_fig.add_trace(
                    go.Scatter(
                        x=train_df['date'],
                        y=train_df['demand'],
                        mode='lines',
                        name='Historical Demand',
                        line=dict(color='blue')
                    )
                )
                
                # Add test data
                forecast_fig.add_trace(
                    go.Scatter(
                        x=test_df['date'],
                        y=test_df['demand'],
                        mode='lines',
                        name='Actual Demand',
                        line=dict(color='green')
                    )
                )
                
                # Add forecast
                forecast_y = [forecast_values] * len(forecast_dates)
                forecast_fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_y,
                        mode='lines',
                        name='Demand Forecast',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Update layout
                forecast_fig.update_layout(
                    title=f"Demand Forecast for {selected_product} in {selected_region}",
                    xaxis_title="Date",
                    yaxis_title="Demand (Units)",
                    height=500
                )
                
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Calculate forecast accuracy
                mape = np.mean(np.abs((test_df['demand'].values - forecast_y) / test_df['demand'].values)) * 100
                st.metric("Forecast Accuracy", f"{100-mape:.2f}%")
                st.text(f"Forecast horizon: {len(test_df)} days")
            else:
                st.warning("Insufficient data for forecasting")
        else:
            st.warning("No data available for the selected filters")

with forecast_tab2:
    if st.button('Generate Lead Time Forecast'):
        st.info('Generating lead time forecast...')
        
        # Similar forecasting logic for lead times
        # ... [Similar forecasting logic as above, but for lead_time instead of demand] ...
        if len(filtered_df) > 0:
            # Use 80% of data for training
            cutoff_idx = int(len(filtered_df) * 0.8)
            train_df = filtered_df.iloc[:cutoff_idx]
            test_df = filtered_df.iloc[cutoff_idx:]
            
            # Simple forecast
            window = 14
            if len(train_df) > window:
                forecast_values = train_df['lead_time'].rolling(window=window).mean().iloc[-1]
                forecast_dates = test_df['date']
                
                # Create forecast figure
                forecast_fig = go.Figure()
                
                # Add historical data
                forecast_fig.add_trace(
                    go.Scatter(
                        x=train_df['date'],
                        y=train_df['lead_time'],
                        mode='lines',
                        name='Historical Lead Time',
                        line=dict(color='blue')
                    )
                )
                
                # Add test data
                forecast_fig.add_trace(
                    go.Scatter(
                        x=test_df['date'],
                        y=test_df['lead_time'],
                        mode='lines',
                        name='Actual Lead Time',
                        line=dict(color='green')
                    )
                )
                
                # Add forecast
                forecast_y = [forecast_values] * len(forecast_dates)
                forecast_fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_y,
                        mode='lines',
                        name='Lead Time Forecast',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Update layout
                forecast_fig.update_layout(
                    title=f"Lead Time Forecast for {selected_product} in {selected_region}",
                    xaxis_title="Date",
                    yaxis_title="Lead Time (Days)",
                    height=500
                )
                
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Calculate forecast accuracy
                mape = np.mean(np.abs((test_df['lead_time'].values - forecast_y) / test_df['lead_time'].values)) * 100
                st.metric("Forecast Accuracy", f"{100-mape:.2f}%")
                st.text(f"Forecast horizon: {len(test_df)} days")
            else:
                st.warning("Insufficient data for forecasting")
        else:
            st.warning("No data available for the selected filters")

# Network Analysis Tab
st.subheader('Network Analysis')

# Create a sample network visualization
with st.expander("Supply Chain Network Visualization"):
    st.info("Network visualization would be rendered here in production version")
    st.image("https://miro.medium.com/max/1400/1*CpyUNiUVNT-gJJPI9HcwCw.png", caption="Example Supply Chain Network")
    
    st.markdown("""
    The network visualization shows the connections between suppliers, distributors, and retailers.
    Node size indicates importance (centrality) in the network, while edge thickness represents volume.
    """)
    
    # Add network metrics
    st.subheader("Network Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Network Density", "0.42")
    with col2:
        st.metric("Average Path Length", "2.3")
    with col3:
        st.metric("Clustering Coefficient", "0.65")

# Optimization Section
st.subheader('Inventory Optimization')

with st.expander("Optimal Order Quantities"):
    # Sample optimization results
    optimization_data = {
        'Product': [selected_product],
        'EOQ (Units)': [np.round(np.sqrt(2 * filtered_df['demand'].mean() * 100 / (0.2 * filtered_df['unit_cost'].mean())))],
        'Reorder Point': [np.round(filtered_df['reorder_point'].mean())],
        'Safety Stock': [np.round(filtered_df['safety_stock'].mean())],
        'Annual Holding Cost': [f"${filtered_df['holding_cost'].sum():,.2f}"],
        'Annual Stockout Cost': [f"${filtered_df['stockout_cost'].sum():,.2f}"]
    }
    
    st.dataframe(pd.DataFrame(optimization_data), use_container_width=True)
    
    st.markdown("""
    The optimization model balances holding costs against stockout costs to determine
    the most cost-effective inventory policy. Economic Order Quantity (EOQ) represents
    the optimal order size that minimizes total inventory costs.
    """)
    
    if st.button('Run Optimization Model'):
        st.success("Optimization completed!")
        
        # In a real implementation, this would run the actual optimization model
        st.markdown("""
        **Recommendations:**
        
        1. Increase safety stock for Product A in the North region by 15%
        2. Reduce lead time by switching to Supplier 2 for East region distribution
        3. Implement more frequent, smaller orders for Product C to reduce holding costs
        """)

# Data Quality Tab
st.subheader('Data Quality Monitoring')

with st.expander("Data Quality Metrics"):
    # Sample data quality metrics
    quality_data = {
        'Metric': ['Completeness', 'Timeliness', 'Accuracy', 'Consistency'],
        'Score': [0.98, 0.92, 0.95, 0.97],
        'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good']
    }
    
    st.dataframe(pd.DataFrame(quality_data), use_container_width=True)
    
    # Data quality trends
    quality_trend_data = {
        'Date': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='W'),
        'Completeness': np.random.uniform(0.95, 0.99, 10),
        'Timeliness': np.random.uniform(0.85, 0.95, 10),
        'Accuracy': np.random.uniform(0.9, 0.98, 10),
        'Consistency': np.random.uniform(0.92, 0.99, 10)
    }
    
    quality_trend_df = pd.DataFrame(quality_trend_data)
    
    st.line_chart(quality_trend_df.set_index('Date')[['Completeness', 'Timeliness', 'Accuracy', 'Consistency']])
    
    st.markdown("""
    Data quality is monitored across four dimensions:
    - **Completeness**: Percentage of non-null values
    - **Timeliness**: Percentage of data received within expected timeframe
    - **Accuracy**: Percentage of values within expected ranges
    - **Consistency**: Percentage of values consistent with business rules
    """)

# Documentation and Help
with st.sidebar.expander("Documentation"):
    st.markdown("""
    ### How to use this dashboard
    
    1. Use the filters in the sidebar to select specific regions, products, and date ranges
    2. The KPI section provides an overview of key metrics
    3. Time Series Analysis shows trends over time
    4. Inventory Management displays inventory levels against reorder points
    5. Cost Analysis breaks down different cost components
    6. Forecasting tools help predict future demand and lead times
    7. Network Analysis visualizes the supply chain structure
    8. Optimization tools provide recommendations for inventory policy
    
    For more information, see the [user guide](https://example.com/guide).
    """)

# Footer
st.markdown("""
---
Supply Chain Analytics Dashboard | Data last updated: 2024-02-28 | Version 1.0
""")

# Define data loading function
def load_or_generate_data():
    """Load data from CSV files or generate sample data if files don't exist."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(data_path, exist_ok=True)
    
    files = {
        'orders': os.path.join(data_path, 'orders.csv'),
        'inventory': os.path.join(data_path, 'inventory.csv'),
        'shipments': os.path.join(data_path, 'shipments.csv'),
        'suppliers': os.path.join(data_path, 'suppliers.csv')
    }
    
    # Check if all files exist
    if all(os.path.exists(file) for file in files.values()):
        logger.info("Loading data from CSV files...")
        orders = pd.read_csv(files['orders'], parse_dates=['order_date', 'due_date', 'ship_date'])
        inventory = pd.read_csv(files['inventory'], parse_dates=['date'])
        shipments = pd.read_csv(files['shipments'], parse_dates=['ship_date', 'delivery_date'])
        suppliers = pd.read_csv(files['suppliers'])
        
        return orders, inventory, shipments, suppliers
    
    # Generate sample data
    logger.info("Generating sample data...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Regions, products, and suppliers
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
    products = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys']
    supplier_names = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
    
    # Create suppliers dataframe
    suppliers = pd.DataFrame({
        'supplier_id': range(1, len(supplier_names) + 1),
        'supplier_name': supplier_names,
        'reliability_score': np.random.uniform(0.7, 0.99, len(supplier_names)),
        'lead_time_days': np.random.randint(2, 20, len(supplier_names)),
        'cost_per_unit': np.random.uniform(50, 200, len(supplier_names))
    })
    
    # Create orders dataframe
    num_orders = 5000
    order_ids = range(1, num_orders + 1)
    order_dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(num_orders)]
    lead_times = np.random.randint(1, 30, num_orders)
    
    orders = pd.DataFrame({
        'order_id': order_ids,
        'order_date': order_dates,
        'product': np.random.choice(products, num_orders),
        'region': np.random.choice(regions, num_orders),
        'supplier_id': np.random.choice(suppliers['supplier_id'], num_orders),
        'quantity': np.random.randint(10, 1000, num_orders),
        'price_per_unit': np.random.uniform(100, 500, num_orders)
    })
    
    # Add due dates and ship dates
    orders['due_date'] = orders.apply(lambda row: row['order_date'] + timedelta(days=np.random.randint(5, 30)), axis=1)
    
    # Some orders are shipped on time, some are delayed, some are not shipped yet
    ship_delay = np.random.choice([0, 1, 2, 5, 10, None], num_orders, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    
    orders['ship_date'] = orders.apply(
        lambda row: None if ship_delay[row.name] is None else row['due_date'] + timedelta(days=ship_delay[row.name]), 
        axis=1
    )
    
    # Calculate order status
    current_date = end_date
    orders['status'] = orders.apply(
        lambda row: 'Delivered' if row['ship_date'] is not None 
                    else ('In Transit' if row['order_date'] <= current_date else 'Pending'),
        axis=1
    )
    
    # Create inventory dataframe
    inventory_entries = []
    for product in products:
        for region in regions:
            # Create a timeseries with trend and seasonality
            for date in date_range:
                # Base quantity with trend
                base_quantity = 500 + (date - start_date).days * 0.5
                
                # Add seasonality (monthly cycle)
                seasonality = 100 * np.sin(2 * np.pi * date.day / 30)
                
                # Add some noise
                noise = np.random.normal(0, 30)
                
                quantity = max(0, base_quantity + seasonality + noise)
                
                inventory_entries.append({
                    'date': date,
                    'product': product,
                    'region': region,
                    'quantity': int(quantity),
                    'reorder_point': 200,
                    'reorder_quantity': 500
                })
    
    inventory = pd.DataFrame(inventory_entries)
    
    # Create shipments dataframe
    shipment_entries = []
    for idx, order in orders.iterrows():
        if order['ship_date'] is not None:
            transit_time = np.random.randint(1, 10)
            delivery_date = order['ship_date'] + timedelta(days=transit_time)
            
            shipment_entries.append({
                'shipment_id': idx + 1,
                'order_id': order['order_id'],
                'ship_date': order['ship_date'],
                'delivery_date': delivery_date,
                'carrier': np.random.choice(['FedEx', 'UPS', 'DHL', 'USPS', 'Amazon Logistics']),
                'status': 'Delivered' if delivery_date <= current_date else 'In Transit',
                'tracking_number': f'TRK{np.random.randint(1000000, 9999999)}',
                'shipping_cost': order['quantity'] * np.random.uniform(5, 15)
            })
    
    shipments = pd.DataFrame(shipment_entries)
    
    # Save data to CSV
    orders.to_csv(files['orders'], index=False)
    inventory.to_csv(files['inventory'], index=False)
    shipments.to_csv(files['shipments'], index=False)
    suppliers.to_csv(files['suppliers'], index=False)
    
    return orders, inventory, shipments, suppliers

# Load data
with st.spinner('Loading data...'):
    orders, inventory, shipments, suppliers = load_or_generate_data()
    
# Cache data processing functions
@st.cache_data
def get_filtered_data(region_filter, product_filter, supplier_filter):
    """Filter data based on selections"""
    filtered_orders = orders
    
    if region_filter != 'All':
        filtered_orders = filtered_orders[filtered_orders['region'] == region_filter]
    
    if product_filter != 'All':
        filtered_orders = filtered_orders[filtered_orders['product'] == product_filter]
    
    if supplier_filter != 'All':
        filtered_orders = filtered_orders[filtered_orders['supplier_id'] == int(supplier_filter.split(' - ')[0])]
        
    # Get related inventory data
    filtered_inventory = inventory
    if region_filter != 'All':
        filtered_inventory = filtered_inventory[filtered_inventory['region'] == region_filter]
    
    if product_filter != 'All':
        filtered_inventory = filtered_inventory[filtered_inventory['product'] == product_filter]
    
    # Get related shipment data
    filtered_shipments = shipments[shipments['order_id'].isin(filtered_orders['order_id'])]
    
    return filtered_orders, filtered_inventory, filtered_shipments

@st.cache_data
def calculate_kpis(filtered_orders, filtered_inventory, filtered_shipments):
    """Calculate KPIs based on filtered data"""
    kpis = {}
    
    # Order KPIs
    kpis['total_orders'] = len(filtered_orders)
    kpis['total_revenue'] = (filtered_orders['quantity'] * filtered_orders['price_per_unit']).sum()
    
    # On-time delivery rate
    delivered_orders = filtered_orders[filtered_orders['ship_date'].notna()]
    if len(delivered_orders) > 0:
        on_time_orders = delivered_orders[delivered_orders['ship_date'] <= delivered_orders['due_date']]
        kpis['on_time_delivery_rate'] = len(on_time_orders) / len(delivered_orders) * 100
    else:
        kpis['on_time_delivery_rate'] = 0
    
    # Order fulfillment rate
    if len(filtered_orders) > 0:
        kpis['order_fulfillment_rate'] = len(filtered_orders[filtered_orders['status'] == 'Delivered']) / len(filtered_orders) * 100
    else:
        kpis['order_fulfillment_rate'] = 0
    
    # Inventory KPIs
    current_inventory = filtered_inventory.groupby('date').sum().reset_index()
    if not current_inventory.empty:
        kpis['inventory_turnover'] = (filtered_orders['quantity'].sum() / current_inventory['quantity'].mean() * 365 / 
                                     (filtered_inventory['date'].max() - filtered_inventory['date'].min()).days)
        
        # Days of supply
        kpis['days_of_supply'] = current_inventory['quantity'].mean() / (filtered_orders['quantity'].sum() / 
                                 (filtered_orders['order_date'].max() - filtered_orders['order_date'].min()).days)
    else:
        kpis['inventory_turnover'] = 0
        kpis['days_of_supply'] = 0
    
    # Calculate average lead time
    if len(filtered_shipments) > 0:
        lead_times = (filtered_shipments['delivery_date'] - filtered_shipments['ship_date']).dt.days
        kpis['avg_lead_time'] = lead_times.mean()
    else:
        kpis['avg_lead_time'] = 0
    
    return kpis

@st.cache_data
def generate_forecast(filtered_inventory, product, region, periods=30):
    """Generate forecast using exponential smoothing"""
    if filtered_inventory.empty:
        return pd.DataFrame()
    
    # Filter for the specific product and region
    prod_reg_inventory = filtered_inventory[
        (filtered_inventory['product'] == product) & 
        (filtered_inventory['region'] == region)
    ]
    
    if prod_reg_inventory.empty:
        return pd.DataFrame()
    
    # Prepare time series data
    ts_data = prod_reg_inventory.sort_values('date').set_index('date')['quantity']
    
    # Apply exponential smoothing
    try:
        model = ExponentialSmoothing(
            ts_data, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=30
        ).fit()
        
        # Generate forecast
        forecast = model.forecast(periods)
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D'),
            'quantity': forecast.values,
            'type': 'Forecast'
        })
        
        # Prepare historical data
        historical_df = pd.DataFrame({
            'date': ts_data.index,
            'quantity': ts_data.values,
            'type': 'Historical'
        })
        
        # Combine historical and forecast data
        result_df = pd.concat([historical_df, forecast_df])
        
        return result_df
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        return pd.DataFrame()

# Dashboard structure
def main():
    # Sidebar filters
    st.sidebar.title("Supply Chain Analytics")
    
    st.sidebar.header("Filters")
    
    # Get unique values for filters
    regions = ['All'] + sorted(orders['region'].unique().tolist())
    products = ['All'] + sorted(orders['product'].unique().tolist())
    
    # Format supplier options with ID and name
    supplier_options = ['All'] + [f"{supplier['supplier_id']} - {supplier['supplier_name']}" 
                                for _, supplier in suppliers.iterrows()]
    
    # Filter selections
    region_filter = st.sidebar.selectbox("Region", regions)
    product_filter = st.sidebar.selectbox("Product", products)
    supplier_filter = st.sidebar.selectbox("Supplier", supplier_options)
    
    # Apply button
    if st.sidebar.button("Apply Filters"):
        st.experimental_rerun()
    
    # Get filtered data
    filtered_orders, filtered_inventory, filtered_shipments = get_filtered_data(
        region_filter, product_filter, supplier_filter
    )
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_orders, filtered_inventory, filtered_shipments)
    
    # Main dashboard area
    st.title("Supply Chain Analytics Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Orders", f"{kpis['total_orders']:,}")
    col2.metric("Total Revenue", f"${kpis['total_revenue']:,.2f}")
    col3.metric("On-Time Delivery", f"{kpis['on_time_delivery_rate']:.1f}%")
    col4.metric("Order Fulfillment", f"{kpis['order_fulfillment_rate']:.1f}%")

    # Second row of KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Inventory Turnover", f"{kpis['inventory_turnover']:.2f}")
    col2.metric("Days of Supply", f"{kpis['days_of_supply']:.1f}")
    col3.metric("Average Lead Time", f"{kpis['avg_lead_time']:.1f} days")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Orders Analysis", "Inventory Management", "Supplier Performance", "Forecasting"])
    
    with tab1:
        st.header("Orders Analysis")
        
        # Order status distribution
        if not filtered_orders.empty:
            # Order status pie chart
            status_counts = filtered_orders['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            fig = px.pie(
                status_counts, 
                values='Count', 
                names='Status', 
                title='Order Status Distribution',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Orders over time chart
            orders_by_date = filtered_orders.groupby('order_date').size().reset_index(name='count')
            orders_by_date = orders_by_date.sort_values('order_date')
            
            fig = px.line(
                orders_by_date, 
                x='order_date', 
                y='count',
                title='Orders Over Time',
                labels={'count': 'Number of Orders', 'order_date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Order value by region/product 
            if region_filter == 'All':
                order_value_by_region = filtered_orders.groupby('region').apply(
                    lambda x: (x['quantity'] * x['price_per_unit']).sum()
                ).reset_index(name='value')
                
                fig = px.bar(
                    order_value_by_region, 
                    x='region', 
                    y='value',
                    title='Order Value by Region',
                    labels={'value': 'Total Value ($)', 'region': 'Region'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if product_filter == 'All':
                order_value_by_product = filtered_orders.groupby('product').apply(
                    lambda x: (x['quantity'] * x['price_per_unit']).sum()
                ).reset_index(name='value')
                
                fig = px.bar(
                    order_value_by_product, 
                    x='product', 
                    y='value',
                    title='Order Value by Product',
                    labels={'value': 'Total Value ($)', 'product': 'Product'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No order data available for the selected filters.")
    
    with tab2:
        st.header("Inventory Management")
        
        if not filtered_inventory.empty:
            # Inventory over time
            inventory_over_time = filtered_inventory.groupby('date')['quantity'].sum().reset_index()
            
            fig = px.line(
                inventory_over_time, 
                x='date', 
                y='quantity',
                title='Inventory Levels Over Time',
                labels={'quantity': 'Quantity', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Current inventory by product
            if region_filter != 'All' and product_filter == 'All':
                latest_date = filtered_inventory['date'].max()
                current_inv_by_product = filtered_inventory[filtered_inventory['date'] == latest_date].groupby('product')['quantity'].sum().reset_index()
                
                fig = px.bar(
                    current_inv_by_product, 
                    x='product', 
                    y='quantity',
                    title=f'Current Inventory by Product (in {region_filter})',
                    labels={'quantity': 'Quantity', 'product': 'Product'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Current inventory by region
            if product_filter != 'All' and region_filter == 'All':
                latest_date = filtered_inventory['date'].max()
                current_inv_by_region = filtered_inventory[filtered_inventory['date'] == latest_date].groupby('region')['quantity'].sum().reset_index()
                
                fig = px.bar(
                    current_inv_by_region, 
                    x='region', 
                    y='quantity',
                    title=f'Current Inventory by Region (for {product_filter})',
                    labels={'quantity': 'Quantity', 'region': 'Region'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Inventory heatmap by product and region
            if product_filter == 'All' and region_filter == 'All':
                latest_date = filtered_inventory['date'].max()
                inv_heatmap = filtered_inventory[filtered_inventory['date'] == latest_date].pivot_table(
                    index='product', 
                    columns='region', 
                    values='quantity',
                    aggfunc='sum'
                ).fillna(0)
                
                fig = px.imshow(
                    inv_heatmap,
                    labels=dict(x="Region", y="Product", color="Quantity"),
                    title="Current Inventory Levels by Product and Region",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No inventory data available for the selected filters.")
    
    with tab3:
        st.header("Supplier Performance")
        
        if not filtered_orders.empty:
            # Add supplier names to orders
            orders_with_suppliers = pd.merge(
                filtered_orders,
                suppliers[['supplier_id', 'supplier_name', 'reliability_score']],
                on='supplier_id'
            )
            
            # Delivery performance by supplier
            supplier_performance = orders_with_suppliers[orders_with_suppliers['ship_date'].notna()].groupby('supplier_name').apply(
                lambda x: (len(x[x['ship_date'] <= x['due_date']]) / len(x)) * 100
            ).reset_index(name='on_time_percentage')
            
            if not supplier_performance.empty:
                fig = px.bar(
                    supplier_performance,
                    x='supplier_name',
                    y='on_time_percentage',
                    title='On-Time Delivery Performance by Supplier',
                    labels={'on_time_percentage': 'On-Time Delivery %', 'supplier_name': 'Supplier'},
                    color='on_time_percentage',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a scatter plot comparing reliability score to actual on-time delivery
                supplier_comparison = pd.merge(
                    supplier_performance,
                    suppliers[['supplier_name', 'reliability_score']],
                    on='supplier_name'
                )
                
                supplier_comparison['reliability_score'] = supplier_comparison['reliability_score'] * 100
                
                fig = px.scatter(
                    supplier_comparison,
                    x='reliability_score',
                    y='on_time_percentage',
                    title='Supplier Reliability Score vs. Actual On-Time Performance',
                    labels={
                        'reliability_score': 'Reliability Score (%)', 
                        'on_time_percentage': 'Actual On-Time Delivery (%)'
                    },
                    text='supplier_name',
                    size_max=60
                )
                
                # Add reference line (y=x)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Correlation'
                    )
                )
                
                fig.update_traces(textposition='top center')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Order quantity by supplier
            orders_by_supplier = orders_with_suppliers.groupby('supplier_name')['quantity'].sum().reset_index()
            
            fig = px.pie(
                orders_by_supplier,
                values='quantity',
                names='supplier_name',
                title='Order Quantity by Supplier',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No order data available for the selected filters.")
    
    with tab4:
        st.header("Inventory Forecasting")
        
        if region_filter != 'All' and product_filter != 'All':
            # Generate forecast for the selected product and region
            forecast_data = generate_forecast(filtered_inventory, product_filter, region_filter)
            
            if not forecast_data.empty:
                # Plot actual vs forecast
                fig = px.line(
                    forecast_data,
                    x='date',
                    y='quantity',
                    color='type',
                    title=f'Inventory Forecast for {product_filter} in {region_filter}',
                    labels={'quantity': 'Quantity', 'date': 'Date'},
                    color_discrete_sequence=['blue', 'red']
                )
                
                # Add reorder point line
                reorder_point = filtered_inventory[
                    (filtered_inventory['product'] == product_filter) & 
                    (filtered_inventory['region'] == region_filter)
                ]['reorder_point'].iloc[0]
                
                fig.add_hline(
                    y=reorder_point,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Reorder Point"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate days until reorder point
                forecast_below_reorder = forecast_data[
                    (forecast_data['type'] == 'Forecast') & 
                    (forecast_data['quantity'] < reorder_point)
                ]
                
                if not forecast_below_reorder.empty:
                    days_until_reorder = (forecast_below_reorder['date'].min() - pd.Timestamp.now().normalize()).days
                    st.warning(f"⚠️ Inventory projected to reach reorder point in {days_until_reorder} days")
                else:
                    st.success("✅ Inventory levels sufficient for the forecast period")
            else:
                st.write("Not enough data to generate forecast for this product and region.")
        else:
            st.info("Please select a specific product and region to view inventory forecasts")
    
    # Raw data section (collapsible)
    with st.expander("View Raw Data"):
        st.subheader("Filtered Orders Data")
        st.dataframe(filtered_orders, use_container_width=True)
        
        st.subheader("Filtered Inventory Data")
        st.dataframe(filtered_inventory, use_container_width=True)
        
        if not filtered_shipments.empty:
            st.subheader("Filtered Shipments Data")
            st.dataframe(filtered_shipments, use_container_width=True)

if __name__ == "__main__":
    main() 