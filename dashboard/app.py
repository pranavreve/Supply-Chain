import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.data.preprocessing import clean_data, calculate_lead_times
from src.visualization.visualize import create_time_series_plot, create_supply_chain_network
from src.utils.utils import load_data, detect_outliers

# Load or generate sample data
def load_or_generate_data():
    """Load processed data or generate sample data if it doesn't exist"""
    processed_data_path = '../data/processed/processed_order_data.csv'
    daily_metrics_path = '../data/processed/daily_metrics.csv'
    
    try:
        # Try to load processed data
        processed_df = pd.read_csv(processed_data_path, parse_dates=['order_date', 'delivery_date'])
        daily_df = pd.read_csv(daily_metrics_path, parse_dates=['order_date'])
        return processed_df, daily_df
    except:
        # Generate sample data if file doesn't exist
        print("Generating sample data...")
        # Create sample dates
        start_date = pd.to_datetime('2023-01-01')
        dates = pd.date_range(start=start_date, periods=365, freq='D')
        
        # Create regions, products, and other dimensions
        regions = ['North', 'South', 'East', 'West']
        products = [f'Product {i}' for i in range(1, 21)]
        suppliers = [f'Supplier {i}' for i in range(1, 11)]
        
        # Generate random data
        np.random.seed(42)
        n_rows = 1000
        
        df = pd.DataFrame({
            'order_id': range(1000, 1000 + n_rows),
            'order_date': np.random.choice(dates, n_rows),
            'region': np.random.choice(regions, n_rows),
            'product': np.random.choice(products, n_rows),
            'supplier': np.random.choice(suppliers, n_rows),
            'quantity': np.random.randint(1, 100, n_rows),
            'price': np.random.uniform(10, 1000, n_rows).round(2),
        })
        
        # Add delivery dates and lead times
        region_delays = {'North': 3, 'South': 5, 'East': 2, 'West': 4}
        df['base_lead_time'] = df['region'].map(region_delays)
        df['lead_time_variance'] = np.random.normal(0, 2, n_rows).round().astype(int)
        df['lead_time'] = df['base_lead_time'] + df['lead_time_variance'] 
        df['lead_time'] = df['lead_time'].apply(lambda x: max(1, x))  # Ensure minimum 1 day
        df['delivery_date'] = df.apply(lambda row: row['order_date'] + pd.Timedelta(days=row['lead_time']), axis=1)
        
        # Add performance metrics
        df['on_time_delivery'] = np.random.choice([0, 1], n_rows, p=[0.1, 0.9])
        df['fulfillment_rate'] = np.random.uniform(0.8, 1.0, n_rows).round(2)
        df['stockout_probability'] = np.random.uniform(0, 0.2, n_rows).round(2)
        
        # Drop temporary columns
        df = df.drop(columns=['base_lead_time', 'lead_time_variance'])
        
        # Generate daily metrics
        daily_df = df.groupby(df['order_date'].dt.date).agg({
            'lead_time': ['mean', 'median', 'std'],
            'quantity': ['sum', 'mean'],
            'price': ['sum', 'mean'],
            'on_time_delivery': 'mean',
            'fulfillment_rate': 'mean',
            'stockout_probability': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
        daily_df = daily_df.rename(columns={'order_date_': 'order_date'})
        daily_df['order_date'] = pd.to_datetime(daily_df['order_date'])
        
        return df, daily_df

# Load data
df, daily_metrics = load_or_generate_data()

# Initialize the Dash app
app = dash.Dash(__name__, title="Supply Chain Analytics Dashboard")
server = app.server  # For deployment

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Supply Chain Analytics Dashboard", style={'textAlign': 'center'}),
        html.P("Comprehensive analytics for supply chain optimization", style={'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    # Main content
    html.Div([
        # Filters panel
        html.Div([
            html.H3("Filters", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-filter',
                start_date=df['order_date'].min(),
                end_date=df['order_date'].max(),
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Region:"),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in df['region'].unique()],
                multi=True,
                placeholder="Select regions",
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Product:"),
            dcc.Dropdown(
                id='product-filter',
                options=[{'label': product, 'value': product} for product in df['product'].unique()],
                multi=True,
                placeholder="Select products",
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Hr(),
            
            html.Button(
                'Apply Filters',
                id='apply-filters-button',
                className='button-primary',
                style={'width': '100%', 'marginTop': '10px'}
            )
        ], style={'width': '20%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
        
        # Charts panel
        html.Div([
            # KPI cards row
            html.Div([
                # Average Lead Time KPI
                html.Div([
                    html.H4("Avg. Lead Time"),
                    html.Div(id='avg-lead-time-kpi', className='kpi-value')
                ], className='kpi-card'),
                
                # On-Time Delivery KPI
                html.Div([
                    html.H4("On-Time Delivery"),
                    html.Div(id='on-time-delivery-kpi', className='kpi-value')
                ], className='kpi-card'),
                
                # Fulfillment Rate KPI
                html.Div([
                    html.H4("Fulfillment Rate"),
                    html.Div(id='fulfillment-rate-kpi', className='kpi-value')
                ], className='kpi-card'),
                
                # Stockout Probability KPI
                html.Div([
                    html.H4("Stockout Risk"),
                    html.Div(id='stockout-prob-kpi', className='kpi-value')
                ], className='kpi-card')
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            # Charts row 1
            html.Div([
                # Lead Time Trends
                html.Div([
                    html.H3("Lead Time Trends", style={'textAlign': 'center'}),
                    dcc.Graph(id='lead-time-graph')
                ], style={'width': '48%', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
                
                # Order Volume Trends
                html.Div([
                    html.H3("Order Volume Trends", style={'textAlign': 'center'}),
                    dcc.Graph(id='order-volume-graph')
                ], style={'width': '48%', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            # Charts row 2
            html.Div([
                # Regional Performance
                html.Div([
                    html.H3("Regional Performance", style={'textAlign': 'center'}),
                    dcc.Graph(id='regional-performance-graph')
                ], style={'width': '48%', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
                
                # Product Performance
                html.Div([
                    html.H3("Product Performance", style={'textAlign': 'center'}),
                    dcc.Graph(id='product-performance-graph')
                ], style={'width': '48%', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            # Data table
            html.Div([
                html.H3("Order Details", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='order-table',
                    columns=[
                        {'name': 'Order ID', 'id': 'order_id'},
                        {'name': 'Order Date', 'id': 'order_date'},
                        {'name': 'Delivery Date', 'id': 'delivery_date'},
                        {'name': 'Region', 'id': 'region'},
                        {'name': 'Product', 'id': 'product'},
                        {'name': 'Quantity', 'id': 'quantity'},
                        {'name': 'Lead Time', 'id': 'lead_time'}
                    ],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ]
                )
            ], style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
        ], style={'width': '78%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}),
    
    # Footer
    html.Div([
        html.P("Supply Chain Analytics Dashboard © 2024", style={'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginTop': '20px'})
])

# Define callbacks
@app.callback(
    [Output('avg-lead-time-kpi', 'children'),
     Output('on-time-delivery-kpi', 'children'),
     Output('fulfillment-rate-kpi', 'children'),
     Output('stockout-prob-kpi', 'children'),
     Output('lead-time-graph', 'figure'),
     Output('order-volume-graph', 'figure'),
     Output('regional-performance-graph', 'figure'),
     Output('product-performance-graph', 'figure'),
     Output('order-table', 'data')],
    [Input('apply-filters-button', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('region-filter', 'value'),
     State('product-filter', 'value')]
)
def update_dashboard(n_clicks, start_date, end_date, regions, products):
    # Filter data based on user selections
    filtered_df = df.copy()
    
    # Apply date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['order_date'] >= start_date) & 
                                 (filtered_df['order_date'] <= end_date)]
    
    # Apply region filter
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    
    # Apply product filter
    if products:
        filtered_df = filtered_df[filtered_df['product'].isin(products)]
    
    # Calculate KPIs
    avg_lead_time = f"{filtered_df['lead_time'].mean():.1f} days"
    on_time_delivery = f"{filtered_df['on_time_delivery'].mean() * 100:.1f}%"
    fulfillment_rate = f"{filtered_df['fulfillment_rate'].mean() * 100:.1f}%"
    stockout_prob = f"{filtered_df['stockout_probability'].mean() * 100:.1f}%"
    
    # Create lead time trend figure
    lead_time_fig = px.line(
        daily_metrics,
        x='order_date',
        y='lead_time_mean',
        title='Average Lead Time Trend'
    )
    lead_time_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Lead Time (days)',
        hovermode='x unified'
    )
    
    # Create order volume trend figure
    volume_fig = px.bar(
        daily_metrics,
        x='order_date',
        y='quantity_sum',
        title='Daily Order Volume'
    )
    volume_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Order Quantity'
    )
    
    # Create regional performance figure
    regional_perf = filtered_df.groupby('region').agg({
        'lead_time': 'mean',
        'on_time_delivery': 'mean',
        'fulfillment_rate': 'mean'
    }).reset_index()
    
    regional_fig = px.bar(
        regional_perf,
        x='region',
        y=['lead_time', 'on_time_delivery', 'fulfillment_rate'],
        barmode='group',
        title='Performance by Region'
    )
    regional_fig.update_layout(
        xaxis_title='Region',
        yaxis_title='Value',
        legend_title='Metric'
    )
    
    # Create product performance figure
    product_perf = filtered_df.groupby('product').agg({
        'quantity': 'sum',
        'lead_time': 'mean'
    }).reset_index().sort_values('quantity', ascending=False).head(10)
    
    product_fig = px.scatter(
        product_perf,
        x='quantity',
        y='lead_time',
        size='quantity',
        color='product',
        title='Product Performance (Top 10 by Volume)'
    )
    product_fig.update_layout(
        xaxis_title='Total Quantity',
        yaxis_title='Average Lead Time (days)'
    )
    
    # Prepare table data
    table_data = filtered_df.head(100).to_dict('records')
    
    return avg_lead_time, on_time_delivery, fulfillment_rate, stockout_prob, lead_time_fig, volume_fig, regional_fig, product_fig, table_data

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 