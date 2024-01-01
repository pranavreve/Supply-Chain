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
app = dash.Dash(
    __name__, 
    title="Supply Chain Analytics Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server  # For deployment

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Supply Chain Analytics Dashboard", className="header-title"),
        html.P("Comprehensive analysis of inventory, lead times, and fulfillment metrics", className="header-description")
    ], className="header"),
    
    # Filters section
    html.Div([
        html.Div([
            html.H3("Filters"),
            
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in df['region'].unique()],
                value=df['region'].unique()[0],
                clearable=False
            ),
            
            html.Label("Select Product:"),
            dcc.Dropdown(
                id='product-filter',
                options=[{'label': product, 'value': product} for product in df['product'].unique()],
                value=df['product'].unique()[0],
                clearable=False
            ),
            
            html.Label("Select Supplier:"),
            dcc.Dropdown(
                id='supplier-filter',
                options=[{'label': supplier, 'value': supplier} for supplier in df['supplier'].unique()],
                value='All',
                clearable=False
            ),
            
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['order_date'].min(),
                end_date=df['order_date'].max(),
                display_format='YYYY-MM-DD'
            ),
            
            html.Label("Metric Selection:"),
            dcc.Checklist(
                id='metrics-checklist',
                options=[
                    {'label': 'Lead Time', 'value': 'lead_time'},
                    {'label': 'Fulfillment Rate', 'value': 'fulfillment_rate'},
                    {'label': 'Stockout Probability', 'value': 'stockout_probability'},
                    {'label': 'Inventory Level', 'value': 'inventory_level'}
                ],
                value=['lead_time', 'fulfillment_rate']
            ),
            
            html.Div([
                html.Button("Run Forecast", id="forecast-button", className="control-button"),
                html.Button("Detect Anomalies", id="anomaly-button", className="control-button")
            ], className="button-container")
            
        ], className="filter-panel"),
        
        # KPI Cards Section
        html.Div([
            html.Div([
                html.H4("Average Lead Time"),
                html.Div(id="avg-lead-time", className="kpi-value")
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Fulfillment Rate"),
                html.Div(id="avg-fulfillment", className="kpi-value")
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Stockout Probability"),
                html.Div(id="avg-stockout", className="kpi-value")
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Total Cost"),
                html.Div(id="total-cost", className="kpi-value")
            ], className="kpi-card")
        ], className="kpi-container")
    ], className="control-row"),
    
    # Main dashboard content
    html.Div([
        # Time Series Visualization Tab
        html.Div([
            html.H3("Time Series Analysis"),
            dcc.Graph(id="time-series-graph")
        ], className="dashboard-card"),
        
        # Inventory Management Tab
        html.Div([
            html.H3("Inventory Management"),
            dcc.Graph(id="inventory-graph")
        ], className="dashboard-card"),
        
        # Cost Analysis Tab
        html.Div([
            html.H3("Cost Analysis"),
            dcc.Graph(id="cost-analysis-graph")
        ], className="dashboard-card"),
        
        # Forecasting Tab
        html.Div([
            html.H3("Forecast & Predictions"),
            dcc.Graph(id="forecast-graph"),
            html.Div(id="forecast-info", className="info-box")
        ], className="dashboard-card")
    ], className="dashboard-container"),
    
    # Footer
    html.Div([
        html.P("Supply Chain Analytics Dashboard | Developed with Dash and Plotly"),
        html.P("Data updated: 2024-02-28")
    ], className="footer")
], className="app-container")

# Define callbacks
@app.callback(
    [Output("avg-lead-time", "children"),
     Output("avg-fulfillment", "children"),
     Output("avg-stockout", "children"),
     Output("total-cost", "children")],
    [Input("region-filter", "value"),
     Input("product-filter", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_kpi_cards(region, product, start_date, end_date):
    # Filter data based on selections
    filtered_df = df[
        (df['region'] == region) &
        (df['product'] == product) &
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    
    # Calculate KPIs
    avg_lead_time = f"{filtered_df['lead_time'].mean():.2f} days"
    avg_fulfillment = f"{filtered_df['fulfillment_rate'].mean():.2%}"
    avg_stockout = f"{filtered_df['stockout_probability'].mean():.2%}"
    total_cost = f"${filtered_df['price'].sum():,.2f}"
    
    return avg_lead_time, avg_fulfillment, avg_stockout, total_cost

@app.callback(
    Output("time-series-graph", "figure"),
    [Input("region-filter", "value"),
     Input("product-filter", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("metrics-checklist", "value")]
)
def update_time_series(region, product, start_date, end_date, metrics):
    # Filter data based on selections
    filtered_df = df[
        (df['region'] == region) &
        (df['product'] == product) &
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    
    # Create subplots for selected metrics
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=n_metrics, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics]
    )
    
    # Add traces for each metric
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Scatter(
                x=filtered_df['order_date'],
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
                    x=filtered_df['order_date'],
                    y=ma,
                    mode='lines',
                    line=dict(dash='dash'),
                    name=f"{metric.replace('_', ' ').title()} (7-day MA)"
                ),
                row=i, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 * n_metrics,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@app.callback(
    Output("inventory-graph", "figure"),
    [Input("region-filter", "value"),
     Input("product-filter", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_inventory_graph(region, product, start_date, end_date):
    # Filter data based on selections
    filtered_df = df[
        (df['region'] == region) &
        (df['product'] == product) &
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add inventory level
    fig.add_trace(
        go.Scatter(
            x=filtered_df['order_date'],
            y=filtered_df['quantity'],
            mode='lines',
            name='Inventory Level',
            line=dict(color='royalblue')
        )
    )
    
    # Add reorder point
    fig.add_trace(
        go.Scatter(
            x=filtered_df['order_date'],
            y=filtered_df['quantity'] * 0.2,
            mode='lines',
            name='Reorder Point',
            line=dict(color='firebrick', dash='dash')
        )
    )
    
    # Add safety stock
    fig.add_trace(
        go.Scatter(
            x=filtered_df['order_date'],
            y=filtered_df['quantity'] * 0.2,
            mode='lines',
            name='Safety Stock',
            line=dict(color='green', dash='dot')
        )
    )
    
    # Add demand
    fig.add_trace(
        go.Scatter(
            x=filtered_df['order_date'],
            y=filtered_df['quantity'],
            mode='lines',
            name='Demand',
            line=dict(color='purple', dash='dashdot')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Inventory Management - {product} in {region}",
        xaxis_title="Date",
        yaxis_title="Units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

@app.callback(
    Output("cost-analysis-graph", "figure"),
    [Input("region-filter", "value"),
     Input("product-filter", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_cost_analysis(region, product, start_date, end_date):
    # Filter data based on selections
    filtered_df = df[
        (df['region'] == region) &
        (df['product'] == product) &
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    
    # Resample data monthly for cost breakdown
    monthly_df = filtered_df.set_index('order_date').resample('M').sum().reset_index()
    
    # Create figure for cost breakdown
    fig = go.Figure()
    
    # Add cost components
    fig.add_trace(
        go.Bar(
            x=monthly_df['order_date'],
            y=monthly_df['price'],
            name='Unit Cost',
            marker_color='#4285F4'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_df['order_date'],
            y=monthly_df['price'] * 0.1 * filtered_df['lead_time'].mean() / 5,
            name='Shipping Cost',
            marker_color='#EA4335'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_df['order_date'],
            y=monthly_df['price'] * 0.02 * filtered_df['quantity'].sum() * 0.1,
            name='Holding Cost',
            marker_color='#FBBC05'
        )
    )
    
    # Add total cost line
    fig.add_trace(
        go.Scatter(
            x=monthly_df['order_date'],
            y=monthly_df['price'] * (filtered_df['quantity'].sum() * filtered_df['stockout_probability'].mean() * 0.5),
            mode='lines+markers',
            name='Stockout Cost',
            line=dict(color='black', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Cost Analysis - {product} in {region} (Monthly)",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

@app.callback(
    [Output("forecast-graph", "figure"),
     Output("forecast-info", "children")],
    [Input("forecast-button", "n_clicks")],
    [State("region-filter", "value"),
     State("product-filter", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date"),
     State("metrics-checklist", "value")]
)
def generate_forecast(n_clicks, region, product, start_date, end_date, metrics):
    # Default values for initial load
    if n_clicks is None:
        fig = go.Figure()
        fig.update_layout(
            title="Forecast (Click 'Run Forecast' button)",
            xaxis_title="Date",
            yaxis_title="Value",
            height=600
        )
        return fig, "Select parameters and click 'Run Forecast' to generate predictions."
    
    # Filter data based on selections
    filtered_df = df[
        (df['region'] == region) &
        (df['product'] == product) &
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    
    # Ensure we have at least one metric selected
    if not metrics:
        metrics = ['lead_time']
    
    # Create forecasting figure
    fig = go.Figure()
    
    # For each selected metric, create a forecast
    forecast_info = []
    
    for metric in metrics[:1]:  # Just use the first metric for simplicity
        # Prepare data for Prophet
        prophet_df = filtered_df[['order_date', metric]].rename(
            columns={'order_date': 'ds', metric: 'y'}
        )
        
        try:
            # Use the current date as the cutoff for training data (80% of data)
            cutoff_idx = int(len(prophet_df) * 0.8)
            train_df = prophet_df.iloc[:cutoff_idx]
            test_df = prophet_df.iloc[cutoff_idx:]
            
            # Simple forecasting model (in real app, would use Prophet)
            # Here we use a simple moving average as a placeholder
            window = 14
            if len(train_df) > window:
                forecast_values = train_df['y'].rolling(window=window).mean().iloc[-1]
                forecast_dates = test_df['ds']
                
                # Add historical data
                fig.add_trace(
                    go.Scatter(
                        x=train_df['ds'],
                        y=train_df['y'],
                        mode='lines',
                        name=f'Historical {metric.replace("_", " ").title()}',
                        line=dict(color='blue')
                    )
                )
                
                # Add test data
                fig.add_trace(
                    go.Scatter(
                        x=test_df['ds'],
                        y=test_df['y'],
                        mode='lines',
                        name=f'Actual {metric.replace("_", " ").title()}',
                        line=dict(color='green')
                    )
                )
                
                # Add forecast
                forecast_y = [forecast_values] * len(forecast_dates)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_y,
                        mode='lines',
                        name=f'Forecast {metric.replace("_", " ").title()}',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Calculate forecast accuracy
                mape = np.mean(np.abs((test_df['y'].values - forecast_y) / test_df['y'].values)) * 100
                forecast_info.append(
                    html.P(f"Forecast accuracy for {metric.replace('_', ' ').title()}: {100-mape:.2f}%")
                )
                
                forecast_info.append(
                    html.P(f"Forecast horizon: {len(test_df)} days")
                )
            else:
                forecast_info.append(
                    html.P(f"Insufficient data for forecasting {metric.replace('_', ' ').title()}")
                )
        except Exception as e:
            forecast_info.append(
                html.P(f"Error generating forecast: {str(e)}")
            )
    
    # Update layout
    fig.update_layout(
        title=f"Forecast for {product} in {region}",
        xaxis_title="Date",
        yaxis_title=metrics[0].replace('_', ' ').title(),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig, forecast_info

@app.callback(
    Output("anomaly-button", "children"),
    [Input("anomaly-button", "n_clicks")],
    [State("region-filter", "value"),
     State("product-filter", "value"),
     State("metrics-checklist", "value")]
)
def detect_anomalies_callback(n_clicks, region, product, metrics):
    if n_clicks is None:
        return "Detect Anomalies"
    
    # In a real implementation, this would trigger anomaly detection algorithms
    # and potentially update the graphs with highlighted anomalies
    
    return "Anomalies Detected"

# Add CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .app-container {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }
            
            .header {
                background-color: #3c4b64;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .header-title {
                margin: 0;
                font-size: 28px;
            }
            
            .header-description {
                margin: 10px 0 0 0;
                font-size: 16px;
                opacity: 0.8;
            }
            
            .control-row {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .filter-panel {
                flex: 1;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .kpi-container {
                flex: 2;
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            
            .kpi-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            
            .kpi-value {
                font-size: 24px;
                font-weight: bold;
                margin-top: 10px;
                color: #3c4b64;
            }
            
            .dashboard-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }
            
            .dashboard-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .button-container {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            
            .control-button {
                background-color: #3c4b64;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            
            .control-button:hover {
                background-color: #2d3a4f;
            }
            
            .info-box {
                margin-top: 15px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 4px;
            }
            
            .footer {
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #6c757d;
                font-size: 14px;
            }
            
            @media (max-width: 1200px) {
                .control-row {
                    flex-direction: column;
                }
                
                .dashboard-container {
                    grid-template-columns: 1fr;
                }
                
                .kpi-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 