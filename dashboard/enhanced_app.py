import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath('../src'))

from models.time_series import decompose_time_series, detect_anomalies, forecast_with_prophet
from models.inventory_optimization import calculate_safety_stock, calculate_reorder_point
from visualization.enhanced_visualize import plot_inventory_levels, plot_supply_chain_metrics_dashboard

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    title="Supply Chain Analytics Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Sample data generation (in real implementation, you would load processed data)
def generate_sample_data():
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=365)
    
    # Define parameters
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C']
    suppliers = ['Supplier 1', 'Supplier 2', 'Supplier 3', 'Supplier 4']
    
    # Generate data
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

# Generate sample data
df = generate_sample_data()

# Define app layout
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
                start_date=df['date'].min(),
                end_date=df['date'].max(),
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
            
            html.Button('Apply Filters', id='apply-filters-button', className='button')
        ], className="filters-panel")
    ], className="filters-container"),
    
    # Main dashboard content
    html.Div([
        # KPI cards
        html.Div([
            html.Div([
                html.H4("Average Lead Time"),
                html.Div(id='avg-lead-time-kpi', className='kpi-value')
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Fulfillment Rate"),
                html.Div(id='fulfillment-rate-kpi', className='kpi-value')
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Stockout Probability"),
                html.Div(id='stockout-probability-kpi', className='kpi-value')
            ], className="kpi-card"),
            
            html.Div([
                html.H4("Total Cost"),
                html.Div(id='total-cost-kpi', className='kpi-value')
            ], className="kpi-card")
        ], className="kpi-container"),
        
        # Tabs for different analyses
        dcc.Tabs([
            # Time Series Tab
            dcc.Tab(label="Time Series Analysis", children=[
                html.Div([
                    html.H3("Time Series Metrics"),
                    html.Div([
                        html.Label("Select Metric:"),
                        dcc.Dropdown(
                            id='ts-metric-dropdown',
                            options=[
                                {'label': 'Lead Time', 'value': 'lead_time'},
                                {'label': 'Demand', 'value': 'demand'},
                                {'label': 'Inventory Level', 'value': 'inventory_level'},
                                {'label': 'Fulfillment Rate', 'value': 'fulfillment_rate'}
                            ],
                            value='lead_time'
                        ),
                        html.Button('Generate Forecast', id='forecast-button', className='button')
                    ], className="control-row"),
                    
                    # Time series plot
                    dcc.Graph(id='time-series-plot'),
                    
                    # Forecast plot
                    html.Div(id='forecast-container', children=[
                        html.H3("Time Series Forecast"),
                        dcc.Graph(id='forecast-plot')
                    ], style={'display': 'none'})
                ], className="tab-content")
            ]),
            
            # Inventory Management Tab
            dcc.Tab(label="Inventory Management", children=[
                html.Div([
                    html.H3("Inventory Analysis"),
                    
                    # Inventory levels plot
                    dcc.Graph(id='inventory-levels-plot'),
                    
                    # Inventory optimization section
                    html.Div([
                        html.H3("Inventory Optimization"),
                        html.Div([
                            html.Div([
                                html.Label("Service Level:"),
                                dcc.Slider(
                                    id='service-level-slider',
                                    min=0.8,
                                    max=0.99,
                                    step=0.01,
                                    value=0.95,
                                    marks={i/100: f'{i}%' for i in range(80, 100, 5)}
                                )
                            ], className="slider-container"),
                            
                            html.Button('Calculate Optimal Levels', id='optimize-button', className='button')
                        ], className="control-row"),
                        
                        # Optimization results
                        html.Div(id='optimization-results')
                    ], className="optimization-section")
                ], className="tab-content")
            ]),
            
            # Cost Analysis Tab
            dcc.Tab(label="Cost Analysis", children=[
                html.Div([
                    html.H3("Cost Breakdown"),
                    dcc.Graph(id='cost-breakdown-plot'),
                    
                    html.H3("Cost Trends"),
                    dcc.Graph(id='cost-trends-plot')
                ], className="tab-content")
            ]),
            
            # Supplier Analysis Tab
            dcc.Tab(label="Supplier Analysis", children=[
                html.Div([
                    html.H3("Supplier Performance"),
                    dcc.Graph(id='supplier-performance-plot'),
                    
                    html.H3("Supplier Lead Time Comparison"),
                    dcc.Graph(id='supplier-leadtime-plot')
                ], className="tab-content")
            ])
        ], id="analysis-tabs", className="tabs-container")
    ], className="dashboard-content"),
    
    # Footer
    html.Div([
        html.P("Supply Chain Analytics Dashboard © 2024")
    ], className="footer")
], className="dashboard-container")

# Define callbacks
@app.callback(
    [Output('avg-lead-time-kpi', 'children'),
     Output('fulfillment-rate-kpi', 'children'),
     Output('stockout-probability-kpi', 'children'),
     Output('total-cost-kpi', 'children')],
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('supplier-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_kpis(n_clicks, region, product, supplier, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if supplier and supplier != 'All':
        filtered_df = filtered_df[filtered_df['supplier'] == supplier]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Calculate KPIs
    avg_lead_time = f"{filtered_df['lead_time'].mean():.2f} days"
    fulfillment_rate = f"{filtered_df['fulfillment_rate'].mean() * 100:.1f}%"
    stockout_probability = f"{filtered_df['stockout_probability'].mean() * 100:.1f}%"
    total_cost = f"${filtered_df['total_cost'].mean():.2f}"
    
    return avg_lead_time, fulfillment_rate, stockout_probability, total_cost

@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks'),
     Input('ts-metric-dropdown', 'value')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_time_series_plot(n_clicks, metric, region, product, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Aggregate data by date
    ts_data = filtered_df.groupby('date')[metric].mean().reset_index()
    
    # Create time series plot
    fig = px.line(
        ts_data, 
        x='date', 
        y=metric,
        title=f'{metric.replace("_", " ").title()} Over Time'
    )
    
    # Add 7-day moving average
    ts_data[f'{metric}_7d_ma'] = ts_data[metric].rolling(window=7).mean()
    fig.add_scatter(
        x=ts_data['date'],
        y=ts_data[f'{metric}_7d_ma'],
        mode='lines',
        name='7-day Moving Average',
        line=dict(width=2, dash='dash')
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric.replace('_', ' ').title(),
        legend_title='Legend',
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    [Output('forecast-container', 'style'),
     Output('forecast-plot', 'figure')],
    [Input('forecast-button', 'n_clicks')],
    [State('ts-metric-dropdown', 'value'),
     State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def generate_forecast(n_clicks, metric, region, product, start_date, end_date):
    if not n_clicks:
        return {'display': 'none'}, go.Figure()
    
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Aggregate data by date
    ts_data = filtered_df.groupby('date')[metric].mean().reset_index()
    
    try:
        # Generate forecast using Prophet
        model, forecast = forecast_with_prophet(ts_data, metric, 'date', forecast_periods=30)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=ts_data[metric],
                mode='markers',
                name='Actual',
                marker=dict(color='black', size=6)
            )
        )
        
        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue')
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'].iloc[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='95% Confidence Interval'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{metric.replace("_", " ").title()} Forecast',
            xaxis_title='Date',
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        return {'display': 'block'}, fig
    except Exception as e:
        # If forecast fails, return empty figure
        return {'display': 'none'}, go.Figure()

@app.callback(
    Output('inventory-levels-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_inventory_levels_plot(n_clicks, region, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Create inventory levels plot
    return plot_inventory_levels(
        filtered_df, 
        product_col='product', 
        inventory_col='inventory_level', 
        reorder_point_col='reorder_point', 
        date_col='date'
    )

@app.callback(
    Output('optimization-results', 'children'),
    [Input('optimize-button', 'n_clicks')],
    [State('service-level-slider', 'value'),
     State('product-filter', 'value'),
     State('region-filter', 'value')]
)
def update_optimization_results(n_clicks, service_level, product, region):
    if not n_clicks:
        return html.Div()
    
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    # Calculate safety stock
    safety_stock = calculate_safety_stock(
        filtered_df, 
        demand_col='demand', 
        lead_time_col='lead_time', 
        service_level=service_level
    )
    
    # Calculate reorder point
    avg_demand = filtered_df['demand'].mean()
    avg_lead_time = filtered_df['lead_time'].mean()
    reorder_point = calculate_reorder_point(safety_stock, avg_demand, avg_lead_time)
    
    return html.Div([
        html.H4("Optimization Results"),
        html.Div([
            html.Div([
                html.P("Safety Stock:"),
                html.H3(f"{safety_stock:.2f} units")
            ], className="result-card"),
            
            html.Div([
                html.P("Reorder Point:"),
                html.H3(f"{reorder_point:.2f} units")
            ], className="result-card"),
            
            html.Div([
                html.P("Service Level:"),
                html.H3(f"{service_level * 100:.1f}%")
            ], className="result-card"),
            
            html.Div([
                html.P("Average Demand:"),
                html.H3(f"{avg_demand:.2f} units")
            ], className="result-card")
        ], className="results-container")
    ])

@app.callback(
    Output('cost-breakdown-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_cost_breakdown_plot(n_clicks, region, product, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Aggregate costs
    cost_data = filtered_df.groupby('product').agg({
        'shipping_cost': 'mean',
        'holding_cost': 'mean',
        'stockout_cost': 'mean'
    }).reset_index()
    
    # Melt for easier plotting
    cost_data_melted = pd.melt(
        cost_data, 
        id_vars=['product'], 
        value_vars=['shipping_cost', 'holding_cost', 'stockout_cost'],
        var_name='cost_type', 
        value_name='cost'
    )
    
    # Create cost breakdown plot
    fig = px.bar(
        cost_data_melted, 
        x='product', 
        y='cost', 
        color='cost_type',
        title='Cost Breakdown by Product',
        labels={'cost': 'Average Cost ($)', 'product': 'Product', 'cost_type': 'Cost Type'},
        color_discrete_map={
            'shipping_cost': '#1f77b4',
            'holding_cost': '#ff7f0e',
            'stockout_cost': '#d62728'
        }
    )
    
    return fig

@app.callback(
    Output('cost-trends-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_cost_trends_plot(n_clicks, region, product, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Aggregate costs by date
    cost_trends = filtered_df.groupby(['date', 'product']).agg({
        'total_cost': 'mean'
    }).reset_index()
    
    # Create cost trends plot
    fig = px.line(
        cost_trends, 
        x='date', 
        y='total_cost', 
        color='product',
        title='Cost Trends Over Time',
        labels={'total_cost': 'Total Cost ($)', 'date': 'Date', 'product': 'Product'}
    )
    
    return fig

@app.callback(
    Output('supplier-performance-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_supplier_performance_plot(n_clicks, region, product, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Aggregate supplier performance
    supplier_performance = filtered_df.groupby('supplier').agg({
        'lead_time': 'mean',
        'fulfillment_rate': 'mean'
    }).reset_index()
    
    # Create supplier performance plot
    fig = px.scatter(
        supplier_performance, 
        x='lead_time', 
        y='fulfillment_rate',
        size=[100] * len(supplier_performance),
        color='supplier',
        title='Supplier Performance Comparison',
        labels={
            'lead_time': 'Average Lead Time (days)',
            'fulfillment_rate': 'Fulfillment Rate',
            'supplier': 'Supplier'
        },
        range_y=[0.7, 1.0]
    )
    
    # Add quadrant lines
    median_lead_time = supplier_performance['lead_time'].median()
    median_fulfillment = supplier_performance['fulfillment_rate'].median()
    
    fig.add_shape(
        type="line", x0=median_lead_time, x1=median_lead_time, 
        y0=0.7, y1=1.0, line=dict(dash="dash", color="gray")
    )
    
    fig.add_shape(
        type="line", x0=0, x1=10, 
        y0=median_fulfillment, y1=median_fulfillment, line=dict(dash="dash", color="gray")
    )
    
    return fig

@app.callback(
    Output('supplier-leadtime-plot', 'figure'),
    [Input('apply-filters-button', 'n_clicks')],
    [State('region-filter', 'value'),
     State('product-filter', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_supplier_leadtime_plot(n_clicks, region, product, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['region'] == region]
    
    if product:
        filtered_df = filtered_df[filtered_df['product'] == product]
    
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Create supplier lead time comparison
    fig = px.box(
        filtered_df, 
        x='supplier', 
        y='lead_time',
        color='supplier',
        title='Supplier Lead Time Distribution',
        labels={'lead_time': 'Lead Time (days)', 'supplier': 'Supplier'}
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 