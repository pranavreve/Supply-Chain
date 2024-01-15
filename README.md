# Supply Chain Analytics Solution

## Overview
This project implements a comprehensive end-to-end supply chain analytics solution that leverages advanced statistical modeling and machine learning techniques to identify critical bottlenecks and optimize order fulfillment.

## Key Features
- Data pipeline for multi-dimensional supply chain data processing
- Time-series decomposition and anomaly detection for lead time analysis
- Multi-variate regression modeling for order fulfillment impact analysis
- Network optimization using graph theory for supplier-distributor-retailer relationships
- Predictive inventory management system using ensemble ML methods
- Interactive dashboard for KPI visualization and scenario analysis

## Technologies
- **Programming**: Python (Pandas, NumPy, SciPy, Statsmodels)
- **Machine Learning**: Scikit-learn, XGBoost, Prophet
- **Network Analysis**: NetworkX
- **Statistical Modeling**: Bayesian inference
- **Optimization**: Linear programming with PuLP
- **Visualization**: Plotly, Dash
- **Infrastructure**: AWS

## Project Structure
- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for analysis and model development
- `src/`: Source code for the project
  - `data/`: Data processing scripts
  - `features/`: Feature engineering code
  - `models/`: ML model training and prediction
  - `visualization/`: Visualization utilities
  - `utils/`: Utility functions
- `dashboard/`: Interactive Dash dashboard
- `tests/`: Unit tests

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/supply-chain-analytics.git
cd supply-chain-analytics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the dashboard
cd dashboard
python app.py

# Run notebooks
jupyter notebook notebooks/
``` 