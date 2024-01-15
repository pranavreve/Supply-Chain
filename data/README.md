# Data Directory

This directory contains data files for the supply chain analytics project.

## Directory Structure

- `raw/`: Contains raw data files before preprocessing
- `processed/`: Contains cleaned and processed data files ready for analysis and modeling
- `external/`: Contains external data that may be used for enrichment or reference

## Data Requirements

### Expected Input Format

The analysis code expects supply chain data with the following minimum columns:

- `order_id`: Unique identifier for each order
- `order_date`: Date when the order was placed
- `delivery_date`: Date when the order was delivered
- `product_id`: Identifier for the product
- `quantity`: Number of units ordered
- `price`: Price per unit
- `region`: Geographical region

Additional columns that enhance the analysis:

- `supplier_id`: Identifier for the supplier
- `distributor_id`: Identifier for the distributor
- `retailer_id`: Identifier for the retailer
- `priority`: Order priority level (e.g., "Low", "Medium", "High")
- `shipping_cost`: Cost of shipping

### Data Sources

The data can come from various sources:

1. ERP systems (SAP, Oracle, etc.)
2. Order management systems
3. Warehouse management systems
4. Transportation management systems
5. Customer relationship management systems

## Sample Data

If you don't have real data available, the notebook `notebooks/01_data_preprocessing.ipynb` 
includes a function to generate synthetic supply chain data for demonstration purposes. 