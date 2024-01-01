#!/usr/bin/env python
"""
Data validation framework for Supply Chain Analytics using Great Expectations and Pydantic.
This module provides utilities for validating data quality and enforcing schemas.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError, validator, root_validator
import great_expectations as ge
from great_expectations.dataset import PandasDataset
from great_expectations.profile import BasicDatasetProfiler
from great_expectations.render import DefaultJinjaPageView
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.exceptions import DataContextError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("supply_chain_data_validation")


# Define Pydantic models for data schemas
class Product(BaseModel):
    """Product data schema model."""
    product_id: str = Field(..., description="Unique identifier for the product")
    product_name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Product category")
    price: float = Field(..., description="Product price", gt=0)
    weight: float = Field(..., description="Product weight in kg", gt=0)

    @validator('product_id')
    def validate_product_id(cls, v):
        """Ensure product_id follows the expected format."""
        if not v.startswith('P') or not v[1:].isdigit():
            raise ValueError('product_id must start with P followed by digits')
        return v


class Supplier(BaseModel):
    """Supplier data schema model."""
    supplier_id: str = Field(..., description="Unique identifier for the supplier")
    supplier_name: str = Field(..., description="Name of the supplier")
    country: str = Field(..., description="Country of the supplier")
    reliability_score: float = Field(..., description="Supplier reliability score", ge=0, le=100)
    lead_time_days: int = Field(..., description="Average lead time in days", ge=0)

    @validator('supplier_id')
    def validate_supplier_id(cls, v):
        """Ensure supplier_id follows the expected format."""
        if not v.startswith('S') or not v[1:].isdigit():
            raise ValueError('supplier_id must start with S followed by digits')
        return v


class Order(BaseModel):
    """Order data schema model."""
    order_id: str = Field(..., description="Unique identifier for the order")
    product_id: str = Field(..., description="Product identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    quantity: int = Field(..., description="Order quantity", gt=0)
    order_date: datetime = Field(..., description="Date of order")
    expected_delivery: datetime = Field(..., description="Expected delivery date")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery date")
    status: str = Field(..., description="Order status")

    @validator('order_id')
    def validate_order_id(cls, v):
        """Ensure order_id follows the expected format."""
        if not v.startswith('O') or not v[1:].isdigit():
            raise ValueError('order_id must start with O followed by digits')
        return v

    @validator('status')
    def validate_status(cls, v):
        """Ensure status is one of the allowed values."""
        allowed_statuses = ['pending', 'shipped', 'delivered', 'canceled']
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v.lower()

    @root_validator
    def validate_delivery_dates(cls, values):
        """Validate that delivery dates make logical sense."""
        order_date = values.get('order_date')
        expected_delivery = values.get('expected_delivery')
        actual_delivery = values.get('actual_delivery')

        if order_date and expected_delivery and expected_delivery < order_date:
            raise ValueError('Expected delivery date cannot be before order date')
        
        if order_date and actual_delivery and actual_delivery < order_date:
            raise ValueError('Actual delivery date cannot be before order date')
        
        return values


class Inventory(BaseModel):
    """Inventory data schema model."""
    inventory_id: str = Field(..., description="Unique identifier for inventory record")
    product_id: str = Field(..., description="Product identifier")
    warehouse_id: str = Field(..., description="Warehouse identifier")
    quantity: int = Field(..., description="Quantity in stock", ge=0)
    reorder_level: int = Field(..., description="Reorder level quantity", ge=0)
    last_restock_date: datetime = Field(..., description="Date of last restock")

    @validator('inventory_id')
    def validate_inventory_id(cls, v):
        """Ensure inventory_id follows the expected format."""
        if not v.startswith('I') or not v[1:].isdigit():
            raise ValueError('inventory_id must start with I followed by digits')
        return v

    @validator('warehouse_id')
    def validate_warehouse_id(cls, v):
        """Ensure warehouse_id follows the expected format."""
        if not v.startswith('W') or not v[1:].isdigit():
            raise ValueError('warehouse_id must start with W followed by digits')
        return v

    @root_validator
    def validate_inventory_levels(cls, values):
        """Validate inventory levels and reorder points."""
        quantity = values.get('quantity')
        reorder_level = values.get('reorder_level')
        
        if quantity is not None and reorder_level is not None:
            if reorder_level > quantity * 2:
                raise ValueError('Reorder level should not be more than twice the current quantity')
        
        return values


# Great Expectations validators
class DataValidator:
    """
    Data validation using Great Expectations.
    Provides methods to validate datasets against expectations and generate reports.
    """
    
    def __init__(self, data_path: str, report_path: str = None):
        """
        Initialize the data validator.
        
        Args:
            data_path: Path to the data directory
            report_path: Path to save validation reports
        """
        self.data_path = Path(data_path)
        self.report_path = Path(report_path) if report_path else Path("reports/validation")
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.validation_results = {}
        self.datasets = {}
        
    def load_dataset(self, file_name: str) -> PandasDataset:
        """
        Load a dataset from file and convert to a Great Expectations dataset.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            A Great Expectations PandasDataset
        """
        file_path = self.data_path / file_name
        logger.info(f"Loading dataset from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Convert to Great Expectations dataset
        ge_dataset = ge.from_pandas(df)
        dataset_name = file_path.stem
        self.datasets[dataset_name] = ge_dataset
        
        return ge_dataset
    
    def profile_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Generate a profile of a dataset.
        
        Args:
            dataset_name: Name of the dataset to profile
            
        Returns:
            Profile results dictionary
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Profiling dataset: {dataset_name}")
        profile = BasicDatasetProfiler.profile(self.datasets[dataset_name])
        
        # Save profile to file
        profile_path = self.report_path / f"{dataset_name}_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.to_json_dict(), f, indent=2)
        
        return profile.to_json_dict()
    
    def create_expectations(self, dataset_name: str, expectations: List[Dict[str, Any]]) -> None:
        """
        Create expectations for a dataset.
        
        Args:
            dataset_name: Name of the dataset to create expectations for
            expectations: List of expectation configurations
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Creating {len(expectations)} expectations for dataset: {dataset_name}")
        for expectation_config in expectations:
            self.datasets[dataset_name].add_expectation(
                ExpectationConfiguration(**expectation_config)
            )
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate a dataset against its expectations.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Validation results dictionary
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Validating dataset: {dataset_name}")
        validation_result = self.datasets[dataset_name].validate()
        self.validation_results[dataset_name] = validation_result
        
        # Save validation results to file
        validation_path = self.report_path / f"{dataset_name}_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_result.to_json_dict(), f, indent=2)
        
        return validation_result.to_json_dict()
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report for all validated datasets.
        
        Returns:
            Report data as a dictionary
        """
        logger.info("Generating comprehensive data quality report")
        report = {
            "report_time": datetime.now().isoformat(),
            "datasets": {},
            "summary": {
                "total_datasets": len(self.validation_results),
                "passed_datasets": 0,
                "total_expectations": 0,
                "passed_expectations": 0
            }
        }
        
        for dataset_name, validation_result in self.validation_results.items():
            result_dict = validation_result.to_json_dict()
            
            # Extract statistics from validation result
            total_expectations = len(result_dict["results"])
            passed_expectations = sum(1 for r in result_dict["results"] if r["success"])
            
            dataset_report = {
                "success": result_dict["success"],
                "total_expectations": total_expectations,
                "passed_expectations": passed_expectations,
                "failure_percent": (total_expectations - passed_expectations) / total_expectations * 100 if total_expectations > 0 else 0,
                "failed_expectations": [
                    {
                        "expectation_type": r["expectation_config"]["expectation_type"],
                        "kwargs": r["expectation_config"]["kwargs"],
                        "exception_info": r.get("exception_info", {})
                    }
                    for r in result_dict["results"] if not r["success"]
                ]
            }
            
            report["datasets"][dataset_name] = dataset_report
            
            # Update summary
            report["summary"]["total_expectations"] += total_expectations
            report["summary"]["passed_expectations"] += passed_expectations
            if dataset_report["success"]:
                report["summary"]["passed_datasets"] += 1
        
        # Calculate overall success percentage
        total_exp = report["summary"]["total_expectations"]
        if total_exp > 0:
            report["summary"]["success_percentage"] = (report["summary"]["passed_expectations"] / total_exp) * 100
        else:
            report["summary"]["success_percentage"] = 0
        
        # Save the report to file
        report_path = self.report_path / "data_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

    def validate_with_pydantic(self, dataset_name: str, model_class: type) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a dataset using Pydantic models.
        
        Args:
            dataset_name: Name of the dataset to validate
            model_class: Pydantic model class to use for validation
            
        Returns:
            Tuple of (valid_records, invalid_records)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset = self.datasets[dataset_name]
        df = dataset.pandas_df
        
        valid_records = []
        invalid_records = []
        
        for _, row in df.iterrows():
            record = row.to_dict()
            try:
                # Convert datetime strings to datetime objects
                for field_name, field in model_class.__fields__.items():
                    if field.type_ == datetime and field_name in record and isinstance(record[field_name], str):
                        try:
                            record[field_name] = pd.to_datetime(record[field_name])
                        except Exception:
                            pass  # Let Pydantic handle the validation error
                
                validated = model_class(**record)
                valid_records.append(validated.dict())
            except ValidationError as e:
                invalid_records.append({
                    "record": record,
                    "errors": e.errors()
                })
        
        # Save validation results
        pydantic_results = {
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "invalid_records": invalid_records
        }
        
        results_path = self.report_path / f"{dataset_name}_pydantic_validation.json"
        with open(results_path, 'w') as f:
            # Convert any non-serializable objects to strings
            json.dump(pydantic_results, f, indent=2, default=str)
        
        return valid_records, invalid_records


# Common data expectations for supply chain datasets
def get_product_expectations() -> List[Dict[str, Any]]:
    """Get common expectations for product data."""
    return [
        {
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": ["product_id", "product_name", "category", "price", "weight"]
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "product_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "product_id",
                "regex": "^P\\d+$"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {
                "column": "product_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "product_name"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {
                "column": "price",
                "type_": "float"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "price",
                "min_value": 0,
                "strict_min": True
            }
        }
    ]


def get_supplier_expectations() -> List[Dict[str, Any]]:
    """Get common expectations for supplier data."""
    return [
        {
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": ["supplier_id", "supplier_name", "country", "reliability_score", "lead_time_days"]
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "supplier_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "supplier_id",
                "regex": "^S\\d+$"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {
                "column": "supplier_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "reliability_score",
                "min_value": 0,
                "max_value": 100
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "lead_time_days",
                "min_value": 0
            }
        }
    ]


def get_order_expectations() -> List[Dict[str, Any]]:
    """Get common expectations for order data."""
    return [
        {
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": ["order_id", "product_id", "supplier_id", "quantity", 
                               "order_date", "expected_delivery", "actual_delivery", "status"]
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "order_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "order_id",
                "regex": "^O\\d+$"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {
                "column": "order_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "status",
                "value_set": ["pending", "shipped", "delivered", "canceled"]
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "quantity",
                "min_value": 0,
                "strict_min": True
            }
        }
    ]


def get_inventory_expectations() -> List[Dict[str, Any]]:
    """Get common expectations for inventory data."""
    return [
        {
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": ["inventory_id", "product_id", "warehouse_id", 
                               "quantity", "reorder_level", "last_restock_date"]
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "inventory_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "inventory_id",
                "regex": "^I\\d+$"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {
                "column": "inventory_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "warehouse_id",
                "regex": "^W\\d+$"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "quantity",
                "min_value": 0
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "reorder_level",
                "min_value": 0
            }
        }
    ]


def main():
    """Main entry point for the data validation script."""
    parser = argparse.ArgumentParser(description="Validate supply chain data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--report-path", type=str, default="reports/validation", help="Path to save validation reports")
    args = parser.parse_args()
    
    # Model class mapping for Pydantic validation
    model_mapping = {
        "products": Product,
        "suppliers": Supplier,
        "orders": Order,
        "inventory": Inventory
    }
    
    # Expectations mapping
    expectation_mapping = {
        "products": get_product_expectations(),
        "suppliers": get_supplier_expectations(),
        "orders": get_order_expectations(),
        "inventory": get_inventory_expectations()
    }
    
    # Create validator
    validator = DataValidator(args.data_path, args.report_path)
    
    # Get all CSV files in the data path
    data_dir = Path(args.data_path)
    data_files = list(data_dir.glob("*.csv"))
    
    if not data_files:
        logger.warning(f"No CSV files found in {args.data_path}")
        return
    
    # Process each data file
    for file_path in data_files:
        dataset_name = file_path.stem
        logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Load the dataset
            validator.load_dataset(file_path.name)
            
            # If we have expectations for this dataset type, apply them
            if dataset_name in expectation_mapping:
                validator.create_expectations(dataset_name, expectation_mapping[dataset_name])
                
                # Validate using Great Expectations
                validation_result = validator.validate_dataset(dataset_name)
                success = validation_result.get("success", False)
                logger.info(f"Validation {'successful' if success else 'failed'} for {dataset_name}")
                
                # Validate using Pydantic if we have a model for this dataset
                if dataset_name in model_mapping:
                    valid_records, invalid_records = validator.validate_with_pydantic(
                        dataset_name, model_mapping[dataset_name]
                    )
                    logger.info(f"Pydantic validation: {len(valid_records)} valid records, "
                                f"{len(invalid_records)} invalid records")
            else:
                # For unknown dataset types, just profile the data
                logger.info(f"No expectations defined for {dataset_name}, generating profile only")
                validator.profile_dataset(dataset_name)
        
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
    
    # Generate comprehensive report
    report = validator.generate_data_quality_report()
    logger.info(f"Data quality report generated: {report['summary']['passed_expectations']} of "
                f"{report['summary']['total_expectations']} expectations passed "
                f"({report['summary']['success_percentage']:.2f}%)")


if __name__ == "__main__":
    main() 