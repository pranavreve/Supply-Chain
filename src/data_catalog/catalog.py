import pandas as pd
import json
import os
import boto3
from datetime import datetime
import logging
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCatalog:
    """
    Data catalog to maintain metadata about datasets in the data lake
    """
    
    def __init__(self, catalog_path='catalog.json', s3_bucket=None, s3_prefix='metadata/'):
        """
        Initialize data catalog
        
        Parameters
        ----------
        catalog_path : str
            Local path to catalog JSON file
        s3_bucket : str, optional
            S3 bucket for remote storage
        s3_prefix : str, optional
            Prefix for S3 objects
        """
        self.catalog_path = catalog_path
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        # Initialize catalog
        self.catalog = self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from file or initialize new catalog"""
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse catalog file: {self.catalog_path}")
                return {'datasets': {}}
        elif self.s3_bucket:
            try:
                s3 = boto3.client('s3')
                obj = s3.get_object(Bucket=self.s3_bucket, Key=f"{self.s3_prefix}catalog.json")
                return json.loads(obj['Body'].read().decode('utf-8'))
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.info("No catalog found in S3, initializing new catalog")
                else:
                    logger.warning(f"Error accessing S3: {str(e)}")
                return {'datasets': {}}
        else:
            return {'datasets': {}}
    
    def _save_catalog(self):
        """Save catalog to file and/or S3"""
        # Save locally
        os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
        with open(self.catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=2)
        
        # Save to S3 if configured
        if self.s3_bucket:
            try:
                s3 = boto3.client('s3')
                s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{self.s3_prefix}catalog.json",
                    Body=json.dumps(self.catalog, indent=2),
                    ContentType='application/json'
                )
            except ClientError as e:
                logger.warning(f"Error saving catalog to S3: {str(e)}")
    
    def register_dataset(self, dataset_id, name, description, location, schema=None, tags=None, metadata=None):
        """
        Register a dataset in the catalog
        
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset
        name : str
            Human-readable name
        description : str
            Description of the dataset
        location : str
            Location of the dataset (e.g., S3 path)
        schema : dict, optional
            Schema information
        tags : list, optional
            List of tags
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        dict
            Dataset entry
        """
        dataset_entry = {
            'name': name,
            'description': description,
            'location': location,
            'schema': schema or {},
            'tags': tags or [],
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.catalog['datasets'][dataset_id] = dataset_entry
        self._save_catalog()
        
        return dataset_entry
    
    def update_dataset(self, dataset_id, **kwargs):
        """
        Update a dataset in the catalog
        
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset
        **kwargs : dict
            Fields to update
            
        Returns
        -------
        dict
            Updated dataset entry
        """
        if dataset_id not in self.catalog['datasets']:
            logger.warning(f"Dataset not found: {dataset_id}")
            return None
        
        dataset_entry = self.catalog['datasets'][dataset_id]
        
        # Update fields
        for key, value in kwargs.items():
            if key in dataset_entry:
                dataset_entry[key] = value
        
        # Update timestamp
        dataset_entry['updated_at'] = datetime.now().isoformat()
        
        self._save_catalog()
        
        return dataset_entry
    
    def get_dataset(self, dataset_id):
        """
        Get information about a dataset
        
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset
            
        Returns
        -------
        dict
            Dataset entry
        """
        return self.catalog['datasets'].get(dataset_id)
    
    def list_datasets(self, tags=None):
        """
        List datasets, optionally filtered by tags
        
        Parameters
        ----------
        tags : list, optional
            List of tags to filter by
            
        Returns
        -------
        dict
            Filtered datasets
        """
        if tags is None:
            return self.catalog['datasets']
        
        filtered_datasets = {}
        for dataset_id, dataset in self.catalog['datasets'].items():
            if all(tag in dataset['tags'] for tag in tags):
                filtered_datasets[dataset_id] = dataset
        
        return filtered_datasets
    
    def infer_schema(self, df):
        """
        Infer schema from DataFrame
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to infer schema from
            
        Returns
        -------
        dict
            Schema information
        """
        schema = {
            'columns': [],
            'row_count': len(df)
        }
        
        for column in df.columns:
            column_info = {
                'name': column,
                'type': str(df[column].dtype),
                'nullable': df[column].isna().any(),
                'unique_count': df[column].nunique(),
                'min': str(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'max': str(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None
            }
            schema['columns'].append(column_info)
        
        return schema
    
    def register_from_dataframe(self, dataset_id, df, name, description, location, tags=None, metadata=None):
        """
        Register a dataset from a DataFrame
        
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset
        df : pandas.DataFrame
            DataFrame to register
        name : str
            Human-readable name
        description : str
            Description of the dataset
        location : str
            Location of the dataset
        tags : list, optional
            List of tags
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        dict
            Dataset entry
        """
        schema = self.infer_schema(df)
        
        return self.register_dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            location=location,
            schema=schema,
            tags=tags,
            metadata=metadata
        )
    
    def generate_data_dictionary(self, output_path=None):
        """
        Generate a data dictionary from the catalog
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the data dictionary
            
        Returns
        -------
        pandas.DataFrame
            Data dictionary
        """
        rows = []
        
        for dataset_id, dataset in self.catalog['datasets'].items():
            for column in dataset.get('schema', {}).get('columns', []):
                rows.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset['name'],
                    'column_name': column['name'],
                    'data_type': column['type'],
                    'nullable': column['nullable'],
                    'unique_count': column['unique_count'],
                    'min': column['min'],
                    'max': column['max'],
                    'description': dataset['description']
                })
        
        data_dict = pd.DataFrame(rows)
        
        if output_path:
            data_dict.to_csv(output_path, index=False)
        
        return data_dict 