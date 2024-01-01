import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

def prepare_features(df, categorical_cols, numerical_cols, target_col):
    """
    Prepare features for regression modeling
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with raw data
    categorical_cols : list
        List of categorical column names
    numerical_cols : list
        List of numerical column names
    target_col : str
        Name of the target column
        
    Returns
    -------
    tuple
        (X, y, preprocessor)
    """
    # Define preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Prepare X and y
    X = df[categorical_cols + numerical_cols]
    y = df[target_col]
    
    return X, y, preprocessor

def train_regression_models(X, y, preprocessor, test_size=0.2, random_state=42):
    """
    Train multiple regression models and evaluate performance
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    preprocessor : ColumnTransformer
        Feature preprocessing pipeline
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary of trained models and their performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Define models to evaluate
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Create pipelines with preprocessing
    pipelines = {
        name: Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        for name, model in models.items()
    }
    
    # Train and evaluate each model
    results = {}
    for name, pipeline in pipelines.items():
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Feature importances (for tree-based models)
        if name in ['random_forest', 'gradient_boosting', 'xgboost']:
            # Get feature names after preprocessing
            feature_names = (
                preprocessor.transformers_[0][1].get_feature_names_out(numerical_cols).tolist() +
                preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
            )
            
            # Get feature importances
            if name == 'xgboost':
                importances = pipeline.named_steps['model'].feature_importances_
            else:
                importances = pipeline.named_steps['model'].feature_importances_
                
            feature_importances = dict(zip(feature_names, importances))
        else:
            feature_importances = None
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'feature_importances': feature_importances
        }
    
    return results

def predict_order_fulfillment(model, new_data):
    """
    Predict order fulfillment metrics using the trained model
    
    Parameters
    ----------
    model : Pipeline
        Trained model pipeline
    new_data : pandas.DataFrame
        New data for prediction
        
    Returns
    -------
    pandas.Series
        Predictions
    """
    # Make predictions
    predictions = model.predict(new_data)
    
    return predictions 