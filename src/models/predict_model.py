import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json

def load_model(filepath):
    """
    Load a model from a file
    
    Parameters
    ----------
    filepath : str
        Path to the saved model
        
    Returns
    -------
    object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model

def preprocess_data(df, scaler=None):
    """
    Preprocess data for prediction
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw data to preprocess
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler to use for feature scaling
        
    Returns
    -------
    numpy.ndarray
        Preprocessed data
    """
    # Convert categorical variables to dummies
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Scale features if scaler is provided
    if scaler is not None:
        df_scaled = scaler.transform(df)
        return df_scaled
    
    return df.values

def predict_lead_times(model, df, scaler=None):
    """
    Make lead time predictions
    
    Parameters
    ----------
    model : object
        Trained model
    df : pandas.DataFrame
        Data to predict on
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler to use for feature scaling
        
    Returns
    -------
    numpy.ndarray
        Predicted lead times
    """
    # Preprocess data
    X = preprocess_data(df, scaler)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

def predict_with_uncertainty(model, df, scaler=None, n_bootstrap=100):
    """
    Make predictions with uncertainty estimation
    
    Parameters
    ----------
    model : object
        Trained model
    df : pandas.DataFrame
        Data to predict on
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler to use for feature scaling
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Dictionary with predictions and confidence intervals
    """
    # Preprocess data
    X = preprocess_data(df, scaler)
    
    # Make base prediction
    base_predictions = model.predict(X)
    
    # Only certain model types support bootstrapping
    if hasattr(model, 'estimators_'):
        bootstrap_predictions = np.zeros((X.shape[0], n_bootstrap))
        
        # Use trained estimators to create bootstrap samples
        for i in range(n_bootstrap):
            # Randomly select estimators with replacement
            indices = np.random.randint(0, len(model.estimators_), size=len(model.estimators_))
            selected_estimators = [model.estimators_[idx] for idx in indices]
            
            # Make predictions with selected estimators
            bootstrap_predictions[:, i] = np.mean([est.predict(X) for est in selected_estimators], axis=0)
        
        # Calculate confidence intervals
        lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=1)
        upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=1)
        
        result = {
            'predictions': base_predictions,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        }
    else:
        # For models without estimators, return only base predictions
        result = {
            'predictions': base_predictions
        }
    
    return result

def save_predictions(predictions, output_path='predictions.json'):
    """
    Save predictions to a file
    
    Parameters
    ----------
    predictions : dict
        Dictionary with predictions and confidence intervals
    output_path : str
        Path to save the predictions
        
    Returns
    -------
    None
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_predictions = {}
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            serializable_predictions[key] = value.tolist()
        else:
            serializable_predictions[key] = value
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_predictions, f, indent=4)
    
    print(f"Predictions saved to {output_path}") 