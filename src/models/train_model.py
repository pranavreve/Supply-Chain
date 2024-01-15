import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def prepare_features(df, target_column='lead_time', test_size=0.2, random_state=42):
    """
    Prepare features for model training
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with processed supply chain data
    target_column : str
        Column name of the target variable
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    
    # Convert categorical variables to dummies
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Define features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

def train_linear_model(X_train, y_train, cv=5):
    """
    Train a linear regression model
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray or pandas.Series
        Training target variable
    cv : int
        Number of cross-validation folds
        
    Returns
    -------
    sklearn.linear_model
        Trained linear model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    
    print(f"Linear Regression - Cross-validated RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    
    return model

def train_xgboost_model(X_train, y_train, X_test, y_test, cv=5):
    """
    Train an XGBoost regression model with hyperparameter tuning
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray or pandas.Series
        Training target variable
    X_test : numpy.ndarray
        Testing features
    y_test : numpy.ndarray or pandas.Series
        Testing target variable
    cv : int
        Number of cross-validation folds
        
    Returns
    -------
    xgboost.XGBRegressor
        Trained XGBoost model
    """
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Base model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost - Test RMSE: {rmse:.4f}")
    print(f"XGBoost - Test MAE: {mae:.4f}")
    print(f"XGBoost - Test R²: {r2:.4f}")
    
    return best_model

def save_model(model, filename, directory='models'):
    """
    Save model to file
    
    Parameters
    ----------
    model : object
        Trained model to save
    filename : str
        Name of the file to save the model
    directory : str
        Directory to save the model
        
    Returns
    -------
    str
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Full path
    filepath = os.path.join(directory, filename)
    
    # Save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")
    
    return filepath 