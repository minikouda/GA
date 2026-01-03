import numpy as np
import pandas as pd
import os

def load_data(data):
    """
    Load data from various formats into a pandas DataFrame or numpy array.
    
    Args:
        data: Can be a file path (str), pandas DataFrame, numpy array, or list.
        
    Returns:
        pd.DataFrame or np.ndarray
    """
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        
        if data.endswith('.csv'):
            return pd.read_csv(data)
        elif data.endswith('.xlsx') or data.endswith('.xls'):
            return pd.read_excel(data)
        elif data.endswith('.txt') or data.endswith('.dat'):
            # Try space or tab separated
            try:
                return pd.read_csv(data, sep=r'\s+', engine='python')
            except:
                return pd.read_csv(data, sep=',')
        else:
            raise ValueError("Unsupported file format. Please use csv, excel, or txt/dat.")
            
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.copy()
    elif isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def preprocess_data(X, y=None):
    """
    Preprocess predictor and response variables to a unified format (numpy float arrays).
    
    Args:
        X: Predictors. Can be file path, DataFrame, array, etc.
        y: Response. Can be file path, Series, array, list, or a string (column name in X).
        
    Returns:
        X_out (np.ndarray): 2D array of shape (n_samples, n_features).
        y_out (np.ndarray): 1D array of shape (n_samples,).
    """
    # 1. Load X
    X_data = load_data(X)
    
    # 2. Handle y
    y_data = None
    
    # Case: y is a column name in X
    if isinstance(y, str) and isinstance(X_data, pd.DataFrame):
        if y in X_data.columns:
            y_data = X_data[y]
            X_data = X_data.drop(columns=[y])
        else:
            # If y is a string but not in X, maybe it's a file path?
            try:
                y_data = load_data(y)
            except:
                raise ValueError(f"y is a string '{y}' but not a column in X, and could not be loaded as a file.")
    elif y is not None:
        y_data = load_data(y)
        
    # 3. Convert X to numpy 2D array
    if isinstance(X_data, (pd.DataFrame, pd.Series)):
        X_out = X_data.values
    else:
        X_out = np.asarray(X_data)
        
    # Ensure X is 2D
    if X_out.ndim == 1:
        X_out = X_out.reshape(-1, 1)
    elif X_out.ndim > 2:
        raise ValueError(f"X must be 2D, but got shape {X_out.shape}")
        
    # 4. Convert y to numpy 1D array
    if y_data is not None:
        if isinstance(y_data, (pd.DataFrame, pd.Series)):
            y_out = y_data.values
        else:
            y_out = np.asarray(y_data)
            
        # Flatten y if it's 2D (e.g. column vector)
        if y_out.ndim > 1:
            y_out = y_out.ravel()
            
        # Check lengths
        if X_out.shape[0] != y_out.shape[0]:
            raise ValueError(f"Sample size mismatch: X has {X_out.shape[0]} samples, y has {y_out.shape[0]}.")
            
        # Ensure numeric types
        try:
            y_out = y_out.astype(float)
        except ValueError:
             raise ValueError("y contains non-numeric data.")
    else:
        y_out = None

    # Ensure X is numeric
    try:
        X_out = X_out.astype(float)
    except ValueError:
        raise ValueError("X contains non-numeric data. Please encode categorical variables before passing to GA.")

    return X_out, y_out
