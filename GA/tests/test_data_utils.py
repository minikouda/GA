import pytest
import numpy as np
import pandas as pd
import os
from GA.data_utils import load_data, preprocess_data

class TestDataUtils:
    def setup_method(self):
        # Create dummy data
        self.X_np = np.random.rand(10, 5)
        self.y_np = np.random.rand(10)
        
        self.df = pd.DataFrame(self.X_np, columns=[f'col_{i}' for i in range(5)])
        self.df['target'] = self.y_np
        
        # Create a temporary CSV file
        self.csv_path = 'test_data.csv'
        self.df.to_csv(self.csv_path, index=False)

    def teardown_method(self):
        # Clean up
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_load_data_csv(self):
        loaded = load_data(self.csv_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == (10, 6) # 5 features + 1 target

    def test_preprocess_numpy(self):
        X_out, y_out = preprocess_data(self.X_np, self.y_np)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)
        assert X_out.shape == (10, 5)
        assert y_out.shape == (10,)

    def test_preprocess_dataframe_split(self):
        # Pass X and y separately as DF/Series
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        X_out, y_out = preprocess_data(X, y)
        assert X_out.shape == (10, 5)
        assert y_out.shape == (10,)

    def test_preprocess_dataframe_colname(self):
        # Pass X as DF and y as column name
        X_out, y_out = preprocess_data(self.df, 'target')
        assert X_out.shape == (10, 5)
        assert y_out.shape == (10,)
        
    def test_preprocess_csv_colname(self):
        # Pass file path and column name
        X_out, y_out = preprocess_data(self.csv_path, 'target')
        assert X_out.shape == (10, 5)
        assert y_out.shape == (10,)

    def test_preprocess_list(self):
        X_list = [[1, 2], [3, 4]]
        y_list = [0, 1]
        X_out, y_out = preprocess_data(X_list, y_list)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == (2, 2)
