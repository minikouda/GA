from .setdata import X_diab, y_diab
import pytest
import numpy as np
import GA
import inspect

def test_output():
    result1 = GA.select(X_diab, y_diab)  # dataframe, series as inputs
    assert isinstance(result1, dict)
    assert 'selected' in result1.keys() and 'R2' in result1.keys() and 'R2pen' in result1.keys()

    # Check that can sum `selected` to get number of predictors selected.
    assert isinstance(np.sum(result1['selected']), (np.int64,np.int32,np.float64))
    
    result2 = GA.select(X_diab, y_diab, penalty=0.1)
    assert isinstance(result2, dict)
    assert 'selected' in result2.keys() and 'R2' in result2.keys() and 'R2pen' in result2.keys()

def test_req_args():
    sig = inspect.signature(GA.select)
    assert "penalty" in sig.parameters
    assert "pop_size" in sig.parameters
    assert "n_gen" in sig.parameters

# Modify this test to test for incorrect input types.
def test_bad_input():
    with pytest.raises(Exception):
        GA.select(y_diab, X_diab)
