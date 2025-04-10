import pytest
from unittest import mock
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch
from src.components.eda.eda import (
    import_df,
    plot_pair_plots,
    plot_categorical_value_distributions,
    plot_box_plots,
    cramers_v,
    calculate_cramers_v_matrix
)

# Sample data for testing
TEST_CSV = """
A,B,C,D
1,2,3,4
5,6,7,8
9,10,11,12
"""

@pytest.fixture
def sample_dataframe():
    return pd.read_csv(StringIO(TEST_CSV))

def test_import_df(mocker):
    # Mock the path and read_csv function
    mocker.patch('pandas.read_csv', return_value=pd.read_csv(StringIO(TEST_CSV)))
    mocker.patch('os.path.exists', return_value=True)
    
    df = import_df(name="test")

    assert not df.empty
    assert list(df.columns) == ['A', 'B', 'C', 'D']
    assert df.shape == (3, 4)




def test_cramers_v():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a', 'b', 'a'],
        'B': ['x', 'x', 'y', 'y', 'x']
    })
    
    result = cramers_v(df['A'], df['B'])
    
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_calculate_cramers_v_matrix():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a', 'b', 'a'],
        'B': ['x', 'x', 'y', 'y', 'x'],
        'C': ['u', 'u', 'v', 'v', 'u']
    })
    
    result = calculate_cramers_v_matrix(df, ['A', 'B', 'C'])
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert all(result.columns == ['A', 'B', 'C'])
    assert all(result.index == ['A', 'B', 'C'])
