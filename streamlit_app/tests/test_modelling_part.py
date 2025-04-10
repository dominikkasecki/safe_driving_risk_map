import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add the module path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'components', 'modelling')))

# Import functions from the module
from modelling_part import (
    load_data,
    import_df,
    preprocess_data,
    create_preprocessor,
    plot_evaluation_metrics,
    format_metrics,
    train_and_evaluate_model,
    save_model_weights,
    run_models,
)

@pytest.fixture
def mock_df():
    data = {
        'col1': ['a', 'b', 'c'],
        'col2': [1, 2, 3],
        'col3': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'dtg': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    }
    return pd.DataFrame(data)

def test_load_data(mock_df):
    with patch('pandas.read_csv', return_value=mock_df) as mock_read_csv:
        df = load_data('test.csv')
        mock_read_csv.assert_called_once_with('test.csv')
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_import_df(mock_df):
    with patch('pandas.read_csv', return_value=mock_df) as mock_read_csv:
        df = import_df('test')
        mock_read_csv.assert_called_once()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty




def test_create_preprocessor(mock_df):
    X, _, _ = preprocess_data(mock_df, 'col1', ['dtg'])
    preprocessor = create_preprocessor(X)
    assert isinstance(preprocessor, ColumnTransformer)

def test_plot_evaluation_metrics(mock_df):
    y_test = [0, 1, 0]
    y_pred = [0, 1, 1]
    evaluation_metric = 'Confusion matrix'
    with patch('matplotlib.pyplot.show'):
        plot_evaluation_metrics(y_test, y_pred, evaluation_metric)



def test_save_model_weights(mock_df):
    model = MagicMock()
    model_name = 'xgboost'
    original_x_data = mock_df.drop(columns=['col1'])
    with patch('joblib.dump') as mock_joblib_dump, patch('builtins.open', new_callable=MagicMock):
        save_model_weights(model, model_name, original_x_data)
        mock_joblib_dump.assert_called_once()
        assert mock_joblib_dump.call_args[0][1].endswith(f'{model_name}.pkl')


