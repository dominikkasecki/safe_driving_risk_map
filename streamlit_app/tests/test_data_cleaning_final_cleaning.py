import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import sys

# Add the path to the src/components/data_cleaning directory
# sys.path.append('../src/components/data_cleaning')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'components', 'data_cleaning')))

from final_cleaning import (
    import_df,
    import_weather_df,
    clean_categorical_data,
    delete_empty_columns,
    show_dataframe_general_info,
    show_value_counts,
    show_dataframe_column_value_counts,
    show_general_duplicate_values,
    show_duplicated_values_in_column,
    drop_duplicates_in_df,
    plot_columns,
    plot_numeric_columns,
    plot_string_columns,
    plot_bool_columns,
    plot_value_distributions_in_df,
    delete_outliers,
    drop_columns_in_df,
    convert_column_to_binary,
    clip_numerical_cols,
    clean_numerical_cols,
    scale_numerical_data,
    convert_unknown_to_nan,
    convert_empty_values_to_nan,
    drop_rows_with_drop_values,
    convert_string_column_to_numerical,
    show_columns_with_missing_values,
    transform_acc_sev_col_to_encoding,
    calc_weighted_mean_of_acc_severity,
    export_cleaned_data_to_csv,
    transform_numerical_column_to_str
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

def test_import_df(mock_df):
    with patch('pandas.read_csv', return_value=mock_df) as mock_read_csv:
        df = import_df('test')
        mock_read_csv.assert_called_once()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_import_weather_df(mock_df):
    with patch('final_cleaning.import_df', return_value=mock_df) as mock_import_df:
        df = import_weather_df('test')
        mock_import_df.assert_called_once()
        assert isinstance(df, pd.DataFrame)

def test_clean_categorical_data(mock_df):
    mock_df['col2'] = mock_df['col2'].astype(str)
    df = clean_categorical_data(mock_df)
    for col in df.columns:
        if col != 'dtg':
            assert df[col].dtype == 'object'


def test_delete_empty_columns(mock_df):
    df = delete_empty_columns(mock_df)
    assert df.shape[1] == 4  # Since mock_df does not have any empty columns

def test_show_dataframe_general_info(mock_df, capsys):
    show_dataframe_general_info(mock_df)
    captured = capsys.readouterr()
    assert "General info of df" in captured.out

def test_show_value_counts(mock_df, capsys):
    show_value_counts(mock_df, 'col1')
    captured = capsys.readouterr()
    assert "Value counts of col1" in captured.out

def test_show_dataframe_column_value_counts(mock_df, capsys):
    show_dataframe_column_value_counts(mock_df)
    captured = capsys.readouterr()
    assert "Value counts of" in captured.out

def test_show_general_duplicate_values(mock_df, capsys):
    show_general_duplicate_values(mock_df)
    captured = capsys.readouterr()
    assert "No duplicated values in this dataframe !!!" in captured.out

def test_show_duplicated_values_in_column(mock_df, capsys):
    show_duplicated_values_in_column(mock_df, 'col1')
    captured = capsys.readouterr()
    assert "Show duplicated values in column: col1" in captured.out

def test_drop_duplicates_in_df(mock_df):
    drop_duplicates_in_df(mock_df, ['col1'])
    assert not mock_df.duplicated().any()

def test_plot_columns(mock_df):
    plot_columns(mock_df, ['col2'], sns.countplot)

def test_plot_numeric_columns(mock_df):
    plot_numeric_columns(mock_df, ['col2'])

def test_plot_string_columns(mock_df):
    plot_string_columns(mock_df, ['col1'])

def test_plot_bool_columns():
    bool_df = pd.DataFrame({'col1': [True, False, True]})
    plot_bool_columns(bool_df, ['col1'])

def test_plot_value_distributions_in_df(mock_df):
    plot_value_distributions_in_df(mock_df)

def test_delete_outliers(mock_df):
    df = delete_outliers(mock_df, ['col2'])
    assert isinstance(df, pd.DataFrame)

def test_drop_columns_in_df(mock_df):
    drop_columns_in_df(mock_df, ['col1'])
    assert 'col1' not in mock_df.columns

def test_convert_column_to_binary(mock_df):
    columns_with_new_values = {
        "col1": {"new_value": "other", "top_values": 1}
    }
    convert_column_to_binary(mock_df, columns_with_new_values)
    assert "col1" in mock_df.columns

def test_clip_numerical_cols(mock_df):
    clip_numerical_cols(mock_df, ['col2'])
    assert mock_df['col2'].equals(pd.Series([1, 2, 3], name='col2'))

def test_clean_numerical_cols(mock_df):
    clean_numerical_cols(mock_df)
    assert mock_df['col2'].dtype == float

def test_scale_numerical_data(mock_df):
    scale_numerical_data(mock_df, ['col2'])
    assert mock_df['col2'].mean() == pytest.approx(0, abs=1e-9)



def test_convert_unknown_to_nan(mock_df):
    df = mock_df.copy()
    df.iloc[0, 0] = 'unknown'
    convert_unknown_to_nan(df)
    assert pd.isna(df.iloc[0, 0])

def test_convert_empty_values_to_nan(mock_df):
    df = mock_df.copy()
    df.iloc[0, 0] = ''
    convert_empty_values_to_nan(df, ['col1'])
    assert pd.isna(df.iloc[0, 0])

def test_drop_rows_with_drop_values(mock_df):
    df = mock_df.copy()
    df.iloc[0, 0] = 'drop'
    drop_rows_with_drop_values(df, 'col1', ['drop'])
    assert 'drop' not in df['col1'].values

def test_convert_string_column_to_numerical(mock_df):
    df = pd.DataFrame({'col1': ['10 km/h', '20 km/h', '30 km/h']})
    convert_string_column_to_numerical(df, 'col1')
    assert df['col1'].dtype == float

def test_show_columns_with_missing_values(mock_df, capsys):
    df = mock_df.copy()
    df.iloc[0, 0] = np.nan
    show_columns_with_missing_values(df)
    captured = capsys.readouterr()
    assert "missing values" in captured.out

def test_transform_acc_sev_col_to_encoding(mock_df):
    df = pd.DataFrame({'accident_severity': ['minor', 'major', 'minor']})
    df = transform_acc_sev_col_to_encoding(df)
    assert 'minor' in df.columns and 'major' in df.columns

def test_calc_weighted_mean_of_acc_severity(mock_df):
    df = pd.DataFrame({
        'street': ['street1', 'street2'],
        'accident_severity': ['minor', 'major'],
        'injury or fatal': [1, 2],
        'material damage only': [3, 4]
    })
    result = calc_weighted_mean_of_acc_severity(df)
    assert 'weighted_avg' in result.columns

def test_export_cleaned_data_to_csv(mock_df):
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        export_cleaned_data_to_csv([mock_df], ['test'])
        mock_to_csv.assert_called_once()

def test_transform_numerical_column_to_str(mock_df):
    df = transform_numerical_column_to_str(mock_df, ['col2'])
    assert df['col2'].dtype == 'object'

