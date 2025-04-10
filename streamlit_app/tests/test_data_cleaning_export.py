import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'components', 'data_cleaning')))



from export_data import (
    connect_to_database,
    make_query,
    get_column_names,
    drop_views,
    create_views,
    load_sql_to_df,
    export_all_data,
)

db_params = {
    "host": "194.171.191.226",
    "port": "6379",
    "database": "postgres",
    "user": "group6",
    "password": "blockd_2024group6_79",
}

@pytest.fixture
def mock_connection():
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor

def test_connect_to_database_success(mock_connection):
    mock_conn, mock_cursor = mock_connection
    conn = connect_to_database()
    assert conn == mock_conn
    mock_conn.cursor.assert_called_once()

def test_connect_to_database_failure():
    with patch('psycopg2.connect', side_effect=psycopg2.OperationalError):
        conn = connect_to_database()
        assert conn is None

def test_make_query_success(mock_connection):
    _, mock_cursor = mock_connection

    query = "SELECT * FROM test_table"
    result = make_query(query)

    mock_cursor.execute.assert_called_once_with(query)
    assert result == "Query succeeded"

def test_make_query_with_results(mock_connection):
    _, mock_cursor = mock_connection

    mock_cursor.fetchall.return_value = [("row1",), ("row2",)]
    query = "SELECT * FROM test_table"
    result = make_query(query, show_results=True)

    mock_cursor.execute.assert_called_once_with(query)
    assert result == [("row1",), ("row2",)]

def test_get_column_names(mock_connection):
    _, mock_cursor = mock_connection
    mock_cursor.fetchall.return_value = [("column1",), ("column2",)]
    
    with patch('export_data.make_query', return_value=[("column1",), ("column2",)]):
        column_names = get_column_names("test_table")

    assert np.array_equal(column_names, np.array(["column1", "column2"]))

def test_drop_views(mock_connection):
    with patch('export_data.make_query') as mock_make_query:
        drop_views()
        assert mock_make_query.call_count == 5

def test_load_sql_to_df(mock_connection):
    _, mock_cursor = mock_connection

    mock_cursor.fetchall.return_value = [("row1", "row2"), ("row3", "row4")]
    columns = ["col1", "col2"]
    
    with patch('export_data.get_column_names', return_value=columns), \
         patch('export_data.make_query', return_value=[("row1", "row2"), ("row3", "row4")]), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv:
        
        df = load_sql_to_df("test_table")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == columns
        assert not df.empty
        mock_to_csv.assert_called()