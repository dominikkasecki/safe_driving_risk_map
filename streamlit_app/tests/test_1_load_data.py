import pytest
import streamlit as st
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamic import for 1_Load_data.py
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the functions from 1_Load_data.py
load_data_module = import_module('load_data', os.path.join(os.path.dirname(__file__), '../pages/1_Load_data.py'))

# Fixtures for mock data
@pytest.fixture
def mock_session_state():
    st.session_state.clear()
    yield st.session_state
    st.session_state.clear()

@pytest.fixture
def mock_import_df():
    return MagicMock(return_value=MagicMock())

# Test load_original_data
def test_load_original_data(mocker):
    mock_export = mocker.patch.object(load_data_module, 'export_all_data')
    mock_success = mocker.patch('streamlit.success')
    load_data_module.load_original_data()
    mock_export.assert_called_once()
    mock_success.assert_called_once_with('Original data retrieved successfully')

# Test load_clean_data
def test_load_clean_data(mocker):
    mock_clean = mocker.patch.object(load_data_module, 'clean_all_data')
    mock_success = mocker.patch('streamlit.success')
    load_data_module.load_clean_data()
    mock_clean.assert_called_once()
    mock_success.assert_called_once_with('Cleaning data successfully')

# Test import_data_to_session_state
def test_import_data_to_session_state(mocker, mock_import_df, mock_session_state):
    mocker.patch.object(load_data_module, 'import_df', mock_import_df)
    load_data_module.import_data_to_session_state()
    assert "data" in st.session_state




# Test check_if_saved_models_exist
def test_check_if_saved_models_exist(mocker):
    mock_path = mocker.patch('pathlib.Path.iterdir', return_value=[Path("weights/model.pkl")])
    assert load_data_module.check_if_saved_models_exist() is True

    mock_path = mocker.patch('pathlib.Path.iterdir', return_value=[])
    assert load_data_module.check_if_saved_models_exist() is False
