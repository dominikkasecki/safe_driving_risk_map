import pytest
import streamlit as st
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os
import importlib.util
from streamlit.runtime.scriptrunner import ScriptRunContext
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamic import for the Streamlit page
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the functions from the Streamlit page
page_module = import_module('model_page', os.path.join(os.path.dirname(__file__), '../pages/3_Modelling.py'))

# Fixtures for mock data
@pytest.fixture
def mock_session_state(monkeypatch):
    # Create a fresh session state
    mock_session_state = MagicMock()
    monkeypatch.setattr(st, 'session_state', mock_session_state)

    # Create a mock context with necessary attributes
    mock_context = MagicMock()
    mock_context.session_id = "test_session_id"
    mock_context._enqueue = lambda msg: None
    mock_context.query_string = ""
    mock_context.session_state = mock_session_state
    mock_context.uploaded_file_mgr = MagicMock()
    mock_context.main_script_path = ""
    mock_context.page_script_hash = ""
    mock_context.user_info = {}
    mock_context.fragment_storage = None

    def mock_get_script_run_ctx(*args, **kwargs):
        return mock_context


@pytest.fixture
def mock_import_df():
    return MagicMock(return_value=pd.DataFrame({
        "event_start": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "event_end": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "y_var": [0, 1, 0],
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1, 2, 3]
    }))

@pytest.fixture
def mock_run_models():
    return MagicMock()


# Test when data is not in session state
def test_main_no_data(mocker, mock_session_state):
    mocker.patch.object(page_module, 'redirect_to_page')
    mocker.patch('streamlit.file_uploader', return_value=None)
    mocker.patch('streamlit.sidebar.file_uploader', return_value=None)
    mocker.patch('streamlit.sidebar.button', return_value=False)
    mocker.patch('streamlit.error')
    mocker.patch('streamlit.markdown')
    mocker.patch('streamlit.write')
    mocker.patch('streamlit.subheader')
    mocker.patch('streamlit.title')

    page_module.main()

    assert "data" not in st.session_state
    assert st.error.called
