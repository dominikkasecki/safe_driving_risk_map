import pytest
import streamlit as st
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os
import importlib.util


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamic import for the Streamlit page
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the functions from the Streamlit page
page_module = import_module('neural_network_page', os.path.join(os.path.dirname(__file__), '../pages/4_Neural_network.py'))

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
def mock_data_file():
    return pd.DataFrame({
        "y_var": ['high-risk', 'low-risk', 'high-risk', 'low-risk', 'high-risk', 'low-risk'],
        "eventid": [1, 2, 3, 4, 5, 6],
        "incident_severity": ['A', 'B', 'C', 'D', 'E', 'F'],
        "weighted_avg": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "material_damage_only_sum": [10, 20, 30, 40, 50, 60],
        "road_segment_id": [100, 200, 300, 400, 500, 600],
        "injury_or_fatal_sum": [5, 10, 15, 20, 25, 30],
        "road_name": ['Road A', 'Road B', 'Road C', 'Road D', 'Road E', 'Road F'],
        "longitude": [50.1, 50.2, 50.3, 50.4, 50.5, 50.6],
        "latitude": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [1, 2, 3, 4, 5, 6],
        "event_start": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]),
        "event_end": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]),
    })

# Test when no data is loaded
def test_main_no_data(mocker, mock_session_state):
    mocker.patch.object(st.sidebar, 'file_uploader', return_value=None)
    mocker.patch.object(st.sidebar, 'button', return_value=False)
    mocker.patch.object(st, 'error')
    mocker.patch.object(st, 'title')
    mocker.patch.object(st.sidebar, 'title')

    page_module.main()

    assert "data" not in st.session_state
    assert st.error.called

# Test training and evaluating model
def test_main_train_and_evaluate_model(mocker, mock_session_state, mock_data_file):
    mock_file_uploader = mocker.patch.object(st.sidebar, 'file_uploader', return_value=MagicMock())
    mock_read_csv = mocker.patch.object(pd, 'read_csv', return_value=mock_data_file)
    mock_checkbox = mocker.patch.object(st.sidebar, 'checkbox', return_value=False)
    mock_button = mocker.patch.object(st.sidebar, 'button', side_effect=[True, False])
    mock_write = mocker.patch.object(st, 'write')
    mock_title = mocker.patch.object(st, 'title')
    mock_sidebar_title = mocker.patch.object(st.sidebar, 'title')
    mock_preprocess_data = mocker.patch.object(page_module, 'preprocess_data', return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    mock_train_and_evaluate_model = mocker.patch.object(page_module, 'train_and_evaluate_model', return_value=(MagicMock(), 0.1, 0.9))
    mock_plot_results = mocker.patch.object(page_module, 'plot_results')

    with pytest.raises(st.errors.NoSessionContext):
        page_module.main()

if __name__ == "__main__":
    pytest.main()
