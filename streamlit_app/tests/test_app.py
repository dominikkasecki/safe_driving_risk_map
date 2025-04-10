import pytest
import pandas as pd
import streamlit as st
from pathlib import Path
from unittest.mock import MagicMock, patch
import folium
import sys
import os
from keras.models import load_model as load_keras_model
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions from the provided code
from App import (
    check_if_saved_models_exist,
    get_model_path,
    load_model,
    get_model_features_path,
    read_model_features,
    preprocess_data,
    create_map,
    plot_map,
    main
)

# Fixtures for mock data
@pytest.fixture
def mock_data():
    data = {
        "event_start": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "event_end": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "road_name": ["road1", "road2", "road3"],
        "latitude": [52.1, 52.2, 52.3],
        "longitude": [0.1, 0.2, 0.3],
        "category": ["low", "high", "low"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0, 1, 0]
    return model

# Test check_if_saved_models_exist
def test_check_if_saved_models_exist(mocker):
    mocker.patch('pathlib.Path.iterdir', return_value=[Path("weights/model.pkl")])
    assert check_if_saved_models_exist() is True

    mocker.patch('pathlib.Path.iterdir', return_value=[])
    assert check_if_saved_models_exist() is False

# Test get_model_path
def test_get_model_path():
    assert get_model_path("neural_network") == "./weights/neural_network.h5"
    assert get_model_path("random_forest") == "./weights/random_forest.pkl"



# Test get_model_features_path
def test_get_model_features_path():
    assert get_model_features_path("neural_network") == "./model_features/neural_network_features.txt"

# Test read_model_features
def test_read_model_features(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data="['feature1', 'feature2']"))
    assert read_model_features("./model_features/neural_network_features.txt") == ['feature1', 'feature2']

# Test preprocess_data
def test_preprocess_data(mock_data):
    model_features = ["latitude", "longitude", "category"]
    is_n_network = True
    processed_data = preprocess_data(mock_data, model_features, is_n_network)
    assert not processed_data.empty
    assert set(processed_data.columns) == set(["road_name", "latitude", "longitude", "category"])

# Test create_map
def test_create_map(mock_data):
    averaged_data = pd.DataFrame({
        "road_name": ["road1", "road2"],
        "risk_level": [0, 1]
    })
    map_object = create_map(mock_data, averaged_data)
    assert isinstance(map_object, folium.Map)

# Test plot_map
def test_plot_map(mocker, mock_data):
    averaged_data = pd.DataFrame({
        "road_name": ["road1", "road2"],
        "risk_level": [0, 1]
    })
    st.session_state.data = mock_data
    st.session_state.map_data = averaged_data

    mocker.patch("streamlit_folium.folium_static")
    plot_map()
    assert "map_data" in st.session_state
