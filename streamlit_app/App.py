import ast
import pandas as pd
import streamlit as st

import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from pathlib import Path

from utils.redirect import redirect_to_page
from streamlit_folium import folium_static
from branca.element import Template, MacroElement
import random
from keras.models import load_model as load_keras_model
import numpy as np
from sklearn import preprocessing
st.set_page_config(layout="wide")

def check_if_saved_models_exist():
    """Check if saved models exist and update the Streamlit session state.

    Returns:
        bool: True if models exist, False otherwise.
    """
    if not any(Path("weights").iterdir()):
        st.write("Models do not exist")
        st.error("Train models by going to modelling page")
        return False
    elif any(Path("weights").iterdir()) and "data" not in st.session_state:
        st.success("Models exist")
        return True
    else:
        return True

@st.cache_data
def get_model_path(model_name):
    """Get the file path for a given model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        str: Path to the model file.
    """
    if 'neural_network' in model_name.lower():
        return f"./weights/{model_name}.h5"
    else:
        return f"./weights/{model_name}.pkl"

def load_model(model_path):
    """Load a model from a given file path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        model: Loaded model.
    """
    if model_path.endswith('.h5'):
        return load_keras_model(model_path)
    else:
        return joblib.load(model_path)

@st.cache_data
def get_model_features_path(model_name):
    """Get the file path for the features of a given model.

    Args:
        model_name (str): Name of the model.

    Returns:
        str: Path to the model features file.
    """
    return f"./model_features/{model_name}_features.txt"

def read_model_features(model_path):
    """Read model features from a file.

    Args:
        model_path (str): Path to the model features file.

    Returns:
        list: List of model features.
    """
    with open(model_path, 'r') as file:
        features = file.read()
        cols_list = ast.literal_eval(features)
        return cols_list

def preprocess_data(data, model_features, is_n_network):
    """Preprocess the input data for model prediction.

    Args:
        data (DataFrame): Input data.
        model_features (list): List of model features.
        is_n_network (bool): Flag indicating if the model is a neural network.

    Returns:
        DataFrame: Preprocessed data.
    """
    # Convert event_start and event_end to datetime
    data["event_start"] = pd.to_datetime(data["event_start"])
    data["event_end"] = pd.to_datetime(data["event_end"])

    # Select relevant columns and drop missing values
    data = data.loc[:, model_features + ["road_name", "event_start"]].dropna()

    if is_n_network:
        # Convert numerical values to float32
        num_cols = data.select_dtypes(include=[float]).columns
        data[num_cols] = data[num_cols].astype(np.float32)

        cat_cols = data.select_dtypes(include=['object']).columns
        label_encoder = preprocessing.LabelEncoder()
        
        for col in cat_cols:
            if col != 'road_name':
                data[col] = label_encoder.fit_transform(data[col])

    # Remove duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]

    # Sort data by event_start in descending order
    data = data.sort_values(by="event_start", ascending=False)

    # Group by road_name and get the top 10 records for each group
    top_10_per_street = data.groupby("road_name").head(10)

    # Identify numeric columns
    numeric_columns = data[model_features].select_dtypes(include="number").columns.tolist()
    numeric_columns = list(filter(lambda col_name: col_name != 'category', numeric_columns))

    # Compute mean for numeric columns
    mean_values = top_10_per_street.groupby("road_name")[numeric_columns].mean()

    # Compute mode for 'category' column
    mode_values = top_10_per_street.groupby("road_name")["category"].agg(
        lambda x: x.mode().iloc[0]
    )

    # Combine mean and mode values
    averaged_data = mean_values.join(mode_values, on="road_name", rsuffix='_mode').reset_index()

    # Use only 600 streets for optimization purposes
    return averaged_data.iloc[:300, :]

def create_map(data, averaged_data):
    """Create a map with markers based on the input data.

    Args:
        data (DataFrame): Input data.
        averaged_data (DataFrame): Preprocessed data with risk levels.

    Returns:
        folium.Map: Folium map with markers.
    """
    with st.spinner('Creating map....'):
        m = folium.Map(
            location=[data["latitude"].mean(), data["longitude"].mean()],
            zoom_start=12,
        )
    
        # Add a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
    
        # Add points to the map
        for idx, row in averaged_data.iterrows():
            street_data = data[data["road_name"] == row["road_name"]]
            if not street_data.empty:
                latitude = street_data["latitude"].values[0]
                longitude = street_data["longitude"].values[0]
                color = "blue" if row["risk_level"] == 0 else "red"
                folium.Marker(
                    location=[latitude, longitude],
                    popup=str(row["road_name"]).capitalize(),
                    icon=folium.Icon(color=color),
                ).add_to(marker_cluster)
    
        return m

def plot_map():
    """Plot the map with risk levels for streets using Folium and Streamlit."""
    data = st.session_state.data
    averaged_data = st.session_state.map_data

    legend_html = """
        <div style='padding: 10px; margin-bottom: 10px; background-color: white; 
                    border:2px solid grey; border-radius:5px; font-size:14px; width: 250px;'>
            <b style='color: black;'>Risk Level</b><br>
            <div style='display: flex; align-items: center;'><div style='height: 16px; width: 16px; background-color: #38A9DB; border-radius: 50%; margin-right: 8px;'></div><span style='color: black;'>Low Risk</span></div>
            <div style='display: flex; align-items: center;'><div style='height: 16px; width: 16px; background-color: #D43E2A; border-radius: 50%; margin-right: 8px;'></div><span style='color: black;'>High Risk</span></div>
        </div>
        """
    st.markdown(legend_html, unsafe_allow_html=True)

    map_object = create_map(data, averaged_data)

    # Display the map using st_folium with a full_width set to True
    folium_static(map_object, width=1200, height=600)

def main():
    """Main function to run the Streamlit app."""
    st.title("Binary Classification Visualiser")
    st.subheader("Map with streets and their respective risk levels")

    if not check_if_saved_models_exist() or "data" not in st.session_state:
        st.error("Please load data first by clicking the button on the left side (Load Data)")
        if st.sidebar.button("Load Data"):
            redirect_to_page("pages/1_Load_data.py")
    elif check_if_saved_models_exist() and "data" in st.session_state:
        st.write("Choose model to show map for")
        model_names = [
            str(model.stem).replace("_", " ").capitalize()
            for model in Path("weights").iterdir() if 'hist' not in str(model.stem)
        ]

        def delete_key_in_dict(key, received_dict):
            """Delete a key from a dictionary if it exists.

            Args:
                key (str): Key to delete.
                received_dict (dict): Dictionary from which to delete the key.
            """
            if key in received_dict:
                del received_dict[key]

        model_name = st.selectbox("Model", model_names, placeholder="Select model", on_change=lambda: delete_key_in_dict('map_data', st.session_state))
        model_name = model_name.replace(" ", "_").lower()
        st.session_state.model_name = model_name

        model_path = get_model_path(st.session_state.model_name)
        is_neural_network = str(model_path).endswith('.h5')
        st.session_state.loaded_model = load_model(model_path)

        if st.button("Show map"):
            model_features_path = get_model_features_path(st.session_state.model_name)
            model_features = read_model_features(model_features_path)
            data = st.session_state.data
            averaged_data = preprocess_data(data, model_features, is_neural_network)
            X_data = averaged_data.loc[:, model_features].copy()
            loaded_model = st.session_state.loaded_model

            with st.spinner(f"Using {model_name} to draw points ...."):
                averaged_data["risk_level"] = loaded_model.predict(X_data)
                st.session_state.map_data = averaged_data

    if "map_data" in st.session_state and "loaded_model" in st.session_state and "data" in st.session_state:
        with st.container():
            plot_map()

if __name__ == "__main__":
    main()
