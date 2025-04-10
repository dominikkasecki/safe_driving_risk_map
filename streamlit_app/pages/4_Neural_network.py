import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from utils.redirect import redirect_to_page
from sklearn import preprocessing

def save_model_weights(model, model_name, original_x_data):
    """Save the trained model and feature names.

    Args:
        model (Sequential): The trained model.
        model_name (str): The name of the model.
        original_x_data (DataFrame): The original input data.
    """
    current_path = Path(__file__).resolve().parent
    
    # Save feature names to a text file
    with open(f"{current_path.parent}/model_features/{model_name}_features.txt", "w") as f:
        f.write(str(original_x_data.columns.tolist()))

    # Save the model weights to an HDF5 file
    model.save(f"{current_path.parent}/weights/{model_name}.h5")
    print(f"Model saved to {model_name}.h5")

def load_model_weights(model_name):
    """Load the model weights from an HDF5 file.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Sequential: The loaded model.
    """
    current_path = Path(__file__).resolve().parent
    model_path = f"{current_path.parent}/weights/{model_name}.h5"
    model = load_model(model_path)
    print(f"Model loaded from {model_name}.h5")
    return model

def save_training_history(history, filename):
    """Save the training history to a JSON file.

    Args:
        history (History): The training history.
        filename (str): The name of the file to save the history.
    """
    history_dict = history.history
    with open(filename, 'w') as f:
        json.dump(history_dict, f)

def load_training_history(filename):
    """Load the training history from a JSON file.

    Args:
        filename (str): The name of the file to load the history from.

    Returns:
        dict: The training history.
    """
    with open(filename, 'r') as f:
        history_dict = json.load(f)
    return history_dict

def preprocess_data(df):
    """Preprocess the input data.

    Args:
        df (DataFrame): The input data.

    Returns:
        tuple: Tuple containing training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Copy the dataframe and limit to the first 10,000 rows
    df = df.copy().iloc[:10000, :]

    df = df.dropna()

    # Columns to drop
    X_droplist = ["y_var", "event_start", "event_end", "eventid", "incident_severity", "weighted_avg", "material_damage_only_sum", "road_segment_id", "injury_or_fatal_sum", "road_name", "longitude", "latitude"]

    # Binary encode the target variable
    y = df['y_var'].apply(lambda row: 1 if row == 'high-risk' else 0)

    # Drop unnecessary columns and one-hot encode categorical variables
    X = df.drop(columns=X_droplist)

    label_encoder = preprocessing.LabelEncoder()

    cat_cols = X.select_dtypes(include=['object']).columns

    for col in cat_cols:
        X[col] = label_encoder.fit_transform(X[col])

    num_cols = X.select_dtypes(include=['float']).columns
    X[num_cols] = X[num_cols].astype('float32')

    # Ensure no NaN values are present
    if X.isnull().any().any():
        st.error("Data contains NaN values. Please clean the data.")
        return None, None, None, None, None, None

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(hp=None, X_train=None, units_1=None, units_2=None, learning_rate=None):
    """Build the neural network model.

    Args:
        hp (HyperParameters, optional): Hyperparameters from Keras Tuner. Defaults to None.
        X_train (DataFrame, optional): Training data. Defaults to None.
        units_1 (int, optional): Number of units in the first layer. Defaults to None.
        units_2 (int, optional): Number of units in the second layer. Defaults to None.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to None.

    Returns:
        Sequential: The built model.
    """
    model = Sequential()
    
    if hp:
        units_1 = hp.Int('units_1', min_value=32, max_value=64, step=8)
        units_2 = hp.Int('units_2', min_value=8, max_value=32, step=8)
        learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
    
    model.add(Dense(units_1, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(units_2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, units_1, units_2, learning_rate):
    """Train and evaluate the model.

    Args:
        X_train (DataFrame): Training data.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation data.
        y_val (Series): Validation labels.
        X_test (DataFrame): Test data.
        y_test (Series): Test labels.
        units_1 (int): Number of units in the first layer.
        units_2 (int): Number of units in the second layer.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: The trained model, training history, test loss, and test accuracy.
    """
    
    
    model = build_model(X_train=X_train, units_1=units_1, units_2=units_2, learning_rate=learning_rate)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    hist = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test)

    return model, hist, test_loss, test_acc

def tune_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Perform hyperparameter tuning using Keras Tuner.

    Args:
        X_train (DataFrame): Training data.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation data.
        y_val (Series): Validation labels.
        X_test (DataFrame): Test data.
        y_test (Series): Test labels.

    Returns:
        tuple: The best model, training history, test loss, and test accuracy.
    """
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp, X_train),
        objective='val_accuracy',
        max_trials=20,
        directory='my_dir',
        project_name='kt_tuner',
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    units_1 = best_hps.get('units_1')
    units_2 = best_hps.get('units_2')
    learning_rate = best_hps.get('learning_rate')

    model, hist, test_loss, test_acc = train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, units_1, units_2, learning_rate)
    
    return model, hist, test_loss, test_acc

def plot_results(history, X_test, y_test, model):
    """Plot the training results and evaluation metrics.

    Args:
        history (dict): Training history.
        X_test (DataFrame): Test data.
        y_test (Series): Test labels.
        model (Sequential): The trained model.
    """
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Train Accuracy')
    ax.plot(history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

def main():
    """Main function to run the Streamlit app for neural network classifier."""
    st.title("Neural Network Classifier for Safe Driving Data")
    
    if "data" not in st.session_state:
        st.error("No data available. Please load the data in another page first.")
        
        if st.sidebar.button("Load Data"):
            redirect_to_page("pages/1_Load_data.py")
        return
    
    df = st.session_state.data

    st.sidebar.header("Manual Model Parameters")
    units_1 = st.sidebar.number_input("Units in First Layer", min_value=32, max_value=64, step=8, value=32)
    units_2 = st.sidebar.number_input("Units in Second Layer", min_value=8, max_value=32, step=8, value=8)
    learning_rate = st.sidebar.selectbox("Learning Rate", [1e-1, 1e-2, 1e-3, 1e-4], index=2)
    
    current_path = Path(__file__).resolve().parent
    model_path = f"{current_path.parent}/weights/neural_network.h5"
    history_path = f"{current_path.parent}/weights/neural_network_hist.json"
    model_exists = Path(model_path).exists()

    if model_exists:
        action = st.radio("Model weights already exist. What would you like to do?", 
                          ("Train and Evaluate Model Again", "Load Model and View Plots"))

        if action == "Load Model and View Plots":
            model = load_model_weights("neural_network")
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
            history = load_training_history(history_path)
            plot_results(history, X_test, y_test, model)
            return

    if st.button("Train and Evaluate Model"):
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
        model, hist, test_loss, test_acc = train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, units_1, units_2, learning_rate)

        save_model_weights(model, "neural_network", X_train)
        save_training_history(hist, history_path)
        
        st.write(f"Test Loss: {test_loss}")
        st.write(f"Test Accuracy: {test_acc}")
        
        plot_results(hist.history, X_test, y_test, model)

    if st.button('Train and Evaluate with Model Tuning'):
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
        model, hist, test_loss, test_acc = tune_model(X_train, y_train, X_val, y_val, X_test, y_test)

        save_model_weights(model, "neural_network_tuned", X_train)
        save_training_history(hist, f"{current_path.parent}/weights/neural_network_tuned_hist.json")
        
        st.write(f"Test Loss: {test_loss}")
        st.write(f"Test Accuracy: {test_acc}")
        
        plot_results(hist.history, X_test, y_test, model)

if __name__ == "__main__":
    main()
