from typing import List, Tuple
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import streamlit as st
from pathlib import Path
import joblib

def load_data(file: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file)
    return df

def import_df(name: str = "", columns_to_convert: List[str] = [], custom_path: str = False) -> pd.DataFrame:
    """
    Import a DataFrame from a CSV file and convert specified columns to datetime.

    Args:
        name (str, optional): Name of the CSV file. Defaults to "".
        columns_to_convert (List[str], optional): List of columns to convert to datetime. Defaults to [].
        custom_path (str, optional): Custom path to the CSV file. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the imported data.
    """
    
    path = f"{Path(__file__).resolve().parent.parent.parent.parent}/data/cleaned_data/{name}.csv"
    if custom_path:
        path = custom_path
    df = pd.read_csv(path)
    for col in columns_to_convert:
        df[col] = pd.to_datetime(df[col])
    return df.iloc[:5000, :]

def preprocess_data(df: pd.DataFrame, target_column: str, drop_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the data by separating features and target, and encoding the target variable.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column.
        drop_columns (List[str]): List of columns to drop.

    Returns:
        Tuple[pd.DataFrame, pd.Series, LabelEncoder]: Tuple containing the features DataFrame, encoded target Series, and the LabelEncoder.
    """
    X = df.drop(columns=drop_columns)
    y = df[target_column]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create a preprocessor for numerical and categorical features.

    Args:
        X (pd.DataFrame): DataFrame containing the features.

    Returns:
        ColumnTransformer: Preprocessor for the features.
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numerical_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
        ]
    )
    return preprocessor

def format_metrics(accuracy: float, precision: float, recall: float, f1: float) -> None:
    """
    Format and display evaluation metrics.

    Args:
        accuracy (float): Accuracy of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
        f1 (float): F1 score of the model.
    """
    st.write(f"Accuracy: {accuracy:.4f}")
    
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

def plot_evaluation_metrics(y_test: pd.Series, y_pred: pd.Series, evaluation_metric: str) -> None:
    """
    Plot evaluation metrics such as confusion matrix, ROC curve, and precision-recall curve.

    Args:
        y_test (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
        evaluation_metric (str): Evaluation metric to plot.
    """

    fig, ax = plt.subplots()

  
    
    if evaluation_metric == "Confusion matrix":
        cm = confusion_matrix(y_test, y_pred)

       
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"  ,ax = ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
     
    elif evaluation_metric == "ROC Curve":
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
       
    elif evaluation_metric == "Precision recall curve":
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")

    st.pyplot(fig)

def train_and_evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, label_encoder: LabelEncoder, model_name: str, evaluation_metric: str):
    """
    Train and evaluate the model.

    Args:
        model (Pipeline): Model pipeline to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        label_encoder (LabelEncoder): Label encoder for the target variable.
        model_name (str): Name of the model.
        evaluation_metric (str): Evaluation metric to use.

    Returns:
        Pipeline: Trained model pipeline.
    """
    # Fit the model with a progress spinner
    with st.spinner("Fitting the model..."):
        model.fit(X_train, y_train)
    st.success(f"{model_name} fitted successfully")

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    st.write(f"{model_name} Cross-Validation Accuracy Scores: {cv_scores}")
    st.write(f"{model_name} Mean Cross-Validation Accuracy: {cv_scores.mean()}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and generate classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Extract precision, recall, and f1 score
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    # Display metrics
    format_metrics(accuracy, precision, recall, f1)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(y_test, y_pred, evaluation_metric)

    return model


def save_model_weights(model: Pipeline, model_name: str, original_x_data: pd.DataFrame) -> None:
    """
    Save the model weights and features.

    Args:
        model (Pipeline): Trained model pipeline.
        model_name (str): Name of the model.
        original_x_data (pd.DataFrame): Original features DataFrame.
    """
    root_path = Path(__file__).resolve().parent.parent.parent.parent
    joblib_file = f"{root_path}/weights/{model_name}.pkl"

    with open(f"{root_path}/model_features/{model_name}_features.txt", "w") as f:
        f.write(str(original_x_data.columns.tolist()))

    joblib.dump(model, joblib_file)

def run_models(df: pd.DataFrame, target_column: str, model_choice: str, model_params: dict, evaluation_metric: str) -> None:
    """
    Run the specified model with the given parameters and evaluation metric.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column.
        model_choice (str): Choice of model to run.
        model_params (dict): Parameters for the model.
        evaluation_metric (str): Evaluation metric to use.

    Returns:
        None
    """
    drop_columns = ["y_var", "event_start", "event_end", "eventid", "incident_severity", "weighted_avg", "material_damage_only_sum", "road_segment_id", "injury_or_fatal_sum", "road_name", "longitude", "latitude"]
    X, y_encoded, label_encoder = preprocess_data(df, target_column, drop_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    original_x_data = df.drop(columns=drop_columns)

    if model_choice == "XGBoost":
        model_params["use_label_encoder"] = False
        model_params["eval_metric"] = "logloss"
        model = XGBClassifier(**model_params)
        model_name = "XGBoost"
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(**model_params)
        model_name = "random_forest"
    elif model_choice == "SVM":
        model = SVC(**model_params)
        model_name = "svm"
    else:
        st.error("Invalid model choice")
        return

    preprocessor = create_preprocessor(X_train)
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    
    trained_model = train_and_evaluate_model(model_pipeline, X_train, y_train, X_test, y_test, label_encoder, model_name, evaluation_metric)
    save_model_weights(trained_model, model_name.lower(), original_x_data)
