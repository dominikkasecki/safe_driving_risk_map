import streamlit as st
from src.components.modelling.modelling_part import run_models, import_df, load_data
import pandas as pd
from utils.redirect import redirect_to_page

def main():
    """Main function to run the Streamlit app for binary classification visualization."""
    st.title("Binary Classification Visualiser")
    st.sidebar.title("Model Selection ")

    if "data" not in st.session_state:
        st.sidebar.subheader("Upload Data (Optional) or load the data")
        st.error("In order to use models you need to first upload or load the data")
        fil = st.sidebar.file_uploader("Select/Upload File")

        if fil is not None:
            driving_df = load_data(fil)
            st.markdown("Here we are choosing driving dataframe to predict which streets are high/low risk")
            st.session_state.data = driving_df

            if st.sidebar.checkbox("Show raw data", value=False):
                st.subheader("Here is the preprocessed raw dataset:")
                st.write(driving_df)
        else:
            if st.sidebar.button("Load Data"):
                redirect_to_page("pages/1_Load_data.py")
    else:
        # Select model
        model_choice = st.sidebar.selectbox(
            "Select classifier algorithm",
            ("select classifier", "XGBoost Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)")
        )

        if model_choice != "select classifier":
            model_params = {}
            if model_choice == "XGBoost Classifier":
                model_params["n_estimators"] = st.sidebar.slider("Number of boosting rounds (n_estimators)", 100, 500, 100)
                model_params["max_depth"] = st.sidebar.number_input("Maximum tree depth for base learners (max_depth)", 1, 10, 1)
                model_params["learning_rate"] = st.sidebar.slider("Learning rate (eta)", 0.01, 0.3, 0.01)
                model_params["subsample"] = st.sidebar.slider("Subsample ratio of the training instances (subsample)", 0.5, 1.0, 0.5)
                model_params["colsample_bytree"] = st.sidebar.slider("Subsample ratio of columns when constructing each tree (colsample_bytree)", 0.5, 1.0, 0.5)
                selected_model = "XGBoost"
            elif model_choice == "Random Forest Classifier":
                model_params["n_estimators"] = st.sidebar.slider("Number of trees in the forest", 100, 500, 100)
                model_params["max_depth"] = st.sidebar.number_input("Maximum depth of the tree", 1, 10, 1)
                model_params["min_samples_split"] = st.sidebar.number_input("Minimum number of samples required to split an internal node", 2, 10, 2)
                model_params["min_samples_leaf"] = st.sidebar.number_input("Minimum number of samples required to be at a leaf node", 1, 10, 1)
                model_params["bootstrap"] = st.sidebar.checkbox("Bootstrap samples when building trees", True)
                selected_model = "Random Forest"
            elif model_choice == "Support Vector Machine (SVM)":
                model_params["C"] = st.sidebar.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
                model_params["kernel"] = st.sidebar.selectbox("Kernel type", ("linear", "poly", "rbf", "sigmoid"))
                model_params["gamma"] = st.sidebar.selectbox("Kernel coefficient (gamma)", ("scale", "auto"))
                selected_model = "SVM"

            evaluation_metric = st.sidebar.selectbox(
                "Which Evaluation Metrics do you want?",
                ("Confusion matrix", "ROC Curve", "Precision recall curve")
            )

            if st.sidebar.button("Apply Classifier"):
                safe_driving_with_accidents_df = import_df("safe_driving_with_accidents", ["event_start", "event_end"]).iloc[:500, :]
                st.write(f"Training {model_choice} model...")
                run_models(safe_driving_with_accidents_df, "y_var", selected_model, model_params, evaluation_metric)

if __name__ == "__main__":
    main()
