from pathlib import Path
import streamlit as st
from src.components.data_cleaning.export_data import export_all_data
from src.components.data_cleaning.final_cleaning import clean_all_data
from utils.import_df import import_df
from utils.redirect import redirect_to_page

def load_original_data():
    """Load original data from an SQL Database and export it.

    Displays a spinner and messages indicating the process status.
    """
    with st.spinner('Loading original data from SQL Database'):
        st.write("Original data does not exist")
        st.write("Exporting data")
        export_all_data()
    st.success('Original data retrieved successfully')

def load_clean_data():
    """Clean the original data.

    Displays a spinner and messages indicating the process status.
    """
    with st.spinner("Cleaning original data"):
        st.write("Cleaned data does not exist")
        st.write("Cleaning data")
        clean_all_data()
    st.success('Cleaning data successfully')

def import_data_to_session_state():
    """Import cleaned data to Streamlit's session state."""
    st.write("      ")
    st.session_state.data = import_df(
        "safe_driving_with_accidents",
        columns_to_convert=["event_start", "event_end"],
    )

def check_files(original_d_exists, cleaned_d_exists):
    """Check if original and cleaned data files exist and load them.

    Args:
        original_d_exists (bool): Flag indicating if original data exists.
        cleaned_d_exists (bool): Flag indicating if cleaned data exists.
    """
    if not original_d_exists and not cleaned_d_exists:
        load_original_data()
        load_clean_data()
    elif not original_d_exists:
        load_original_data()
    elif not cleaned_d_exists:
        load_clean_data()
    
    import_data_to_session_state()

def check_if_saved_models_exist():
    """Check if saved models exist in the weights directory.

    Returns:
        bool: True if models exist, False otherwise.
    """
    if not any(Path("./weights").iterdir()):
        st.write("Models do not exist")
        st.error("Train models by going to modelling page")
        return False
    elif any(Path("./weights").iterdir()) and "data" not in st.session_state:
        st.success("Models exist")
        return True
    else:
        return True

def main():
    """Main function to run the Streamlit app."""
    st.header("Let's check if data exists")
    data_dir_exists = Path('data').exists()
    if not data_dir_exists:
        Path('data/original_data').mkdir(parents=True, exist_ok=True)
        Path('data/cleaned_data').mkdir(parents=True, exist_ok=True)

    original_data_exists = any(Path("data/original_data").iterdir())
    cleaned_data_exists = any(Path("data/cleaned_data").iterdir())

    st.write("Checking if files exist in the directory")

    if "data" not in st.session_state:
        with st.spinner(text="checking"):
            check_files(original_data_exists, cleaned_data_exists)
            st.success("Done!")

    if "data" in st.session_state:
        st.write("Data was loaded properly and exists")
        st.write(st.session_state.data.head())

        if st.button("Proceed to modelling"):
            redirect_to_page("pages/3_Modelling.py")

        if check_if_saved_models_exist():
            if st.button("Show map"):
                redirect_to_page("App.py")

if __name__ == "__main__":
    main()
