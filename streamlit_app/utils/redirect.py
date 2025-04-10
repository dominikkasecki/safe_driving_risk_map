import streamlit as st

def redirect_to_page(page_name: str):
    """Redirect to a specified page in a Streamlit application.

    Args:
        page_name (str): The name of the page to redirect to.
    """
    st.switch_page(page_name)
