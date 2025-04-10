import streamlit as st
from src.components.eda.eda import show_eda_analysis

def main():
    """Main function to run the Streamlit app and display EDA analysis."""
    show_eda_analysis()

if __name__ == "__main__":
    main()
