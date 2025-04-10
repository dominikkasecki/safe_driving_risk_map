Sure, here is a concise and informative README for your project:

---

# Safe Driving Risk Prediction

This project predicts the risk level of streets in Breda, Netherlands, based on driving data. The predictions are visualized on an interactive map to indicate whether a street is high or low risk.

## Project Structure

```
.
├── App.py                  # Main application file
├── LICENSE
├── README.md
├── data                    # Data directory
│   ├── cleaned_data
│   │   └── safe_driving_with_accidents.csv
│   └── original_data
│       ├── accident_data_17_23.csv
│       ├── precipitation.csv
│       ├── safe_driving.csv
│       ├── temperature.csv
│       └── wind.csv
├── docs                    # Documentation files
├── htmlcov                 # Coverage reports
├── model_features          # Feature files for different models
├── my_dir                  # Directory for Keras Tuner trials
├── pages                   # Streamlit app pages
├── src                     # Source code for components
│   ├── components
│   │   ├── data_cleaning
│   │   ├── eda
│   │   └── modelling
├── tests                   # Test files
├── utils                   # Utility scripts
└── weights                 # Model weights
```

## Features

- **Risk Prediction Model**: Predicts whether a street is high or low risk based on historical driving data.
- **Interactive Map**: Visualizes risk levels of different streets in Breda.
- **Data Cleaning and Preprocessing**: Includes scripts for cleaning and preprocessing the data.
- **Model Training**: Scripts to train models using various algorithms (neural network, random forest, SVM, XGBoost).
- **Testing**: Includes unit tests for data loading, cleaning, and model training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/safe-driving-risk-prediction.git
    cd safe-driving-risk-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the environment variables:
    - Create a `.env` file in the root directory with the required environment variables.

5. Run the Streamlit app:
    ```bash
    streamlit run App.py
    ```

## Usage

- **Load Data**: Use the `1_Load_data.py` script to load and explore the data.
- **Perform EDA**: Use the `2_Eda.py` script for exploratory data analysis.
- **Model Training**: Use the `3_Modelling.py` and `4_Neural_network.py` scripts to train the models.
- **Visualize Predictions**: Use the `5_Idea.py` script to visualize the predictions on the interactive map.

## Testing

Run the tests using pytest:
```bash
pytest
```

## Tips for Improvement

- Store sensitive data in a `.env` file locally, and avoid uploading it to GitHub.
- Ensure all CSV files containing sensitive data are not included in the repository.
- Use pytest to test critical parts of the code, such as data loading and model training.

## Future Enhancements

- Improve the Interactive Map with additional features and better visualization.
- Experiment with more advanced machine learning models to enhance prediction accuracy.
- Optimize the app for better performance and user experience.

## License

This project is licensed under the MIT License.

---

