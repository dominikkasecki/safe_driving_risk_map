#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency

def import_df(name="", columns_to_convert=[], custom_path=None):
    """
    Imports a DataFrame from a CSV file.

    Parameters:
    name (str): Name of the file to be imported.
    columns_to_convert (list): List of columns to be converted to datetime.
    custom_path (str): Custom path for the file.

    Returns:
    pd.DataFrame: Imported DataFrame.
    """
    if custom_path:
        path = custom_path
    else:
        path = f"{Path(__file__).resolve().parent.parent.parent.parent}/data/cleaned_data/{name}.csv"



    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")

    df = pd.read_csv(path)

    if len(columns_to_convert) > 0:
        for col in columns_to_convert:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in the DataFrame.")
            df[col] = pd.to_datetime(df[col])

    return df

def show_dataframe_general_info(df):
    """
    Displays general information about the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to be described.
    """
    st.write("General info of df")
   

    st.write("Description of df")
    st.write(df.describe())

def check_df_missing_values(df):
    """
    Checks for missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to be checked.
    """
    # total_missing_values = df.isna().sum().sum()
    # st.write(f"Total number of missing values: {total_missing_values}")
    # if total_missing_values > 0:
    #     st.write("Number of missing values in particular columns:")
    #     st.write(df.isna().sum())

important_numerical_features = [
    "duration_seconds",
    "latitude",
    "longitude",
    "speed_kmh",
    "end_speed_kmh",
    "maxwaarde",
    "last_hour_wind_avg",
    "last_hour_temp_avg",
    "last_hour_rain_avg",
]

def plot_value_distributions(df):
    """
    Plots distributions of numerical values.

    Parameters:
    df (pd.DataFrame): DataFrame to be plotted.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in important_numerical_features:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Box plot of {col}")
        st.pyplot(plt.gcf())
        plt.close()

def plot_pair_plots(df):
    """
    Plots pair plots for important numerical features.

    Parameters:
    df (pd.DataFrame): DataFrame to be plotted.
    """
    sns.pairplot(df[important_numerical_features])
    st.pyplot(plt.gcf())
    plt.close()

def plot_categorical_value_distributions(df):
    """
    Plots distributions of categorical values.

    Parameters:
    df (pd.DataFrame): DataFrame to be plotted.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        if col == "road_name":
            top_20_roads = df["road_name"].value_counts().nlargest(20).index
            sns.countplot(x=df[df["road_name"].isin(top_20_roads)]["road_name"])
            plt.title("Distribution of Top 20 Road Names")

        elif col == "street":
            top_20_roads = df["street"].value_counts().nlargest(20).index
            sns.countplot(x=df[df["street"].isin(top_20_roads)]["street"])
            plt.title("Distribution of Top 20 Streets")
        else:
            sns.countplot(x=df[col])
            plt.title(f"Distribution of {col}")
        plt.xticks(rotation=90)
        st.pyplot(plt.gcf())
        plt.close()

def plot_box_plots(df):
    """
    Plots box plots for numerical features across categorical features.

    Parameters:
    df (pd.DataFrame): DataFrame to be plotted.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for cat_col in categorical_cols:
        for num_col in important_numerical_features:
            if num_col in df.columns:
                plt.figure(figsize=(12, 6))
                if cat_col == "road_name":
                    top_20_roads = df["road_name"].value_counts().nlargest(20).index
                    sns.boxplot(x=df[df["road_name"].isin(top_20_roads)][cat_col], y=df[num_col])
                    plt.title(f"{num_col} distribution across Top 20 {cat_col}")
                else:
                    sns.boxplot(x=df[cat_col], y=df[num_col])
                    plt.title(f"{num_col} distribution across {cat_col}")
                plt.xticks(rotation=90)
                st.pyplot(plt.gcf())
                plt.close()

def plot_value_distributions_2(df):
    """
    Plots distributions of all numerical values.

    Parameters:
    df (pd.DataFrame): DataFrame to be plotted.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        st.pyplot(plt.gcf())
        plt.close()

def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Parameters:
    x, y (pd.Series): Categorical columns to calculate association for.

    Returns:
    float: Cramér's V statistic.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def calculate_cramers_v_matrix(df, cols):
    """
    Calculate Cramér's V matrix for a given DataFrame and columns.

    Parameters:
    df (pd.DataFrame): DataFrame to be analyzed.
    cols (list): List of categorical columns to calculate Cramér's V for.

    Returns:
    pd.DataFrame: Cramér's V matrix.
    """
    cramers_v_matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0
            else:
                cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    cramers_v_matrix = cramers_v_matrix.astype(float)
    return cramers_v_matrix

def show_eda_analysis():
    """
    Conducts EDA and plots various graphs using Streamlit.
    """
    safe_driving_df = import_df("safe_driving_with_accidents", ["event_start", "event_end"])

    safe_driving_df = safe_driving_df.iloc[:10000, :]

    accidents_17_23_df = import_df(
        custom_path=f"{Path(__file__).resolve().parent.parent.parent.parent}/data/original_data/accident_data_17_23.csv"
    )

    check_df_missing_values(safe_driving_df)

    plot_value_distributions(safe_driving_df)

    plt.figure(figsize=(16, 10))
    correlation_matrix = safe_driving_df[important_numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    st.pyplot(plt.gcf())
    plt.close()

    safe_driving_df["event_start"] = pd.to_datetime(safe_driving_df["event_start"])
    safe_driving_df["event_end"] = pd.to_datetime(safe_driving_df["event_end"])

    plt.figure(figsize=(12, 6))
    safe_driving_df["hour"] = safe_driving_df["event_start"].dt.hour
    sns.countplot(x="hour", data=safe_driving_df)
    plt.title("Incidents by Hour of Day")
    st.pyplot(plt.gcf())
    plt.close()

    show_dataframe_general_info(accidents_17_23_df)

    plot_value_distributions_2(accidents_17_23_df)

    plot_categorical_value_distributions(accidents_17_23_df)

    categorical_columns = accidents_17_23_df.select_dtypes(include=["object"]).columns
    cramers_v_matrix = calculate_cramers_v_matrix(accidents_17_23_df, categorical_columns)

    plt.figure(figsize=(16, 10))
    sns.heatmap(cramers_v_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Cramér's V Correlation Heatmap")
    st.pyplot(plt.gcf())
    plt.close()

if __name__ == '__main__':
    show_eda_analysis()
