#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def import_df(name, columns_to_convert=[]):
    """Imports a DataFrame from a CSV file.

    Args:
        name (str): The name of the CSV file (without extension) to import.
        columns_to_convert (list, optional): List of columns to convert to datetime. Defaults to [].

    Returns:
        pd.DataFrame: The imported DataFrame.
    """
    df = pd.read_csv(
        f"{Path(__file__).resolve().parent.parent.parent.parent}/data/original_data/{name}.csv",
        on_bad_lines='warn',
        delimiter=','
    )
    for col in columns_to_convert:
        df[col] = pd.to_datetime(df[col])
    return df


def import_weather_df(table_name):
    """Imports a weather DataFrame and processes the 'dtg' column.

    Args:
        table_name (str): The name of the CSV file (without extension) to import.

    Returns:
        pd.DataFrame: The processed weather DataFrame.
    """
    df = import_df(table_name)
    try:
        df['dtg'] = pd.to_datetime(df['dtg'], errors='coerce')
        df = df.set_index("dtg").loc["2018-01-01":, :]
        df = df.sort_index()
        return df
    except ValueError as e:
        print(f"Error converting datetime: {e}")
        return None


def clean_categorical_data(df):
    """Cleans categorical data in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    string_cols = [col for col in df.columns if "object" == str(df[col].dtype)]
    for col in string_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)
        df[col] = df[col].str.lower()
    for col in string_cols:
        df = df.rename(columns={col: str(col).lower().replace(" ", "_")})
    return df


def delete_empty_columns(df):
    """Deletes columns in the DataFrame that are completely empty.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with empty columns removed.
    """
    columns_cols_to_drop = []
    for col in df.columns:
        if df[col].isna().sum() == df.shape[0]:
            columns_cols_to_drop.append(col)
    return df.drop(columns=columns_cols_to_drop)


def print_line_break():
    """Prints a decorative line break."""
    print("=" * 20)
    print(" " + "-" * 18 + " ")
    print("=" * 20)


def show_dataframe_general_info(df):
    """Displays general information about the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to describe.
    """
    print("General info of df")
    print(df.info())
    print("Description of df")
    print(df.describe())
    check_df_missing_values(df)


def check_df_missing_values(df):
    """Checks for missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
    """
    total_missing_values = df.isna().sum().sum()
    print(f"Total number of missing values: ", total_missing_values)
    if total_missing_values > 0:
        print("Number of missing values in particular columns: ")
        print(df.isna().sum())


def show_value_counts(df, col):
    """Displays value counts for a specific column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col (str): The name of the column to analyze.
    """
    print(f"Value counts of {col}")
    sorted_val_df = df[col].value_counts().sort_values(ascending=False)
    if sorted_val_df.shape[0] > 6:
        sorted_val_df = sorted_val_df.iloc[:6]
    print(sorted_val_df)
    print_line_break()
    print(f"Least used values in {col} column: ")
    print(df[col].value_counts().sort_values(ascending=True).iloc[:5])
    unique_vals_in_col = len(pd.unique(df[col]))
    col_dtype = str(df[col].dtype)
    if col_dtype.startswith("int") or col_dtype.startswith("float"):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x=col, data=df, ax=ax)
        plt.show()
    elif unique_vals_in_col < 20 and col_dtype.startswith("object"):
        fig, ax = plt.subplots(figsize=(18, 8))
        missing_vals = df[col].isna().sum()
        if missing_vals > 0:
            ax.axhline(
                y=missing_vals,
                color="r",
                linestyle="--",
                linewidth=2,
                label="Missing values in df",
            )
            ax.legend()
            sns.countplot(x=col, data=df.replace({np.nan: "unknown"}), ax=ax)
        sns.countplot(x=col, data=df, ax=ax)
        plt.show()


def show_dataframe_column_value_counts(df):
    """Displays value counts for each column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    cols = df.columns
    for col in cols:
        print_line_break()
        show_value_counts(df, col)
        missing_vals_in_col = df[col].isna().sum()
        if missing_vals_in_col > 0:
            print(f"Missing values in {col}")
            print(f"{col}: {missing_vals_in_col}")


def show_duplicated_values_in_column(df, col_name):
    """Displays duplicated values in a specific column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the column to analyze.
    """
    print(f"Show duplicated values in column: {col_name}")
    total_duplicated_values = df[col_name].duplicated().sum()
    if total_duplicated_values > 0:
        print(f"Duplicated values in {col_name} :")
        print("Number of duplicated values / all rows")
        duplicated_values_perc = round(
            (total_duplicated_values / df[col_name].shape[0] * 100), 2
        )
        print(
            f"{total_duplicated_values}/{df[col_name].shape[0]} :  which is around {duplicated_values_perc}%"
        )
        sorted_val_df = df[col_name].value_counts().sort_values(ascending=False)
        sorted_val_df = sorted_val_df[sorted_val_df > 1]
        print(sorted_val_df)
        duplicated_values = sorted_val_df.reset_index()[col_name]
        print("Show duplicated column rows :")
        print(df[df[col_name].isin(duplicated_values.to_list())])
    else:
        print("No duplicated values in this column !!!")


def show_general_duplicate_values(df, col_name=None):
    """Displays duplicated values in the DataFrame or a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        col_name (str, optional): The name of the column to analyze. Defaults to None.
    """
    if col_name is not None:
        show_duplicated_values_in_column(df, col_name)
    else:
        total_duplicated_values = df.duplicated().sum()
        if total_duplicated_values > 0:
            print(f"Duplicated values in df:")
            print("Number of duplicated values / all rows")
            duplicated_values_perc = round(
                (total_duplicated_values / df.shape[0] * 100), 2
            )
            print(
                f"{total_duplicated_values}/{df.shape[0]} :  which is around {duplicated_values_perc}%"
            )
        else:
            print("No duplicated values in this dataframe !!!")


def drop_duplicates_in_df(df, columns):
    """Drops duplicated values in specified columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to drop duplicates from.

    Returns:
        bool: True if duplicates were dropped, False otherwise.
    """
    drop_duplicated = False
    if len(columns) > 1:
        for col in columns:
            print(f"Duplicated values in {columns} after dropping them")
            print_line_break()
            drop_duplicated = df.drop_duplicates(subset=[col], inplace=True)
            show_duplicated_values_in_column(df, col)
            print_line_break()
    else:
        print(f"Duplicated values in {columns} after dropping them")
        print_line_break()
        drop_duplicated = df.drop_duplicates(subset=[*columns], inplace=True)
        show_duplicated_values_in_column(df, columns[0])
    return drop_duplicated


def plot_columns(df, columns, plot):
    """Plots specified columns using a given plot function.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): List of columns to plot.
        plot (function): The plot function to use (e.g., sns.boxplot).
    """
    if len(columns) == 0:
        print("No columns to plot")
        print_line_break()
        print_line_break()
        return
    cols_length = len(columns)
    fig, axes = plt.subplots(
        nrows=cols_length,
        ncols=1,
        figsize=(12, cols_length * 6),
        sharex=False,
        sharey=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for idx, current_ax in enumerate(axes.flatten()):
        if idx < len(columns):
            current_col = columns[idx]
            current_ax.set_title(f"Column: {current_col}")
            plot(x=df[current_col], ax=current_ax)
    plt.show()


def plot_numeric_columns(df, columns):
    """Plots numeric columns using a boxplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): List of numeric columns to plot.
    """
    plot_columns(df, columns, sns.boxplot)


def plot_string_columns(df, columns):
    """Plots string columns using a countplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): List of string columns to plot.
    """
    plot_columns(df, columns, sns.countplot)


def plot_bool_columns(df, columns):
    """Plots boolean columns using a countplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): List of boolean columns to plot.
    """
    plot_columns(df, columns, sns.countplot)


def plot_value_distributions_in_df(df, columns_to_avoid=[]):
    """Plots value distributions for numeric, string, and boolean columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        columns_to_avoid (list, optional): List of columns to avoid plotting. Defaults to [].
    """
    numeric_cols = [
        col
        for col in df.columns
        if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype))
        and col not in columns_to_avoid
    ]
    string_cols = [
        col
        for col in df.columns
        if "object" == str(df[col].dtype) and col not in columns_to_avoid
    ]
    bool_cols = [
        col
        for col in df.columns
        if "bool" == str(df[col].dtype) and col not in columns_to_avoid
    ]
    if numeric_cols:
        print("Numerical columns plotted :")
        plot_numeric_columns(df, numeric_cols)
        print_line_break()
        print_line_break()
        print_line_break()
    if string_cols:
        print("String columns plotted :")
        plot_string_columns(df, string_cols)
        print_line_break()
        print_line_break()
        print_line_break()
    if bool_cols:
        print("Bool columns plotted :")
        plot_bool_columns(df, bool_cols)


def show_outliers_fraction(df, col, Q1, Q3, IQR):
    """Shows the fraction of outliers in a specific column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col (str): The name of the column to analyze.
        Q1 (float): The first quartile value.
        Q3 (float): The third quartile value.
        IQR (float): The interquartile range.
    """
    print_line_break()
    print(f"The fraction of outliers in {col}")
    total_outliers_number_in_col_mask = (df[col] < Q1 - 1.5 * IQR) | (
        df[col] > Q3 + 1.5 * IQR
    )
    total_outliers_number_in_col = df[total_outliers_number_in_col_mask].shape[0]
    if total_outliers_number_in_col <= 0:
        print(f"No outliers detected in {col} column")
        return
    print(total_outliers_number_in_col)
    total_outliers_number_in_col_perc = (
        round((total_outliers_number_in_col / df.shape[0]), 2) * 100
    )
    print(
        f"{total_outliers_number_in_col}  / {df.shape[0]} which is around {total_outliers_number_in_col_perc}%"
    )
    print_line_break()


def delete_outliers(df, columns, multiplier=1.5):
    """Deletes outliers in specified columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to delete outliers from.
        multiplier (float, optional): Multiplier for the interquartile range. Defaults to 1.5.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    df_no_outliers = df.copy()
    for col in columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (IQR * multiplier)
        upper_bound = Q3 + (IQR * multiplier)
        show_outliers_fraction(df, col, Q1, Q3, IQR)
        print(
            f"{col}: Q1={Q1}, Q3={Q3}, IQR={IQR}, Lower Bound={lower_bound}, Upper Bound={upper_bound}"
        )
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound)
            & (df_no_outliers[col] <= upper_bound)
        ]
    return df_no_outliers


def drop_columns_in_df(df, columns_to_drop):
    """Drops specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns_to_drop (list): List of columns to drop.
    """
    cols_drop_len = len(columns_to_drop)
    for col_to_drop in columns_to_drop:
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)


def convert_column_to_binary(df, columns_with_new_values):
    """Converts specified columns to binary values based on provided criteria.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns_with_new_values (dict): Dictionary with columns as keys and values as conversion criteria.
    """
    for key, val in columns_with_new_values.items():
        col = key
        multiple_values = val["top_values"]
        new_replace_value = val["new_value"]
        most_frequent_values = df[col].value_counts().index[0:multiple_values]
        df[col] = df[col].apply(
            lambda row: (
                row if str(row) in most_frequent_values else new_replace_value
            )
        )


def clip_numerical_cols(df, columns):
    """Rounds numerical columns to two decimal places.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to round.
    """
    for col in columns:
        df[col] = df[col].round(2)


def clean_numerical_cols(df):
    """Cleans numerical columns by taking absolute values and converting to float.

    Args:
        df (pd.DataFrame): The DataFrame to process.
    """
    numeric_cols = [
        col
        for col in df.columns
        if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype))
    ]
    for col in numeric_cols:
        df[col] = df[col].abs()
        df[col] = df[col].astype(float)


def scale_numerical_data(df, columns):
    """Scales numerical data using StandardScaler.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to scale.
    """
    for col in columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])


def calculate_rolling_average(weather_df, value_column):
    """Calculates rolling average for a specific column in the weather DataFrame.

    Args:
        weather_df (pd.DataFrame): The weather DataFrame.
        value_column (str): The name of the column to calculate rolling average for.

    Returns:
        pd.DataFrame: The DataFrame with rolling average added.
    """
    weather_df["rolling_mean"] = (
        weather_df[value_column].rolling("1h", closed="both").mean()
    )
    return weather_df.reset_index()


def merge_driving_with_weather_df(
    driving_df_original, weather_df_original, value_column, new_value_column_name
):
    """Merges driving and weather DataFrames on a rolling average of a specified column.

    Args:
        driving_df_original (pd.DataFrame): The original driving DataFrame.
        weather_df_original (pd.DataFrame): The original weather DataFrame.
        value_column (str): The column to calculate rolling average for.
        new_value_column_name (str): The name of the new column in the merged DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    driving_df = driving_df_original.copy()
    weather_df = weather_df_original.copy()
    weather_df = calculate_rolling_average(weather_df, value_column)
    driving_df["event_start"] = pd.to_datetime(driving_df["event_start"])
    weather_df = weather_df.sort_values("dtg")
    driving_df = driving_df.sort_values("event_start")
    merged_df = pd.merge_asof(
        driving_df,
        weather_df[["dtg", "rolling_mean"]],
        left_on="event_start",
        right_on="dtg",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    merged_df = merged_df.rename(columns={"rolling_mean": new_value_column_name})
    merged_df = merged_df.drop(columns=["dtg"])
    return merged_df


def convert_unknown_to_nan(df):
    """Converts 'unknown' values to NaN in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
    """
    df.replace({"unknown": np.nan}, inplace=True)


def convert_empty_values_to_nan(df, columns):
    """Converts empty values to NaN in specified columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to convert.
    """
    for col in columns:
        df[col] = df[col].replace({"": np.nan})


def drop_rows_with_drop_values(df, col, drop_values):
    """Drops rows with specified values in a given column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to check for drop values.
        drop_values (list): List of values to drop.
    """
    if drop_values:
        mask = df[col].apply(lambda row: str(row) in drop_values)
        idxs_to_drop = df[mask].index
        df.drop(index=idxs_to_drop, inplace=True)


def convert_string_column_to_numerical(df, col, drop_values=[]):
    """Converts a string column to numerical values, optionally dropping specified values.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to convert.
        drop_values (list, optional): List of values to drop. Defaults to [].
    """
    drop_rows_with_drop_values(df, col, drop_values)

    def return_speed(row):
        splitted_row = str(row).split(" ")
        return float(splitted_row[0])
    
    df[col] = df[col].apply(
        lambda row: return_speed(row) if not pd.isnull(row) else row
    )


def show_columns_with_missing_values(df):
    """Displays columns with missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    df_cols = df.columns
    for col in df_cols:
        missing_vals_in_col = df[col].isna().sum()
        if missing_vals_in_col > 0:
            nan_perc = round((missing_vals_in_col / df.shape[0]) * 100, 2)
            print(f"Col: {col} has {missing_vals_in_col} missing values")
            print(
                f"Percentage of missing values / all values in column: {nan_perc } %"
            )
            show_dataframe_column_value_counts(df[[col]])
            print_line_break()


def transform_acc_sev_col_to_encoding(df):
    """Transforms 'accident_severity' column to one-hot encoding.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df = df.copy()
    df = df.join(pd.get_dummies(df["accident_severity"], dtype=float))
    return df


def w_avg(row, weights):
    """Calculates weighted average of accident severity.

    Args:
        row (pd.Series): The row of the DataFrame to process.
        weights (list): List of weights for the calculation.

    Returns:
        float: The weighted average.
    """
    w1, w2 = weights
    values_with_w_sum = (
        row["injury_or_fatal_sum"] * w1 + row["material_damage_only_sum"] * w2
    )
    return values_with_w_sum / (w1 + w2)


def calc_weighted_mean_of_acc_severity(df):
    """Calculates weighted mean of accident severity for streets.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with weighted mean of accident severity.
    """
    df = df.copy()
    df = transform_acc_sev_col_to_encoding(df)
    new_df = (
        df.groupby(["street"])
        .agg(
            injury_or_fatal_sum=("injury or fatal", "sum"),
            material_damage_only_sum=("material damage only", "sum"),
        )
        .reset_index()
    )
    new_df["weighted_avg"] = new_df.apply(lambda row: w_avg(row, [2, 1]), axis=1)
    print(new_df["weighted_avg"].describe())
    return new_df


def export_cleaned_data_to_csv(df_list=[], df_names=[]):
    """Exports cleaned DataFrames to CSV files.

    Args:
        df_list (list, optional): List of DataFrames to export. Defaults to [].
        df_names (list, optional): List of names for the CSV files. Defaults to [].
    """
    for idx, df in enumerate(df_list):
        df.to_csv(
            f"{Path(__file__).resolve().parent.parent.parent.parent}/data/cleaned_data/{df_names[idx]}.csv",
            index=False,
        )


def transform_numerical_column_to_str(df, columns):
    """Transforms numerical columns to strings in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to transform.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df.loc[:, columns] = df.loc[:, columns].astype(str)
    return df


def clean_all_data():
    """Cleans and processes all data, performing various operations on the DataFrames."""
    safe_driving_df = import_df(
        "safe_driving", columns_to_convert=["event_start", "event_end"]
    )
    wind_df = import_weather_df("wind")
    temp_df = import_weather_df("temperature")
    prec_df = import_weather_df("precipitation")
    accidents_17_23_df = import_df("accident_data_17_23")

    safe_driving_df = clean_categorical_data(safe_driving_df)
    safe_driving_df = delete_empty_columns(safe_driving_df)
    show_dataframe_general_info(safe_driving_df)
    show_dataframe_column_value_counts(safe_driving_df)
    show_general_duplicate_values(safe_driving_df)
    show_duplicated_values_in_column(safe_driving_df, "eventid")
    show_duplicated_values_in_column(safe_driving_df, "event_start")
    drop_duplicates_in_df(safe_driving_df, ["eventid", "event_start"])
    safe_driving_df = delete_outliers(
        safe_driving_df, ["end_speed_kmh", "speed_kmh", "duration_seconds"]
    )
    plot_value_distributions_in_df(
        safe_driving_df, ["eventid", "road_segment_id", "latitude", "longitude"]
    )
    drop_columns_in_df(
        safe_driving_df,
        [
            "is_valid",
            "road_manager_type",
            "road_number",
            "road_manager_name",
            "municipality_name",
            "place_name",
        ],
    )
    columns_with_new_values_dict = {
        "incident_severity": {
            "new_value": "other incident severities",
            "top_values": 2,
        },
    }
    convert_column_to_binary(safe_driving_df, columns_with_new_values_dict)
    plot_value_distributions_in_df(
        safe_driving_df, ["eventid", "road_segment_id", "latitude", "longitude"]
    )
    clean_numerical_cols(safe_driving_df)
    clip_numerical_cols(safe_driving_df, ["speed_kmh", "end_speed_kmh", "maxwaarde"])
    scale_numerical_data(
        safe_driving_df, ["duration_seconds", "speed_kmh", "end_speed_kmh", "maxwaarde"]
    )
    safe_driving_df = merge_driving_with_weather_df(
        safe_driving_df, wind_df, "ff_sensor_10", "last_hour_wind_avg"
    )
    safe_driving_df = merge_driving_with_weather_df(
        safe_driving_df, temp_df, "t_dryb_10", "last_hour_temp_avg"
    )
    safe_driving_df = merge_driving_with_weather_df(
        safe_driving_df, prec_df, "ri_pws_10", "last_hour_rain_avg"
    )

    transform_numerical_column_to_str(accidents_17_23_df, ["Year"])
    accidents_17_23_df = clean_categorical_data(accidents_17_23_df)
    convert_unknown_to_nan(accidents_17_23_df)
    convert_empty_values_to_nan(accidents_17_23_df, ["first_mode_of_transport"])
    convert_string_column_to_numerical(
        accidents_17_23_df, "speed_limit", drop_values=["footpace  homezone"]
    )
    drop_columns_in_df(accidents_17_23_df, ["municipality"])
    show_columns_with_missing_values(accidents_17_23_df)
    show_general_duplicate_values(accidents_17_23_df)
    plot_value_distributions_in_df(accidents_17_23_df, columns_to_avoid=[])

    columns_with_new_values_dict = {
        "accident_severity": {"new_value": "injury or fatal", "top_values": 1},
        "town": {"new_value": "other city", "top_values": 1},
        "first_mode_of_transport": {"new_value": "other", "top_values": 1},
        "second_mode_of_transport": {"new_value": "other", "top_values": 2},
        "light_condition": {"new_value": "darkness or twilight", "top_values": 1},
        "road_condition": {"new_value": "wetdamp or snowblack ice", "top_values": 1},
        "road_situation": {"new_value": "other road situation", "top_values": 4},
        "weather": {"new_value": "other weather situation", "top_values": 2},
    }
    convert_column_to_binary(accidents_17_23_df, columns_with_new_values_dict)
    accidents_17_23_df = delete_outliers(
        accidents_17_23_df, ["speed_limit", "accidents"]
    )
    plot_value_distributions_in_df(accidents_17_23_df, columns_to_avoid=[])
    clean_numerical_cols(accidents_17_23_df)
    scale_numerical_data(
        accidents_17_23_df, accidents_17_23_df.select_dtypes(include=["float", "int"])
    )
    drop_columns_in_df(accidents_17_23_df, ["accidents"])

    streets_with_accidents_ratio_df = calc_weighted_mean_of_acc_severity(
        accidents_17_23_df
    )

    safe_driving_with_accidents_df = safe_driving_df.copy().merge(
        streets_with_accidents_ratio_df,
        how="left",
        left_on="road_name",
        right_on="street",
    )
    plot_value_distributions_in_df(
        safe_driving_with_accidents_df[["weighted_avg"]], columns_to_avoid=[]
    )
    mean_weighted_avg = safe_driving_with_accidents_df["weighted_avg"].mean()
    safe_driving_with_accidents_df["y_var"] = np.where(
        safe_driving_with_accidents_df["weighted_avg"] < mean_weighted_avg,
        "low-risk",
        "high-risk",
    )
    drop_columns_in_df(safe_driving_with_accidents_df, ["street"])
    export_cleaned_data_to_csv(
        [safe_driving_with_accidents_df], ["safe_driving_with_accidents"]
    )
