import pandas as pd

def import_df(name="", columns_to_convert=[], custom_path=False):
    """Import a DataFrame from a CSV file and convert specified columns to datetime.

    Args:
        name (str, optional): Name of the CSV file (without extension) located in the cleaned_data directory. Defaults to "".
        columns_to_convert (list, optional): List of column names to convert to datetime. Defaults to [].
        custom_path (str, optional): Custom path to the CSV file. If provided, this path is used instead of the default path. Defaults to False.

    Returns:
        DataFrame: The imported DataFrame with specified columns converted to datetime.
    """
    path = f"./data/cleaned_data/{name}.csv"
    if custom_path:
        path = custom_path
    df = pd.read_csv(path)

    for col in columns_to_convert:
        df[col] = pd.to_datetime(df[col])
    return df
