import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import errors
from pathlib import Path

def connect_to_database():
    """Establish a connection to the PostgreSQL database.

    Returns:
        connection: A connection object to the PostgreSQL database if successful, None otherwise.
    """
    db_params = {
        "host": "194.171.191.226",
        "port": "6379",
        "database": "postgres",
        "user": "group6",
        "password": "blockd_2024group6_79",
    }
    try:
        conn_psycopg2 = psycopg2.connect(**db_params)
        print("Connection was successful!")
        cursor = conn_psycopg2.cursor()  # Explicit cursor call for debug
        cursor.close()  # Close the cursor to clean up
        return conn_psycopg2
    except Exception as e:
        print("Connection was not successful!")
        print(e)
        return None

def create_cursor(connection):
    """Create a cursor from the database connection.

    Args:
        connection: The database connection object.

    Returns:
        cursor: The cursor object.
    """
    return connection.cursor()

def close_cursor(cursor):
    """Close the cursor.

    Args:
        cursor: The cursor object to close.
    """
    cursor.close()

def close_connection(connection):
    """Close the database connection.

    Args:
        connection: The connection object to close.
    """
    connection.close()

def init_database_connection(func):
    """Decorator to initialize a database connection and cursor for a function.

    Args:
        func: The function to wrap.

    Returns:
        wrapper: The wrapped function with database connection and cursor.
    """
    def wrapper(*args, **kwargs):
        connection = connect_to_database()
        if connection is None:
            return None

        cursor = create_cursor(connection)
        res = func(cursor, *args, **kwargs)

        if kwargs.get("commit", False):
            connection.commit()

        close_cursor(cursor)
        close_connection(connection)
        return res

    return wrapper

@init_database_connection
def make_query(cursor, query, show_results=False, commit=False):
    """Execute a SQL query on the database.

    Args:
        cursor: The cursor object to execute the query.
        query (str): The SQL query to execute.
        show_results (bool, optional): Flag to indicate if results should be fetched. Defaults to False.
        commit (bool, optional): Flag to indicate if the transaction should be committed. Defaults to False.

    Returns:
        list or str: The query results if show_results is True, otherwise "Query succeeded".
    """
    try:
        cursor.execute(query)
    except errors.DuplicateTable as e:
        print(e)
        print("The table already exists but since this is a View creation it is allowed")
    except Exception as e:
        print(e)
        return None

    if show_results:
        rows = cursor.fetchall()
        return rows

    return "Query succeeded"

def get_column_names(table_name):
    """Get the column names of a table.

    Args:
        table_name (str): The name of the table.

    Returns:
        np.array: Array of column names.
    """
    q = f"""
    SELECT COLUMN_NAME
    FROM information_schema.columns
    WHERE table_schema ='group6_warehouse'
    AND table_name ='{table_name}'
    ORDER BY ordinal_position
    """
    return np.array(make_query(q, show_results=True)).flatten()

def drop_views():
    """Drop predefined views from the database."""
    views = [
        "safe_driving",
        "wind",
        "precipitation",
        "temperature",
        "accident_data_17_23"
    ]
    for view in views:
        make_query(f"DROP VIEW IF EXISTS group6_warehouse.{view}", show_results=False, commit=True)

def create_views():
    """Create predefined views in the database."""
    views_queries = {
        "safe_driving": """
            CREATE VIEW group6_warehouse.safe_driving AS
            SELECT * FROM data_lake.safe_driving
        """,
        "accident_data_17_23": """
            CREATE VIEW group6_warehouse.accident_data_17_23 AS
            SELECT * FROM data_lake.accident_data_17_23
        """,
        "precipitation": """
            CREATE VIEW group6_warehouse.precipitation AS
            SELECT DTG, RI_PWS_10 FROM data_lake.precipitation
        """,
        "temperature": """
            CREATE VIEW group6_warehouse.temperature AS
            SELECT DTG, T_DRYB_10 FROM data_lake.temperature
        """,
        "wind": """
            CREATE VIEW group6_warehouse.wind AS
            SELECT DTG, FF_SENSOR_10 FROM data_lake.wind
        """
    }
    for query in views_queries.values():
        make_query(query, show_results=False, commit=True)

def load_sql_to_df(table_name):
    """Load data from a SQL table into a DataFrame and save it as a CSV file.

    Args:
        table_name (str): The name of the table to load.

    Returns:
        DataFrame: The loaded DataFrame.
    """
    col_names = get_column_names(table_name)

    fetch_query = f"SELECT * FROM group6_warehouse.{table_name};"
    result = make_query(fetch_query, show_results=True)

    if result is None:
        return None

    df = pd.DataFrame(columns=list(col_names), data=result)

    df.to_csv(
        f"{Path(__file__).resolve().parent.parent.parent.parent}/data/original_data/{table_name}.csv",
        index=False,
    )

    return df

def export_all_data():
    """Drop and create views, then load data from the database into CSV files."""
    drop_views()
    create_views()
    
    tables = ["safe_driving", "precipitation", "wind", "temperature", "accident_data_17_23"]
    for table in tables:
        load_sql_to_df(table)

if __name__ == "__main__":
    export_all_data()
