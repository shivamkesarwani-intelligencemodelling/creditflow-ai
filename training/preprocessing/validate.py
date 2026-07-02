import pandas as pd


def validate_dataframe(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isnull().all().any():
        raise ValueError("A column contains only null values")

    return True
