import pandas as pd


def get_feature_types(df: pd.DataFrame):
    """Return numerical and categorical feature lists."""

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features
