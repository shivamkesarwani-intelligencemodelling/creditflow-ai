import pandas as pd

from training.preprocessing.validate import DataValidator


def test_empty_dataframe():
    validator = DataValidator()

    df = pd.DataFrame()

    result = validator.validate(df)

    assert result.passed is False


def test_valid_dataframe():
    validator = DataValidator()

    df = pd.DataFrame(
        {
            "TARGET": [0, 1],
            "AGE": [25, 32],
        }
    )

    result = validator.validate(
        df,
        required_columns=["TARGET"],
    )

    assert result.passed
