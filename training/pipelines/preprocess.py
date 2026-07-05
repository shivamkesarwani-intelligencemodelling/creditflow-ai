from pathlib import Path

import joblib

from app.core import logger, settings
from training.data import DataLoader
from training.feature_engineering.features import get_feature_types
from training.feature_engineering.pipeline import build_preprocessor
from training.preprocessing.validate import DataValidator
from training.pipelines.split import split_data


TARGET_COLUMN = "TARGET"


def main():
    logger.info("Starting preprocessing pipeline...")

    loader = DataLoader()
    validator = DataValidator()

    df = loader.load("application_train.csv")

    result = validator.validate(
        df,
        required_columns=[TARGET_COLUMN],
    )

    if not result.passed:
        raise ValueError(result.errors)

    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COLUMN,
    )

    numerical, categorical = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    processed_dir = Path(settings.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        processed_dir / "preprocessor.joblib",
    )

    joblib.dump(
        (X_train_processed, y_train),
        processed_dir / "train.joblib",
    )

    joblib.dump(
        (X_test_processed, y_test),
        processed_dir / "test.joblib",
    )

    logger.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
