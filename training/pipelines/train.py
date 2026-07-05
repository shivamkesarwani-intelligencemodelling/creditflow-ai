from pathlib import Path

import joblib

from app.core import logger
from training.models import LogisticRegressionModel


DATA_DIR = Path("data/processed")


def main():
    logger.info("Loading processed dataset...")

    X_train, y_train = joblib.load(DATA_DIR / "train.joblib")

    model = LogisticRegressionModel()

    logger.info("Training Logistic Regression...")

    model.train(X_train, y_train)

    model.save("models/logistic.joblib")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
