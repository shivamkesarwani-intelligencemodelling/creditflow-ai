from pathlib import Path

import joblib

from app.core import logger
from training.models import LogisticRegressionModel
from training.evaluation import ModelEvaluator
from app.utils.mlflow_logger import MLflowLogger


DATA_DIR = Path("data/processed")


def main():
    logger.info("Loading processed dataset...")

    X_train, y_train = joblib.load(DATA_DIR / "train.joblib")
    X_test, y_test = joblib.load(DATA_DIR / "test.joblib")

    model = LogisticRegressionModel()

    logger.info("Training Logistic Regression...")

    model.train(X_train, y_train)

    model.save("models/logistic.joblib")

    logger.info("Training completed.")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()

    report = evaluator.evaluate(
        y_test,
        predictions,
        probabilities,
    )

    logger.info(
        "ROC-AUC: %.4f",
        report.roc_auc,
    )

    logger.info(
        "F1 Score: %.4f",
        report.f1,
    )

    mlflow_logger = MLflowLogger()

    mlflow_logger.log_run(
        model=model,
        report=report,
        params={
            "model": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42,
        },
        artifact_path="logistic_regression",
    )


if __name__ == "__main__":
    main()
