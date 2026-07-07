from pathlib import Path

import joblib
import argparse
from app.core import logger
from training.evaluation import ModelEvaluator
from training.models.factory import ModelFactory
from app.utils.mlflow_logger import MLflowLogger


DATA_DIR = Path("data/processed")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    parser.add_argument(
        "--model",
        choices=[
            "logistic_regression",
            "random_forest",
            "xgboost",
        ],
        default="logistic_regression",
        help="Model to train",
    )

    return parser.parse_args()


def main():
    logger.info("Loading processed dataset...")

    X_train, y_train = joblib.load(DATA_DIR / "train.joblib")
    X_test, y_test = joblib.load(DATA_DIR / "test.joblib")

    args = parse_args()

    model = ModelFactory.create(args.model)

    logger.info(f"Training {model.__class__.__name__}...")

    model.train(X_train, y_train)

    model.save(f"models/{model.__class__.__name__}.joblib")

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

    params = model.model.get_params()
    params["model"] = model.__class__.__name__

    mlflow_logger.log_run(
        model=model,
        report=report,
        params=params,
        artifact_path=model.__class__.__name__,
    )


if __name__ == "__main__":
    main()
