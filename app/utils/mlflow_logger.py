import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from training.evaluation import EvaluationReport


class MLflowLogger:
    """Utility class for MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "creditflow-ai",
    ):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("creditflow-ai")

        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        model,
        report: EvaluationReport,
        params: dict,
        artifact_path: str,
    ):
        with mlflow.start_run():
            mlflow.log_params(params)

            mlflow.log_metric(
                "accuracy",
                report.accuracy,
            )

            mlflow.log_metric(
                "precision",
                report.precision,
            )

            mlflow.log_metric(
                "recall",
                report.recall,
            )

            mlflow.log_metric(
                "f1",
                report.f1,
            )

            mlflow.log_metric(
                "roc_auc",
                report.roc_auc,
            )

            if isinstance(model.model, XGBClassifier):
                mlflow.xgboost.log_model(
                    xgb_model=model.model,
                    name=model.__class__.__name__,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model.model,
                    name=model.__class__.__name__,
                )
