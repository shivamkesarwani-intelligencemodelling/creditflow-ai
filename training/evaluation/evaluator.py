from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.evaluation.metrics import EvaluationReport


class ModelEvaluator:
    """Evaluate binary classification models."""

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob,
    ) -> EvaluationReport:
        return EvaluationReport(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            roc_auc=roc_auc_score(y_true, y_prob),
            confusion_matrix=confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),
            classification_report=classification_report(
                y_true,
                y_pred,
                output_dict=True,
            ),
        )
