from dataclasses import dataclass

# from sklearn.metrics import (
#    accuracy_score,
#    classification_report,
#    confusion_matrix,
#    f1_score,
#    precision_score,
#    recall_score,
#    roc_auc_score,
# )


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list
    classification_report: dict
