from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


class EvaluationPlots:
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def confusion_matrix(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(6, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax,
        )

        path = self.output_dir / "confusion_matrix.png"

        plt.savefig(path, dpi=200)
        plt.close(fig)

        return path

    def roc_curve(self, y_true, probabilities):
        fig, ax = plt.subplots(figsize=(6, 6))

        RocCurveDisplay.from_predictions(
            y_true,
            probabilities,
            ax=ax,
        )

        path = self.output_dir / "roc_curve.png"

        plt.savefig(path, dpi=200)
        plt.close(fig)

        return path

    def precision_recall_curve(self, y_true, probabilities):
        fig, ax = plt.subplots(figsize=(6, 6))

        PrecisionRecallDisplay.from_predictions(
            y_true,
            probabilities,
            ax=ax,
        )

        path = self.output_dir / "precision_recall_curve.png"

        plt.savefig(path, dpi=200)
        plt.close(fig)

        return path
