import numpy as np

from training.evaluation import ModelEvaluator


def test_evaluator():
    y_true = np.array([0, 1, 0, 1])

    y_pred = np.array([0, 1, 0, 0])

    y_prob = np.array([0.1, 0.9, 0.2, 0.4])

    evaluator = ModelEvaluator()

    report = evaluator.evaluate(
        y_true,
        y_pred,
        y_prob,
    )

    assert report.accuracy > 0
    assert report.roc_auc > 0
