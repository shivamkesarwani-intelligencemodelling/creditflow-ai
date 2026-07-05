import numpy as np

from training.models import LogisticRegressionModel


def test_logistic_model_trains():
    X = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ]
    )

    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionModel()

    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
