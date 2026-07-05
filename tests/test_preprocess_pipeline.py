from training.feature_engineering.pipeline import build_preprocessor


def test_preprocessor_creation():
    pipeline = build_preprocessor(
        numerical_features=["age"],
        categorical_features=["gender"],
    )

    assert pipeline is not None
