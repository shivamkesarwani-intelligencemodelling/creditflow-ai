import pandas as pd

from training.feature_engineering.features import get_feature_types
from training.feature_engineering.pipeline import build_preprocessor


def test_pipeline_builds():
    df = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [1000, 2000],
            "gender": ["M", "F"],
        }
    )

    num, cat = get_feature_types(df)

    pipeline = build_preprocessor(num, cat)

    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 2
