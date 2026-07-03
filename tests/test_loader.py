from pathlib import Path

import pandas as pd

from training.data import DataLoader


def test_load_csv(tmp_path: Path):
    sample = tmp_path / "sample.csv"

    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(sample, index=False)

    loader = DataLoader(data_dir=str(tmp_path))

    df = loader.load("sample.csv")

    assert df.shape == (2, 2)
