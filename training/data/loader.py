from pathlib import Path

import pandas as pd

from app.core import logger, settings
from training.data.exceptions import DataLoadError


SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


class DataLoader:
    """Utility class for loading datasets."""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.raw_data_dir)

    def load(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename

        logger.info(f"Loading dataset: {path}")

        if not path.exists():
            raise DataLoadError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise DataLoadError(f"Unsupported file type: {suffix}")

        try:
            if suffix == ".csv":
                df = pd.read_csv(path)

            elif suffix == ".parquet":
                df = pd.read_parquet(path)

            logger.info(
                "Loaded dataset | rows=%d columns=%d",
                len(df),
                len(df.columns),
            )

            return df

        except Exception as exc:
            raise DataLoadError(str(exc)) from exc
