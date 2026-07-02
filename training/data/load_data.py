from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


def load_csv(filename: str) -> pd.DataFrame:
    path = Path(settings.raw_data_dir) / filename

    logger.info(f"Loading {path}")

    df = pd.read_csv(path)

    logger.info(f"Loaded {len(df)} rows")

    return df
