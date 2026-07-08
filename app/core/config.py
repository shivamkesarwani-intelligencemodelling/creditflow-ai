from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.models import (
    LogisticRegressionConfig,
    RandomForestConfig,
    XGBoostConfig,
)


class Settings(BaseSettings):
    app_name: str = "CreditFlow AI"
    environment: str = "development"

    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    log_level: str = "INFO"

    # Model configurations
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
