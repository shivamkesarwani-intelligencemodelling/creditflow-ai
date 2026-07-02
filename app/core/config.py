from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CreditFlow AI"
    environment: str = "development"

    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()
