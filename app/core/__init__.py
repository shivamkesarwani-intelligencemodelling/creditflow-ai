from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
logger = setup_logging(settings.log_level)

__all__ = ["settings", "logger"]
