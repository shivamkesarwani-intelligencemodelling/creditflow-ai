# scripts/test_config.py

from app.core.config import get_settings

settings = get_settings()

print(settings.app_name)
print(settings.raw_data_dir)
print(settings.log_level)
