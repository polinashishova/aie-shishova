"""src/mrh/config.py
Конфигурация переменных окружения.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

def load_env_file() -> None:
    """Загружает .env файл, если он существует."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

load_env_file()

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
APP_HOST = os.environ.get('APP_HOST', '0.0.0.0')
APP_PORT = int(os.environ.get('APP_PORT', 8000))
CB_FEATURE_COLUMN = os.environ.get('CB_FEATURE_COLUMN', 'genres_decade_tags')
CB_THRESHOLD = float(os.environ.get('CB_THRESHOLD', 3.0))
CF_THRESHOLD = float(os.environ.get('CF_THRESHOLD', 4.0))
CONFIGS_DIR = os.environ.get('CONFIGS_DIR', Path.cwd() / 'config')
