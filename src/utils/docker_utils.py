import os
from pathlib import Path
from src.utils.config import DATA_DIR, MODELS_DIR

def is_running_in_docker() -> bool:
    """Detects whether code is executing inside Docker container."""
    return Path('/.dockerenv').exists() or os.getenv('RUNNING_IN_DOCKER') == 'true'

def get_data_dir() -> Path:
    """Returns absolute path to data directory appropriate for host or container environment."""
    if is_running_in_docker():
        return Path('/app/data')
    return DATA_DIR

def get_models_dir() -> Path:
    """Returns absolute path to models directory appropriate for host or container environment."""
    if is_running_in_docker():
        return Path('/app/models')
    return MODELS_DIR
