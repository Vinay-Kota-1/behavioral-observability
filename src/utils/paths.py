"""
Shared paths and configuration utilities.
"""

import os
from pathlib import Path

# Project root is two levels up from src/utils/
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Key directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Config files
FEATURE_CONFIG = CONFIG_DIR / "feature_config.yaml"
ADVERSARIAL_CONFIG = CONFIG_DIR / "adversarial_config.yaml"

# Model files
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_xgb.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "anomaly_xgb_metadata.json"

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
)

def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT

def get_config_path(name: str) -> Path:
    """Get path to a config file."""
    return CONFIG_DIR / name

def get_model_path(name: str) -> Path:
    """Get path to a model file."""
    return MODELS_DIR / name
