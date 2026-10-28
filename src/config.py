"""Configuration settings and single source of truth for the project.

This module centralizes filesystem paths, column schemas, target definitions,
and hyperparameters used across the entire intrusion detection pipeline.
"""

from pathlib import Path
from typing import List

# Reproducibility
RANDOM_STATE: int = 42

# Filesystem layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Raw dataset files (UNSW-NB15 default)
TRAIN_DATA_PATH: Path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_DATA_PATH: Path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
SAMPLE_INPUT_PATH: Path = SAMPLE_DATA_DIR / "sample_input.csv"

# Model and preprocessor artifact paths
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
FEATURE_COLUMNS_PATH: Path = MODELS_DIR / "feature_columns.json"
SELECTED_FEATURES_PATH: Path = MODELS_DIR / "selected_features.json"
BEST_BINARY_MODEL_PATH: Path = MODELS_DIR / "best_binary.joblib"
BEST_MULTICLASS_MODEL_PATH: Path = MODELS_DIR / "best_multiclass.joblib"

# Evaluation report paths
BINARY_COMPARISON_PATH: Path = REPORTS_DIR / "binary_comparison.csv"
MULTICLASS_COMPARISON_PATH: Path = REPORTS_DIR / "multiclass_comparison.csv"
PER_CLASS_REPORT_PATH: Path = REPORTS_DIR / "per_class_report.csv"

# Column specifications
BINARY_TARGET: str = "label"
MULTI_TARGET: str = "attack_cat"
DROP_COLS: List[str] = ["id"]
CATEGORICAL_COLS: List[str] = ["proto", "service", "state"]

# Train/Test validation split ratio if re-splitting
TEST_SIZE: float = 0.2

# Attack categories in UNSW-NB15
ATTACK_CLASSES: List[str] = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]

NON_NORMAL_ATTACK_CLASSES: List[str] = [
    c for c in ATTACK_CLASSES if c != "Normal"
]


def ensure_directories() -> None:
    """Create all required project directories if they do not already exist.

    Inputs: None
    Outputs: None
    Purpose: Ensures output directories exist before file I/O operations.
    """
    directories: List[Path] = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
