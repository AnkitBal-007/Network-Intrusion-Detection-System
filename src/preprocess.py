"""Data preprocessing module for feature scaling and categorical encoding.

Defines the ColumnTransformer pipeline containing StandardScaler and
OneHotEncoder, fitted strictly on training data to prevent leakage.
Persists fitted preprocessor and input feature schemas to disk.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config

logger = logging.getLogger(__name__)


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for numeric and categorical features.

    Inputs:
        numeric_cols: List of numeric column names to scale with StandardScaler
        categorical_cols: List of categorical column names to encode with OneHotEncoder
    Outputs:
        Unfitted ColumnTransformer instance configured with handle_unknown='ignore'
    Purpose:
        Define consistent preprocessing steps for network traffic records across training and app.
    """
    logger.info(
        "Building preprocessor with %d numeric columns and %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0,
    )
    return preprocessor


def get_feature_column_lists(
    df: pd.DataFrame
) -> Tuple[List[str], List[str], List[str]]:
    """Determine expected feature columns, splitting into numeric and categorical subsets.

    Inputs:
        df: Input pandas DataFrame (e.g. training set or upload template)
    Outputs:
        Tuple of (all_feature_cols, numeric_cols, categorical_cols)
    Purpose:
        Extract input schema excluding dropped columns and targets per config.py.
    """
    exclude_cols = set(config.DROP_COLS + [config.BINARY_TARGET, config.MULTI_TARGET])
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    categorical_cols = [c for c in config.CATEGORICAL_COLS if c in all_feature_cols]
    numeric_cols = [c for c in all_feature_cols if c not in categorical_cols]

    return all_feature_cols, numeric_cols, categorical_cols


def fit_and_save_preprocessor(
    train_df: pd.DataFrame
) -> Tuple[ColumnTransformer, List[str]]:
    """Fit ColumnTransformer on training features only and persist artifacts to models/.

    Inputs:
        train_df: Cleaned training DataFrame containing features and targets
    Outputs:
        Tuple of (fitted_preprocessor, feature_columns)
    Purpose:
        Ensure zero data leakage by fitting exclusively on the training split, saving
        models/preprocessor.joblib and models/feature_columns.json.
    """
    config.ensure_directories()
    feature_cols, numeric_cols, categorical_cols = get_feature_column_lists(train_df)

    logger.info(
        "Fitting preprocessor on training data (%d rows, %d raw features: %d num, %d cat)",
        len(train_df),
        len(feature_cols),
        len(numeric_cols),
        len(categorical_cols),
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(train_df[feature_cols])

    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    logger.info(
        "Fitted preprocessor successfully. Expanded feature space: %d columns",
        len(transformed_feature_names),
    )

    # Save fitted preprocessor
    logger.info("Saving fitted preprocessor to: %s", config.PREPROCESSOR_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # Save list of expected raw feature columns for app validation contract
    logger.info("Saving feature column schema to: %s", config.FEATURE_COLUMNS_PATH)
    with open(config.FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    return preprocessor, feature_cols


def load_preprocessor(path: Optional[Path] = None) -> ColumnTransformer:
    """Load the persisted fitted ColumnTransformer from disk.

    Inputs:
        path: Optional Path to preprocessor artifact (default: config.PREPROCESSOR_PATH)
    Outputs:
        Fitted ColumnTransformer instance
    Purpose:
        Provide loaded preprocessor for model training, evaluation, and online inference.
    """
    target_path = path or config.PREPROCESSOR_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {target_path}. "
            "Please run fit_and_save_preprocessor() first."
        )
    return joblib.load(target_path)


def load_feature_columns(path: Optional[Path] = None) -> List[str]:
    """Load the list of expected input feature columns from disk.

    Inputs:
        path: Optional Path to feature columns schema JSON (default: config.FEATURE_COLUMNS_PATH)
    Outputs:
        List of column name strings
    Purpose:
        Validate uploaded CSV schemas against the expected feature contract.
    """
    target_path = path or config.FEATURE_COLUMNS_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Feature columns schema not found at {target_path}. "
            "Please run fit_and_save_preprocessor() first."
        )
    with open(target_path, "r", encoding="utf-8") as f:
        columns: List[str] = json.load(f)
    return columns


def transform_features(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame
) -> np.ndarray:
    """Transform input features using the fitted preprocessor.

    Inputs:
        preprocessor: Fitted ColumnTransformer instance
        df: Input DataFrame containing the required feature columns
    Outputs:
        Dense numpy 2D array of scaled numeric and one-hot categorical features
    Purpose:
        Provide consistent feature transformation pipeline for testing and prediction.
    """
    expected_cols = load_feature_columns()
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required feature columns: {missing_cols}")

    return preprocessor.transform(df[expected_cols])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src import data_loader
    train_df, _ = data_loader.load_processed()
    fit_and_save_preprocessor(train_df)

