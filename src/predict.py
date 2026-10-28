"""Prediction and cascading inference module.

Validates input DataFrame against expected schemas, applies preprocessor transforms,
executes binary detection, and runs multi-class categorization on detected attack rows.
Follows the cascade design in architecture.md: Normal rows get category 'Normal',
while Attack rows are attributed to the highest-probability non-Normal attack class.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from src import config, preprocess

logger = logging.getLogger(__name__)


def load_artifacts(
    preprocessor_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    best_binary_path: Optional[Path] = None,
    best_multiclass_path: Optional[Path] = None,
) -> Tuple[Any, List[str], Any, Any]:
    """Load all necessary serialized artifacts for inference.

    Inputs:
        preprocessor_path: Path to models/preprocessor.joblib (default from config)
        feature_columns_path: Path to models/feature_columns.json (default from config)
        best_binary_path: Path to models/best_binary.joblib (default from config)
        best_multiclass_path: Path to models/best_multiclass.joblib (default from config)
    Outputs:
        Tuple of (preprocessor, feature_columns, best_binary_pipeline, best_multiclass_pipeline)
    Raises:
        FileNotFoundError: If any of the required model or preprocessor files are missing.
    Purpose:
        Single entry point for loading models in inference pipelines and web application.
    """
    p_path = preprocessor_path or config.PREPROCESSOR_PATH
    fc_path = feature_columns_path or config.FEATURE_COLUMNS_PATH
    bin_path = best_binary_path or config.BEST_BINARY_MODEL_PATH
    multi_path = best_multiclass_path or config.BEST_MULTICLASS_MODEL_PATH

    required_files = [
        ("Preprocessor", p_path),
        ("Feature columns", fc_path),
        ("Binary model", bin_path),
        ("Multi-class model", multi_path),
    ]

    missing = [name for name, path in required_files if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Model artifacts not found ({missing_str}). Run the training pipeline first: python -m src.train --task all"
        )

    preprocessor = preprocess.load_preprocessor(p_path)
    feature_cols = preprocess.load_feature_columns(fc_path)
    best_binary = joblib.load(bin_path)
    best_multiclass = joblib.load(multi_path)

    return preprocessor, feature_cols, best_binary, best_multiclass


def predict_dataframe(
    df: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None,
    best_binary: Optional[Any] = None,
    best_multiclass: Optional[Any] = None,
) -> pd.DataFrame:
    """Run cascade intrusion detection prediction on an input DataFrame.

    Cascade Workflow (per architecture.md Section 3.8):
        1. Validate columns against feature_columns.json.
        2. Raise clear errors if DataFrame is empty or columns are missing.
        3. Drop id / label / attack_cat from feature extraction if present.
        4. Preprocess features through saved ColumnTransformer.
        5. Predict binary label ('Normal' vs 'Attack').
        6. For rows flagged as Attack, predict multi-class probabilities and choose
           the non-Normal category with the highest probability.
        7. Assign category 'Normal' to all rows predicted as Normal.
        8. Return input DataFrame with added 'prediction' and 'category' columns.

    Inputs:
        df: Input network traffic DataFrame
        preprocessor: Optional pre-loaded preprocessor
        feature_cols: Optional list of expected feature columns
        best_binary: Optional pre-loaded binary model pipeline
        best_multiclass: Optional pre-loaded multi-class model pipeline
    Outputs:
        DataFrame copy with added 'prediction' ('Normal'/'Attack') and 'category' columns.
    Raises:
        ValueError: If input DataFrame is empty or missing required feature columns.
    """
    if df is None or df.empty or len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    # Load artifacts if not explicitly provided (e.g. by caching in Streamlit)
    if (
        preprocessor is None
        or feature_cols is None
        or best_binary is None
        or best_multiclass is None
    ):
        p_loaded, fc_loaded, bin_loaded, multi_loaded = load_artifacts()
        preprocessor = preprocessor or p_loaded
        feature_cols = feature_cols or fc_loaded
        best_binary = best_binary or bin_loaded
        best_multiclass = best_multiclass or multi_loaded

    # Check for missing required feature columns
    missing_columns = [col for col in feature_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The input data is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    # Copy input DataFrame to preserve original user data without in-place mutation
    result_df = df.copy()

    # Preprocess features: extract required columns and transform
    X_features = df[feature_cols]
    X_trans = preprocess.transform_features(preprocessor, X_features)

    # Stage 1: Binary Classification
    binary_preds = best_binary.predict(X_trans)
    result_df["prediction"] = np.where(binary_preds == 1, "Attack", "Normal")

    # Stage 2: Multi-Class Attribution for rows flagged as Attack
    categories = ["Normal"] * len(df)
    attack_mask = (binary_preds == 1)

    if np.any(attack_mask):
        X_attack = X_trans[attack_mask]
        probs = best_multiclass.predict_proba(X_attack)

        # Classes represented in the multi-class model
        classes = best_multiclass.classes_
        non_normal_indices = [i for i, c in enumerate(classes) if c != "Normal"]

        if len(non_normal_indices) > 0:
            sub_probs = probs[:, non_normal_indices]
            best_cat_idx = np.argmax(sub_probs, axis=1)
            attack_categories = [classes[non_normal_indices[idx]] for idx in best_cat_idx]
        else:
            attack_categories = ["Attack"] * len(X_attack)

        attack_positions = np.where(attack_mask)[0]
        for row_pos, cat_name in zip(attack_positions, attack_categories):
            categories[row_pos] = cat_name

    result_df["category"] = categories
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if config.SAMPLE_INPUT_PATH.exists():
        sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)
        res = predict_dataframe(sample_df)
        print("Sample predictions completed successfully:")
        print(res[["prediction", "category"]].value_counts().to_string())
    else:
        print("Sample input CSV not found at", config.SAMPLE_INPUT_PATH)

