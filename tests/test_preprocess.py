"""Unit tests for preprocessor artifact, unseen category handling, and feature selection."""

import numpy as np
import pandas as pd
from src import config, features, preprocess


def test_models_artifacts_exist() -> None:
    """Verify that all required model and preprocessor artifacts exist on disk."""
    assert config.PREPROCESSOR_PATH.exists(), f"Missing {config.PREPROCESSOR_PATH}"
    assert config.FEATURE_COLUMNS_PATH.exists(), f"Missing {config.FEATURE_COLUMNS_PATH}"
    assert config.SELECTED_FEATURES_PATH.exists(), f"Missing {config.SELECTED_FEATURES_PATH}"

    # Verify non-empty
    assert config.PREPROCESSOR_PATH.stat().st_size > 0
    assert config.FEATURE_COLUMNS_PATH.stat().st_size > 0
    assert config.SELECTED_FEATURES_PATH.stat().st_size > 0


def test_preprocessor_transforms_new_data() -> None:
    """Verify that the saved preprocessor transforms new data and produces valid numeric matrix."""
    preprocessor = preprocess.load_preprocessor()
    feature_cols = preprocess.load_feature_columns()

    # Read sample data
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)
    assert all(col in sample_df.columns for col in feature_cols)

    X_transformed = preprocess.transform_features(preprocessor, sample_df)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_df)
    assert X_transformed.shape[1] == 194
    # Ensure no NaN or infinite values after transform
    assert not np.isnan(X_transformed).any()
    assert not np.isinf(X_transformed).any()


def test_preprocessor_handles_unseen_category() -> None:
    """Verify preprocessor handles unseen categorical levels gracefully with zero-padding."""
    preprocessor = preprocess.load_preprocessor()
    feature_cols = preprocess.load_feature_columns()

    # Load 1 row from sample and inject completely novel/unseen categorical strings
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH).head(1).copy()
    sample_df["proto"] = "quantum_protocol_xyz"
    sample_df["service"] = "nonexistent_service_999"
    sample_df["state"] = "UNSEEN_STATE_XYZ"

    # Must transform without error
    X_unseen = preprocess.transform_features(preprocessor, sample_df)

    assert isinstance(X_unseen, np.ndarray)
    assert X_unseen.shape == (1, 194)
    assert not np.isnan(X_unseen).any()

    # Check that one-hot encoded columns for unseen categories evaluate to 0
    all_names = preprocessor.get_feature_names_out().tolist()
    cat_indices = [i for i, name in enumerate(all_names) if name.startswith("cat__")]
    # All categorical indicators should be 0 because all 3 categories were unseen
    assert np.all(X_unseen[0, cat_indices] == 0.0)


def test_preprocessor_keeps_consistent_column_count() -> None:
    """Verify column count remains identical regardless of input batch size or seen/unseen categories."""
    preprocessor = preprocess.load_preprocessor()
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)

    batch_standard = preprocess.transform_features(preprocessor, sample_df.iloc[:5])
    batch_single = preprocess.transform_features(preprocessor, sample_df.iloc[[0]])

    unseen_batch = sample_df.iloc[:3].copy()
    unseen_batch["proto"] = "custom_proto"
    batch_unseen = preprocess.transform_features(preprocessor, unseen_batch)

    assert batch_standard.shape[1] == 194
    assert batch_single.shape[1] == 194
    assert batch_unseen.shape[1] == 194
    assert batch_standard.shape[1] == batch_single.shape[1] == batch_unseen.shape[1]


def test_selected_features_filtering() -> None:
    """Verify selected features can be loaded and sliced from transformed matrix."""
    preprocessor = preprocess.load_preprocessor()
    selected_cols = features.load_selected_features()
    all_cols = preprocessor.get_feature_names_out().tolist()

    assert len(selected_cols) == 30
    assert all(col in all_cols for col in selected_cols)

    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)
    X_full = preprocess.transform_features(preprocessor, sample_df)
    X_selected = features.filter_features(X_full, all_cols, selected_cols)

    assert X_selected.shape == (len(sample_df), 30)
    assert not np.isnan(X_selected).any()
