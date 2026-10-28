"""Unit tests for prediction pipeline, cascade inference logic, and model artifacts."""

import numpy as np
import pandas as pd
import pytest
from src import config, predict


def test_attack_classes_defined() -> None:
    """Verify that binary and multi-class categories are properly initialized."""
    assert config.BINARY_TARGET == "label"
    assert config.MULTI_TARGET == "attack_cat"
    assert "Normal" in config.ATTACK_CLASSES
    assert len(config.NON_NORMAL_ATTACK_CLASSES) == 9
    assert "Exploits" in config.NON_NORMAL_ATTACK_CLASSES
    assert "Worms" in config.NON_NORMAL_ATTACK_CLASSES


def test_load_artifacts() -> None:
    """Verify that load_artifacts retrieves preprocessor, feature columns, and models."""
    preprocessor, feature_cols, best_binary, best_multiclass = predict.load_artifacts()
    assert preprocessor is not None
    assert isinstance(feature_cols, list)
    assert len(feature_cols) == 42
    assert hasattr(best_binary, "predict")
    assert hasattr(best_multiclass, "predict_proba")
    assert hasattr(best_multiclass, "classes_")


def test_predict_dataframe_sample_input() -> None:
    """Verify cascade inference on data/sample/sample_input.csv.

    Checks:
        1. Output DataFrame contains all original columns plus 'prediction' and 'category'.
        2. Prediction values are strictly 'Normal' or 'Attack'.
        3. All rows predicted as Normal have category 'Normal'.
        4. All rows predicted as Attack have a category from NON_NORMAL_ATTACK_CLASSES.
        5. Row count is preserved.
    """
    assert config.SAMPLE_INPUT_PATH.exists()
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)
    original_cols = list(sample_df.columns)

    result_df = predict.predict_dataframe(sample_df)

    # 1. Output columns check
    assert "prediction" in result_df.columns
    assert "category" in result_df.columns
    for col in original_cols:
        assert col in result_df.columns

    # 2. Row count preservation
    assert len(result_df) == len(sample_df)

    # 3. Prediction values domain
    unique_preds = set(result_df["prediction"].unique())
    assert unique_preds.issubset({"Normal", "Attack"})

    # 4. Normal rows have category 'Normal'
    normal_mask = result_df["prediction"] == "Normal"
    normal_categories = result_df.loc[normal_mask, "category"].unique()
    assert len(normal_categories) == 1
    assert normal_categories[0] == "Normal"

    # 5. Attack rows have non-Normal attack categories
    attack_mask = result_df["prediction"] == "Attack"
    if np.any(attack_mask):
        attack_categories = set(result_df.loc[attack_mask, "category"].unique())
        assert attack_categories.issubset(set(config.NON_NORMAL_ATTACK_CLASSES))
        assert "Normal" not in attack_categories


def test_predict_dataframe_empty_input_error() -> None:
    """Verify that passing an empty DataFrame raises a clear ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        predict.predict_dataframe(empty_df)


def test_predict_dataframe_missing_column_error() -> None:
    """Verify that missing required feature columns raises a descriptive ValueError."""
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)

    # Drop two essential feature columns
    drop_candidates = ["proto", "sbytes"]
    dropped_df = sample_df.drop(columns=drop_candidates)

    with pytest.raises(ValueError, match="The input data is missing required columns") as exc_info:
        predict.predict_dataframe(dropped_df)

    # Verify both dropped column names appear in the error message
    err_msg = str(exc_info.value)
    assert "proto" in err_msg
    assert "sbytes" in err_msg


def test_predict_dataframe_extra_columns_ignored() -> None:
    """Verify that extra arbitrary columns in uploaded CSV are preserved and ignored."""
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH)
    sample_df["id"] = range(len(sample_df))
    sample_df["custom_metadata"] = "test_sensor_01"

    result_df = predict.predict_dataframe(sample_df)

    assert "id" in result_df.columns
    assert "custom_metadata" in result_df.columns
    assert "prediction" in result_df.columns
    assert "category" in result_df.columns
    assert len(result_df) == len(sample_df)


def test_predict_dataframe_unseen_categorical_value() -> None:
    """Verify that unseen categorical values are handled gracefully by handle_unknown='ignore'."""
    sample_df = pd.read_csv(config.SAMPLE_INPUT_PATH).head(10).copy()
    sample_df["proto"] = "completely_novel_protocol_xyz"

    result_df = predict.predict_dataframe(sample_df)

    assert len(result_df) == 10
    assert "prediction" in result_df.columns
    assert "category" in result_df.columns
