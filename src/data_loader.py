"""Data loading, cleaning, and preprocessing export module for UNSW-NB15 dataset.

Handles raw dataset ingestion, whitespace stripping, infinite/missing value handling,
duplicate removal with explicit logging, and processed dataset serialization.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from src import config

# Setup module logger using standard logging per rules.md
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and testing CSV files as pandas DataFrames.

    Inputs:
        None (uses paths from src.config)
    Outputs:
        Tuple of (train_df, test_df)
    Purpose:
        Read the official UNSW-NB15 dataset files from disk with existence checks.
    """
    if not config.TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {config.TRAIN_DATA_PATH}. "
            "Please ensure raw dataset CSVs are placed in data/raw/."
        )
    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found at {config.TEST_DATA_PATH}. "
            "Please ensure raw dataset CSVs are placed in data/raw/."
        )

    logger.info("Loading raw training set from: %s", config.TRAIN_DATA_PATH)
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    logger.info("Loaded raw training set with shape: %s", train_df.shape)

    logger.info("Loading raw testing set from: %s", config.TEST_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    logger.info("Loaded raw testing set with shape: %s", test_df.shape)

    return train_df, test_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean traffic DataFrame by handling strings, invalid numeric values, and duplicates.

    Inputs:
        df: Input pandas DataFrame (raw or unprocessed)
    Outputs:
        Cleaned pandas DataFrame
    Purpose:
        Ensure data integrity by:
        1. Stripping leading/trailing whitespace from string columns.
        2. Normalizing attack category names (e.g. 'Backdoors' -> 'Backdoor').
        3. Replacing inf/-inf with NaN and dropping/logging NaN rows.
        4. Removing duplicate records (ignoring the artificial 'id' column if present)
           and logging row removal reasons.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    logger.info("Starting cleaning pipeline on DataFrame with %d rows and %d columns", initial_rows, df_clean.shape[1])

    # 1. Strip whitespace from string/object columns
    string_cols = [c for c in df_clean.columns if df_clean[c].dtype == "object" or df_clean[c].dtype.name.startswith("str")]
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Normalize target column names if present
    if config.MULTI_TARGET in df_clean.columns:
        df_clean[config.MULTI_TARGET] = df_clean[config.MULTI_TARGET].replace({
            "Backdoors": "Backdoor"
        })

    # 2. Check and handle infinite values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        inf_mask = np.isinf(df_clean[numeric_cols].to_numpy())
        inf_count = int(inf_mask.sum())
        if inf_count > 0:
            logger.warning("Found %d infinite values across numeric columns; replacing with NaN", inf_count)
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        else:
            logger.info("No infinite values found in numeric columns")

    # 3. Check and handle NaN values
    nan_rows = int(df_clean.isna().any(axis=1).sum())
    if nan_rows > 0:
        logger.warning("Found %d rows containing NaN values; removing them to preserve model training integrity", nan_rows)
        df_clean = df_clean.dropna().reset_index(drop=True)
    else:
        logger.info("No NaN values found in DataFrame")

    # 4. Check and remove duplicates
    # In UNSW-NB15, the 'id' column is an arbitrary sequence number (1..N).
    # Duplicate checking excludes 'id' so identical network connections are caught.
    subset_cols = [c for c in df_clean.columns if c not in config.DROP_COLS]
    duplicates_mask = df_clean.duplicated(subset=subset_cols, keep="first")
    n_duplicates = int(duplicates_mask.sum())

    if n_duplicates > 0:
        logger.info(
            "Removing %d duplicate rows (identified across all feature/target columns excluding 'id')",
            n_duplicates,
        )
        df_clean = df_clean[~duplicates_mask].reset_index(drop=True)
    else:
        logger.info("No duplicate rows found")

    final_rows = len(df_clean)
    rows_dropped = initial_rows - final_rows
    logger.info(
        "Cleaning finished: %d rows retained, %d rows removed (%.2f%% of initial rows)",
        final_rows,
        rows_dropped,
        (rows_dropped / initial_rows) * 100 if initial_rows > 0 else 0.0,
    )

    return df_clean


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Path, Path]:
    """Save cleaned datasets to data/processed in both CSV and Parquet formats.

    Inputs:
        train_df: Cleaned training DataFrame
        test_df: Cleaned testing DataFrame
    Outputs:
        Tuple of (train_processed_path, test_processed_path)
    Purpose:
        Persist clean splits to disk for reproducible model training and fast reloading.
    """
    config.ensure_directories()
    train_path = config.PROCESSED_DATA_DIR / "train_cleaned.parquet"
    test_path = config.PROCESSED_DATA_DIR / "test_cleaned.parquet"
    train_csv_path = config.PROCESSED_DATA_DIR / "train_cleaned.csv"
    test_csv_path = config.PROCESSED_DATA_DIR / "test_cleaned.csv"

    logger.info("Saving cleaned training set (%d rows) to %s", len(train_df), train_path)
    train_df.to_parquet(train_path, index=False)
    train_df.to_csv(train_csv_path, index=False)

    logger.info("Saving cleaned testing set (%d rows) to %s", len(test_df), test_path)
    test_df.to_parquet(test_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return train_path, test_path


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned training and testing datasets from data/processed.

    Inputs:
        None (uses paths from src.config)
    Outputs:
        Tuple of (train_df, test_df)
    Purpose:
        Load preprocessed datasets, preferring Parquet for speed and falling back to CSV.
    """
    train_parquet = config.PROCESSED_DATA_DIR / "train_cleaned.parquet"
    test_parquet = config.PROCESSED_DATA_DIR / "test_cleaned.parquet"
    train_csv = config.PROCESSED_DATA_DIR / "train_cleaned.csv"
    test_csv = config.PROCESSED_DATA_DIR / "test_cleaned.csv"

    if train_parquet.exists() and test_parquet.exists():
        logger.info("Loading cleaned datasets from Parquet: %s, %s", train_parquet, test_parquet)
        return pd.read_parquet(train_parquet), pd.read_parquet(test_parquet)

    if train_csv.exists() and test_csv.exists():
        logger.info("Loading cleaned datasets from CSV: %s, %s", train_csv, test_csv)
        return pd.read_csv(train_csv), pd.read_csv(test_csv)

    raise FileNotFoundError(
        "Processed datasets not found in data/processed/. "
        "Run data cleaning pipeline first via save_processed()."
    )


def create_sample_input(
    test_df: pd.DataFrame,
    target_samples: int = 200,
    random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Generate a stratified sample from the test set for app demonstration and testing.

    Inputs:
        test_df: Raw or cleaned test DataFrame containing attack_cat
        target_samples: Approximate number of rows to sample (~200)
        random_state: Random state for deterministic reproducibility
    Outputs:
        Sample DataFrame with 'id', 'label', and 'attack_cat' removed
    Purpose:
        Create data/sample/sample_input.csv with representatives from all attack categories.
    """
    config.ensure_directories()
    logger.info("Generating stratified sample of approx %d rows from test set", target_samples)

    if config.MULTI_TARGET not in test_df.columns:
        raise ValueError(f"Column '{config.MULTI_TARGET}' required for stratified sampling.")

    # Sample proportionally per category, guaranteeing at least 1 row per attack class
    sample_parts: List[pd.DataFrame] = []
    total_test = len(test_df)

    for cat_name, group in test_df.groupby(config.MULTI_TARGET, observed=False):
        # Calculate proportional allocation with a minimum of 1 row
        n_cat = max(1, int(round((len(group) / total_test) * target_samples)))
        # Sample with fixed random state
        sampled_group = group.sample(n=min(n_cat, len(group)), random_state=random_state)
        sample_parts.append(sampled_group)

    sample_combined = pd.concat(sample_parts, ignore_index=True)
    logger.info(
        "Stratified sample generated with %d total rows across %d categories:\n%s",
        len(sample_combined),
        sample_combined[config.MULTI_TARGET].nunique(),
        sample_combined[config.MULTI_TARGET].value_counts().to_string(),
    )

    # Exclude id, label, and attack_cat per contract
    cols_to_drop = [c for c in [config.DROP_COLS[0], config.BINARY_TARGET, config.MULTI_TARGET] if c in sample_combined.columns]
    sample_features = sample_combined.drop(columns=cols_to_drop).copy()

    # Save to data/sample/sample_input.csv
    sample_features.to_csv(config.SAMPLE_INPUT_PATH, index=False)
    logger.info("Saved sample input to: %s (%d rows, %d feature columns)", config.SAMPLE_INPUT_PATH, len(sample_features), sample_features.shape[1])

    return sample_features


if __name__ == "__main__":
    train_raw, test_raw = load_raw()
    train_clean = clean(train_raw)
    test_clean = clean(test_raw)
    save_processed(train_clean, test_clean)
    create_sample_input(test_clean)

