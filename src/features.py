"""Feature selection and engineering module.

Identifies the most informative network traffic features using model-based
Random Forest importance or mutual information, computed exclusively on training data.
Persists chosen feature sets to models/selected_features.json.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from src import config

logger = logging.getLogger(__name__)


def select_features_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_k: int = 30,
    random_state: int = config.RANDOM_STATE,
) -> List[str]:
    """Select top K features based on Random Forest feature importances on training data.

    Inputs:
        X_train: Transformed 2D feature matrix (fitted on training data only)
        y_train: 1D array of training target labels
        feature_names: List of all feature names corresponding to columns of X_train
        top_k: Number of most important features to retain (default: 30)
        random_state: Random state for deterministic model fitting
    Outputs:
        List of the selected top K feature names
    Purpose:
        Perform leak-free feature reduction to accelerate modeling and remove noise.
    """
    logger.info(
        "Fitting Random Forest feature selector on training split (%d samples, %d features)",
        X_train.shape[0],
        X_train.shape[1],
    )

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_indices = sorted_indices[:top_k]

    selected_features = [feature_names[i] for i in top_indices]
    feature_importance_dict = {
        feature_names[i]: float(round(importances[i], 5))
        for i in top_indices
    }

    cumulative_importance = float(round(np.sum(importances[top_indices]), 4))
    logger.info(
        "Selected top %d features accounting for %.2f%% cumulative importance",
        top_k,
        cumulative_importance * 100,
    )

    # Persist selection to models/selected_features.json
    save_payload: Dict[str, Any] = {
        "method": "random_forest_importance",
        "top_k": top_k,
        "cumulative_importance": cumulative_importance,
        "selected_features": selected_features,
        "importances": feature_importance_dict,
    }

    config.ensure_directories()
    with open(config.SELECTED_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    logger.info("Saved selected features metadata to: %s", config.SELECTED_FEATURES_PATH)
    return selected_features


def select_features_mi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_k: int = 30,
    sample_size: int = 25000,
    random_state: int = config.RANDOM_STATE,
) -> List[str]:
    """Select top K features using mutual information computed on training data.

    Inputs:
        X_train: Transformed 2D feature matrix
        y_train: 1D array of training target labels
        feature_names: List of all feature names
        top_k: Number of top features to retain
        sample_size: Stratified sample size for fast mutual information estimation
        random_state: Deterministic random seed
    Outputs:
        List of selected feature names
    Purpose:
        Alternative non-linear feature selection method based on information theory.
    """
    logger.info("Computing mutual information feature scores on training data sample")
    n_samples = len(y_train)

    if n_samples > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    mi_scores = mutual_info_classif(X_sub, y_sub, random_state=random_state)
    top_indices = np.argsort(mi_scores)[::-1][:top_k]
    selected_features = [feature_names[i] for i in top_indices]

    return selected_features


def load_selected_features() -> List[str]:
    """Load the persisted selected feature names from disk.

    Inputs:
        None (loads from config.SELECTED_FEATURES_PATH)
    Outputs:
        List of selected feature name strings
    Purpose:
        Retrieve the exact chosen features for training and inference pipelines.
    """
    if not config.SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Selected features not found at {config.SELECTED_FEATURES_PATH}. "
            "Please run feature selection first."
        )

    with open(config.SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "selected_features" in data:
        return data["selected_features"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized format in {config.SELECTED_FEATURES_PATH}")


def filter_features(
    X: np.ndarray,
    all_feature_names: List[str],
    selected_feature_names: List[str],
) -> np.ndarray:
    """Filter transformed feature matrix columns to retain only the selected features.

    Inputs:
        X: Transformed 2D feature matrix
        all_feature_names: Column names corresponding to X
        selected_feature_names: List of features to retain
    Outputs:
        Filtered 2D numpy array containing only selected columns
    Purpose:
        Apply consistent column slicing between full preprocessor output and models.
    """
    name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
    missing = [name for name in selected_feature_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Selected features not present in transformed matrix: {missing}")

    indices = [name_to_idx[name] for name in selected_feature_names]
    return X[:, indices]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that filters columns by selected feature names."""

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        all_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize FeatureSelector with optional feature list and full feature names."""
        self.selected_features = selected_features
        self.all_feature_names = all_feature_names
        self.feature_names_in_: Optional[List[str]] = None
        self.indices_: Optional[List[int]] = None

    def fit(self, X: Any, y: Any = None) -> "FeatureSelector":
        """Fit feature selector by mapping column names to integer indices."""
        if self.selected_features is None:
            self.selected_features = load_selected_features()

        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            name_to_idx = {name: i for i, name in enumerate(self.feature_names_in_)}
            self.indices_ = [name_to_idx[name] for name in self.selected_features if name in name_to_idx]
        else:
            if self.all_feature_names is None:
                from src import preprocess
                try:
                    prep = preprocess.load_preprocessor()
                    self.all_feature_names = prep.get_feature_names_out().tolist()
                except Exception:
                    self.all_feature_names = None

            if self.all_feature_names is not None:
                name_to_idx = {name: i for i, name in enumerate(self.all_feature_names)}
                self.indices_ = [name_to_idx[name] for name in self.selected_features if name in name_to_idx]
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Extract selected columns from 2D array or DataFrame."""
        if self.indices_ is None:
            self.fit(X)

        if hasattr(X, "iloc"):
            if self.indices_ is not None and len(self.indices_) > 0:
                return X.iloc[:, self.indices_].to_numpy()
            return X[self.selected_features].to_numpy()
        elif isinstance(X, np.ndarray):
            if self.indices_ is not None and len(self.indices_) > 0:
                return X[:, self.indices_]
            return X
        return np.asarray(X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src import data_loader, preprocess
    train_df, _ = data_loader.load_processed()
    prep = preprocess.load_preprocessor()
    X_train_full = prep.transform(train_df[preprocess.load_feature_columns()])
    y_train = train_df[config.BINARY_TARGET].to_numpy(dtype=int)
    feature_names = prep.get_feature_names_out().tolist()
    select_features_rf(X_train_full, y_train, feature_names=feature_names, top_k=30)

