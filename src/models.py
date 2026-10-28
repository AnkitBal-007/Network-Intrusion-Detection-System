"""Model factory and hyperparameter search grids.

Provides initializers, wrappers, and parameter grids for the algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Naive Bayes (GaussianNB)
6. XGBoost (XGBClassifier)
7. Support Vector Machine (Linear/RBF SVM trained on a stratified sample)
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from src import config


class SampledSVMClassifier(BaseEstimator, ClassifierMixin):
    """Support Vector Machine classifier trained on a stratified sample of the training data.

    Solves the computational bottleneck of training SVM on 100k+ records on 8 GB RAM
    laptops while preserving class distributions and providing calibrated probabilities.
    """

    def __init__(
        self,
        sample_size: int = 10000,
        C: float = 1.0,
        kernel: str = "linear",
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        """Initialize SampledSVMClassifier with sample size and regularization parameters."""
        self.sample_size = sample_size
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.clf_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "SampledSVMClassifier":
        """Fit the SVM estimator on a stratified sample of X and y."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if len(X_arr) > self.sample_size:
            X_sample, _, y_sample, _ = train_test_split(
                X_arr,
                y_arr,
                train_size=self.sample_size,
                stratify=y_arr,
                random_state=self.random_state,
            )
        else:
            X_sample, y_sample = X_arr, y_arr

        if self.kernel == "linear":
            base_svc = LinearSVC(
                C=self.C,
                random_state=self.random_state,
                max_iter=3000,
                dual="auto",
                class_weight="balanced",
            )
            self.clf_ = CalibratedClassifierCV(estimator=base_svc, cv=3)
        else:
            self.clf_ = SVC(
                C=self.C,
                kernel=self.kernel,
                class_weight="balanced",
                random_state=self.random_state,
            )

        self.clf_.fit(X_sample, y_sample)
        if hasattr(self.clf_, "classes_"):
            self.classes_ = self.clf_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.clf_ is None:
            raise RuntimeError("SampledSVMClassifier must be fitted before predict.")
        return self.clf_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        if self.clf_ is None:
            raise RuntimeError("SampledSVMClassifier must be fitted before predict_proba.")
        if hasattr(self.clf_, "predict_proba"):
            return self.clf_.predict_proba(X)
        decision = self.clf_.decision_function(X)
        if decision.ndim == 1:
            prob_1 = 1.0 / (1.0 + np.exp(-decision))
            return np.vstack([1.0 - prob_1, prob_1]).T
        exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)

    def decision_function(self, X: Any) -> np.ndarray:
        """Evaluate the decision function for samples in X."""
        if self.clf_ is None:
            raise RuntimeError("SampledSVMClassifier must be fitted before decision_function.")
        if hasattr(self.clf_, "decision_function"):
            return self.clf_.decision_function(X)
        probs = self.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs


class MultiClassXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier adapted for multi-class classification with class balancing and string/numeric target support."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = config.RANDOM_STATE,
        n_jobs: int = -1,
        eval_metric: str = "mlogloss",
    ) -> None:
        """Initialize MultiClassXGBClassifier with hyperparameter settings."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.clf_: Optional[XGBClassifier] = None
        self.classes_: Optional[np.ndarray] = None
        self.label_encoder_: Optional[LabelEncoder] = None

    def fit(self, X: Any, y: Any) -> "MultiClassXGBClassifier":
        """Fit the XGBoost classifier with balanced sample weights for multi-class targets."""
        y_arr = np.asarray(y)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_arr)
        self.classes_ = self.label_encoder_.classes_

        sw = compute_sample_weight("balanced", y_enc)
        self.clf_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric=self.eval_metric,
        )
        self.clf_.fit(X, y_enc, sample_weight=sw)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for samples in X, inverse transformed to original class names."""
        if self.clf_ is None or self.label_encoder_ is None:
            raise RuntimeError("MultiClassXGBClassifier must be fitted before predict.")
        y_pred = self.clf_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        if self.clf_ is None:
            raise RuntimeError("MultiClassXGBClassifier must be fitted before predict_proba.")
        return self.clf_.predict_proba(X)


def get_models(task: str = "binary") -> Dict[str, BaseEstimator]:
    """Instantiate candidate classifiers for evaluation with balanced weighting and seed=42.

    Inputs:
        task: 'binary' or 'multiclass'
    Outputs:
        Dictionary mapping model names to configured estimator instances
    Purpose:
        Provide standard algorithms for comparative benchmarking across tasks.
    """
    if task == "binary":
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=15,
                random_state=config.RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=15,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1,
            ),
            "Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "SVM (Sampled)": SampledSVMClassifier(
                sample_size=10000,
                C=1.0,
                kernel="linear",
                random_state=config.RANDOM_STATE,
            ),
        }
    elif task == "multiclass":
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=15,
                random_state=config.RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=15,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1,
            ),
            "Naive Bayes": GaussianNB(),
            "XGBoost": MultiClassXGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "SVM (Sampled)": SampledSVMClassifier(
                sample_size=10000,
                C=1.0,
                kernel="linear",
                random_state=config.RANDOM_STATE,
            ),
        }
    else:
        raise ValueError(f"Unsupported task: {task}. Choose 'binary' or 'multiclass'.")


def get_param_grids(task: str = "binary") -> Dict[str, Dict[str, Any]]:
    """Retrieve hyperparameter tuning search grids for candidate algorithms.

    Inputs:
        task: 'binary' or 'multiclass'
    Outputs:
        Dictionary mapping model names to hyperparameter distributions
    Purpose:
        Define tuning parameter spaces for top-performing algorithms.
    """
    return {
        "Random Forest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 150],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.08, 0.15],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "Decision Tree": {
            "max_depth": [8, 12, 16, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5],
            "criterion": ["gini", "entropy"],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        },
    }
