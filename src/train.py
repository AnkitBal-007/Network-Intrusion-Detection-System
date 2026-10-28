"""Model training, stratified cross-validation, and hyperparameter tuning orchestration.

Executes stratified 5-fold cross-validation on all candidate algorithms,
tunes the leading models by macro-F1, packages the winning estimator with
FeatureSelector into a unified pipeline, and persists models/best_binary.joblib.
"""

import logging
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src import config, evaluate, features, models, preprocess

logger = logging.getLogger(__name__)


def evaluate_cv_binary_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Run stratified 5-fold cross-validation across all seven algorithms for binary classification.

    Inputs:
        X_train: Transformed and feature-selected 2D array of training samples
        y_train: 1D array of binary ground truth targets (0 = Normal, 1 = Attack)
        cv_splits: Number of cross-validation folds (default: 5)
        random_state: Deterministic seed for stratified folding
    Outputs:
        DataFrame summarizing mean CV metrics (accuracy, precision, recall, f1_macro, roc_auc)
    Purpose:
        Perform leak-free comparative benchmarking across candidate algorithms.
    """
    logger.info(
        "Starting stratified %d-fold cross-validation across all candidate models (%d samples, %d features)",
        cv_splits,
        X_train.shape[0],
        X_train.shape[1],
    )

    candidate_models = models.get_models(task="binary")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1_macro", "roc_auc"]

    cv_results: List[Dict[str, Any]] = []

    for name, model in candidate_models.items():
        logger.info("Evaluating 5-fold cross-validation for: %s", name)
        # Run sequential jobs per model to prevent CPU contention on 8 GB RAM machines
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        acc = float(scores["test_accuracy"].mean())
        prec = float(scores["test_precision"].mean())
        rec = float(scores["test_recall"].mean())
        f1_macro = float(scores["test_f1_macro"].mean())
        roc_auc = float(scores["test_roc_auc"].mean())

        logger.info(
            "%s -> Acc: %.4f | Prec: %.4f | Recall: %.4f | Macro-F1: %.4f | ROC-AUC: %.4f",
            name,
            acc,
            prec,
            rec,
            f1_macro,
            roc_auc,
        )

        cv_results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(roc_auc, 4),
        })

    df_cv = pd.DataFrame(cv_results).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    return df_cv


def tune_binary_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 8,
    cv_splits: int = 3,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune hyperparameters for a specified candidate model using RandomizedSearchCV.

    Inputs:
        model_name: Name of algorithm ('Random Forest', 'XGBoost', etc.)
        X_train: 2D training feature array
        y_train: 1D training target array
        n_iter: Number of parameter combinations sampled
        cv_splits: Stratified CV splits during hyperparameter search
        random_state: Deterministic search seed
    Outputs:
        Tuple of (best_estimator, best_params, best_macro_f1_score)
    Purpose:
        Optimize leading algorithms on training folds without test-set leakage.
    """
    logger.info("Tuning hyperparameters for %s using RandomizedSearchCV (n_iter=%d)", model_name, n_iter)
    all_models = models.get_models(task="binary")
    param_grids = models.get_param_grids(task="binary")

    if model_name not in all_models or model_name not in param_grids:
        raise ValueError(f"Model '{model_name}' not available for hyperparameter tuning.")

    base_estimator = all_models[model_name]
    param_grid = param_grids[model_name]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Tuned %s successfully. Best Macro-F1: %.4f | Best Params: %s",
        model_name,
        search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_and_select_best_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """Execute complete binary modeling workflow, evaluate on held-out test set, and persist best model.

    Inputs:
        train_df: Cleaned training DataFrame
        test_df: Cleaned testing DataFrame
    Outputs:
        Tuple of (fitted_pipeline, comparison_dataframe, test_metrics_dictionary)
    Purpose:
        Train all 7 models, tune top candidates, select the best by F1 and Attack recall,
        evaluate on held-out test set, export figures and reports, and save models/best_binary.joblib.
    """
    config.ensure_directories()
    preprocessor = preprocess.load_preprocessor()
    feature_cols = preprocess.load_feature_columns()
    selected_features = features.load_selected_features()
    all_feature_names = preprocessor.get_feature_names_out().tolist()

    # Preprocess training and testing splits
    logger.info("Transforming training features and applying feature selection...")
    X_train_full = preprocess.transform_features(preprocessor, train_df[feature_cols])
    X_train_sel = features.filter_features(X_train_full, all_feature_names, selected_features)
    y_train = np.asarray(train_df[config.BINARY_TARGET], dtype=int)

    X_test_full = preprocess.transform_features(preprocessor, test_df[feature_cols])
    X_test_sel = features.filter_features(X_test_full, all_feature_names, selected_features)
    y_test = np.asarray(test_df[config.BINARY_TARGET], dtype=int)

    # 1. 5-Fold Cross-Validation on all 7 algorithms
    df_cv = evaluate_cv_binary_models(X_train_sel, y_train, cv_splits=5)

    # 2. Tune top 2-3 models
    top_models = ["Random Forest", "XGBoost", "Decision Tree"]
    tuned_estimators: Dict[str, Any] = {}

    for model_name in top_models:
        tuned_clf, _, _ = tune_binary_model(model_name, X_train_sel, y_train, n_iter=6, cv_splits=3)
        tuned_estimators[model_name] = tuned_clf

    # 3. Evaluate all candidate and tuned models on the held-out test set
    test_benchmark_results: List[Dict[str, Any]] = []
    trained_candidates = models.get_models(task="binary")

    # Fit baseline models on full training set and evaluate on test set
    fitted_models: Dict[str, Any] = {}
    for name, clf in trained_candidates.items():
        logger.info("Fitting baseline %s on full training set...", name)
        clf.fit(X_train_sel, y_train)
        fitted_models[name] = clf

        preds = clf.predict(X_test_sel)
        probs = clf.predict_proba(X_test_sel) if hasattr(clf, "predict_proba") else None
        metrics = evaluate.compute_binary_metrics(y_test, preds, probs)
        metrics["model"] = name
        test_benchmark_results.append(metrics)

    # Evaluate tuned models
    for name, tuned_clf in tuned_estimators.items():
        tuned_name = f"{name} (Tuned)"
        preds = tuned_clf.predict(X_test_sel)
        probs = tuned_clf.predict_proba(X_test_sel) if hasattr(tuned_clf, "predict_proba") else None
        metrics = evaluate.compute_binary_metrics(y_test, preds, probs)
        metrics["model"] = tuned_name
        test_benchmark_results.append(metrics)
        fitted_models[tuned_name] = tuned_clf

    # 4. Save test comparison table to reports/binary_comparison.csv
    df_comparison = evaluate.save_comparison_table(
        test_benchmark_results,
        config.BINARY_COMPARISON_PATH,
    )

    # 5. Select best model based on macro-F1 and recall on Attack class per rules.md
    best_row = df_comparison.iloc[0]
    best_model_name = str(best_row["model"])
    best_estimator = fitted_models[best_model_name]
    logger.info(
        "Selected best binary model: %s (Test Macro-F1: %.4f, Attack Recall: %.4f)",
        best_model_name,
        best_row["f1_macro"],
        best_row["recall"],
    )

    # 6. Wrap best estimator with FeatureSelector in a Pipeline
    # Slices the 194-column preprocessed matrix to the 30 selected features automatically
    best_pipeline = Pipeline([
        ("selector", features.FeatureSelector(selected_features=selected_features, all_feature_names=all_feature_names)),
        ("classifier", best_estimator),
    ])

    # Save artifact to models/best_binary.joblib
    logger.info("Saving best binary model pipeline to: %s", config.BEST_BINARY_MODEL_PATH)
    joblib.dump(best_pipeline, config.BEST_BINARY_MODEL_PATH)

    # 7. Generate and save evaluation figures for the best model
    best_preds = best_pipeline.predict(X_test_full)
    best_probs = best_pipeline.predict_proba(X_test_full) if hasattr(best_pipeline, "predict_proba") else None

    evaluate.plot_binary_confusion_matrix(
        y_test,
        best_preds,
        best_model_name,
        config.FIGURES_DIR / "cm_binary_best.png",
    )
    if best_probs is not None:
        evaluate.plot_binary_roc_curve(
            y_test,
            best_probs,
            best_model_name,
            config.FIGURES_DIR / "roc_binary_best.png",
        )

    best_metrics = evaluate.compute_binary_metrics(y_test, best_preds, best_probs)
    best_metrics["model"] = best_model_name

    return best_pipeline, df_comparison, best_metrics


def evaluate_cv_multiclass_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Run stratified 5-fold cross-validation across all seven algorithms for multi-class classification.

    Inputs:
        X_train: Transformed and feature-selected 2D array of training samples
        y_train: 1D array of multi-class ground truth targets (strings)
        cv_splits: Number of cross-validation folds (default: 5)
        random_state: Deterministic seed for stratified folding
    Outputs:
        DataFrame summarizing mean CV metrics (accuracy, precision_macro, recall_macro, f1_macro, f1_weighted)
    Purpose:
        Perform leak-free comparative benchmarking across candidate algorithms for attack categorization.
    """
    X_mat = np.asarray(X_train)
    y_arr = np.asarray(y_train, dtype=str)

    logger.info(
        "Starting stratified %d-fold cross-validation across all multi-class candidate models (%d samples, %d features)",
        cv_splits,
        X_mat.shape[0],
        X_mat.shape[1],
    )

    candidate_models = models.get_models(task="multiclass")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]

    cv_results: List[Dict[str, Any]] = []

    for name, model in candidate_models.items():
        logger.info("Evaluating 5-fold cross-validation for multi-class model: %s", name)
        scores = cross_validate(
            model,
            X_mat,
            y_arr,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        acc = float(scores["test_accuracy"].mean())
        prec_macro = float(scores["test_precision_macro"].mean())
        rec_macro = float(scores["test_recall_macro"].mean())
        f1_macro = float(scores["test_f1_macro"].mean())
        f1_weighted = float(scores["test_f1_weighted"].mean())

        logger.info(
            "%s -> Acc: %.4f | Macro-Prec: %.4f | Macro-Recall: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
            name,
            acc,
            prec_macro,
            rec_macro,
            f1_macro,
            f1_weighted,
        )

        cv_results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        })

    df_cv = pd.DataFrame(cv_results).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    return df_cv


def tune_multiclass_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 8,
    cv_splits: int = 3,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune hyperparameters for a candidate multi-class model using RandomizedSearchCV.

    Inputs:
        model_name: Name of algorithm ('Random Forest', 'XGBoost', etc.)
        X_train: 2D training feature array
        y_train: 1D training target array
        n_iter: Number of parameter combinations sampled
        cv_splits: Stratified CV splits during hyperparameter search
        random_state: Deterministic search seed
    Outputs:
        Tuple of (best_estimator, best_params, best_macro_f1_score)
    Purpose:
        Optimize leading algorithms on training folds by macro-F1 without test-set leakage.
    """
    logger.info("Tuning multi-class hyperparameters for %s using RandomizedSearchCV (n_iter=%d)", model_name, n_iter)
    all_models = models.get_models(task="multiclass")
    param_grids = models.get_param_grids(task="multiclass")

    if model_name not in all_models or model_name not in param_grids:
        raise ValueError(f"Model '{model_name}' not available for hyperparameter tuning.")

    base_estimator = all_models[model_name]
    param_grid = param_grids[model_name]

    X_mat = np.asarray(X_train)
    y_arr = np.asarray(y_train, dtype=str)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_mat, y_arr)

    logger.info(
        "Tuned %s successfully. Best Macro-F1: %.4f | Best Params: %s",
        model_name,
        search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_and_select_best_multiclass(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Execute complete multi-class modeling workflow, evaluate on held-out test set, and persist best model.

    Inputs:
        train_df: Cleaned training DataFrame
        test_df: Cleaned testing DataFrame
    Outputs:
        Tuple of (fitted_pipeline, cv_comparison_dataframe, test_comparison_dataframe)
    Purpose:
        Train all 7 models, tune top candidates, select the best by macro-F1 and rare class recall,
        evaluate on held-out test set, export figures and reports, and save models/best_multiclass.joblib.
    """
    config.ensure_directories()
    preprocessor = preprocess.load_preprocessor()
    feature_cols = preprocess.load_feature_columns()
    selected_features = features.load_selected_features()
    all_feature_names = preprocessor.get_feature_names_out().tolist()

    # Preprocess training and testing splits
    logger.info("Transforming training features and applying feature selection for multi-class...")
    X_train_full = preprocess.transform_features(preprocessor, train_df[feature_cols])
    X_train_sel = features.filter_features(X_train_full, all_feature_names, selected_features)
    y_train = np.asarray(train_df[config.MULTI_TARGET], dtype=str)

    X_test_full = preprocess.transform_features(preprocessor, test_df[feature_cols])
    X_test_sel = features.filter_features(X_test_full, all_feature_names, selected_features)
    y_test = np.asarray(test_df[config.MULTI_TARGET], dtype=str)

    # 1. 5-Fold Cross-Validation on all 7 algorithms
    df_cv = evaluate_cv_multiclass_models(X_train_sel, y_train, cv_splits=5)

    # 2. Tune top 2-3 models (Random Forest, XGBoost, Decision Tree)
    top_models = ["Random Forest", "XGBoost", "Decision Tree"]
    tuned_estimators: Dict[str, Any] = {}

    for model_name in top_models:
        tuned_clf, _, _ = tune_multiclass_model(model_name, X_train_sel, y_train, n_iter=6, cv_splits=3)
        tuned_estimators[model_name] = tuned_clf

    # 3. Evaluate all candidate and tuned models on the held-out test set
    test_benchmark_results: List[Dict[str, Any]] = []
    trained_candidates = models.get_models(task="multiclass")

    fitted_models: Dict[str, Any] = {}
    for name, clf in trained_candidates.items():
        logger.info("Fitting baseline multi-class %s on full training set...", name)
        clf.fit(X_train_sel, y_train)
        fitted_models[name] = clf

        preds = clf.predict(X_test_sel)
        metrics = evaluate.compute_multiclass_metrics(y_test, preds)
        metrics["model"] = name
        test_benchmark_results.append(metrics)

    for name, tuned_clf in tuned_estimators.items():
        tuned_name = f"{name} (Tuned)"
        preds = tuned_clf.predict(X_test_sel)
        metrics = evaluate.compute_multiclass_metrics(y_test, preds)
        metrics["model"] = tuned_name
        test_benchmark_results.append(metrics)
        fitted_models[tuned_name] = tuned_clf

    # 4. Save test comparison table to reports/multiclass_comparison.csv
    df_comparison = evaluate.save_comparison_table(
        test_benchmark_results,
        config.MULTICLASS_COMPARISON_PATH,
    )

    # 5. Select best model based on macro-F1 per rules.md
    best_row = df_comparison.iloc[0]
    best_model_name = str(best_row["model"])
    best_estimator = fitted_models[best_model_name]
    logger.info(
        "Selected best multi-class model: %s (Test Macro-F1: %.4f, Macro-Recall: %.4f)",
        best_model_name,
        best_row["f1_macro"],
        best_row["recall_macro"],
    )

    # 6. Wrap best estimator with FeatureSelector in a Pipeline
    best_pipeline = Pipeline([
        ("selector", features.FeatureSelector(selected_features=selected_features, all_feature_names=all_feature_names)),
        ("classifier", best_estimator),
    ])

    # Save artifact to models/best_multiclass.joblib
    logger.info("Saving best multi-class model pipeline to: %s", config.BEST_MULTICLASS_MODEL_PATH)
    joblib.dump(best_pipeline, config.BEST_MULTICLASS_MODEL_PATH)

    # 7. Generate and save evaluation figures and per-class report for best model
    best_preds = best_pipeline.predict(X_test_full)

    evaluate.compute_per_class_report(
        y_test,
        best_preds,
        classes=config.ATTACK_CLASSES,
        output_path=config.PER_CLASS_REPORT_PATH,
    )

    evaluate.plot_multiclass_confusion_matrix(
        y_test,
        best_preds,
        classes=config.ATTACK_CLASSES,
        model_name=best_model_name,
        save_path=config.FIGURES_DIR / "cm_multiclass_best.png",
    )

    return best_pipeline, df_cv, df_comparison


if __name__ == "__main__":
    import argparse
    from src import data_loader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Model training and evaluation pipeline")
    parser.add_argument("--task", choices=["binary", "multiclass", "all"], default="multiclass", help="Task to train")
    args = parser.parse_args()

    train_data, test_data = data_loader.load_processed()

    if args.task in ["binary", "all"]:
        logger.info("Executing binary classification pipeline...")
        train_and_select_best_binary(train_data, test_data)

    if args.task in ["multiclass", "all"]:
        logger.info("Executing multi-class classification pipeline...")
        train_and_select_best_multiclass(train_data, test_data)

