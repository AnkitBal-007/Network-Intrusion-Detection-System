"""Model evaluation and metrics reporting module.

Computes accuracy, precision, recall, macro/weighted F1, ROC-AUC, confusion matrices,
and ROC curves. Exports comparison tables and figure plots to reports/.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from src import config

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics for binary classification.

    Inputs:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack)
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for class 1 (optional for ROC-AUC)
    Outputs:
        Dictionary of computed metrics (accuracy, precision, recall, f1_macro, f1_weighted, roc_auc)
    Purpose:
        Comply with rules.md prohibiting reporting accuracy alone.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1_binary = float(f1_score(y_t, y_p, zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    roc_auc = 0.0
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            roc_auc = float(roc_auc_score(y_t, prob_pos))
        except Exception as e:
            logger.warning("Could not calculate ROC-AUC: %s", e)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_attack": round(f1_binary, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(roc_auc, 4),
    }


def plot_binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save a normalized confusion matrix figure for binary classification.

    Inputs:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        model_name: Name of the algorithm evaluated
        save_path: Target PNG file path (defaults to reports/figures/)
    Outputs:
        None (saves figure to disk)
    Purpose:
        Provide visual error analysis showing false positives vs false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Normal (0)", "Attack (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    fmt = ".2%"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_norm[i, j]:{fmt}})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    if save_path is None:
        config.ensure_directories()
        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = config.FIGURES_DIR / f"cm_binary_{clean_name}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to: %s", save_path)


def plot_binary_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save the ROC curve figure for binary classification.

    Inputs:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities for Attack class (class 1)
        model_name: Name of the evaluated algorithm
        save_path: Target PNG file path
    Outputs:
        None (saves figure to disk)
    Purpose:
        Visual assessment of true positive vs false positive trade-offs.
    """
    if y_prob.ndim == 2:
        prob_pos = y_prob[:, 1]
    else:
        prob_pos = y_prob

    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc_score = roc_auc_score(y_true, prob_pos)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"{model_name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        config.ensure_directories()
        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = config.FIGURES_DIR / f"roc_binary_{clean_name}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve plot to: %s", save_path)


def save_comparison_table(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """Format and save benchmark comparison results to CSV.

    Inputs:
        results: List of result dictionaries containing model name and metrics
        output_path: Target CSV file path
    Outputs:
        Summary pandas DataFrame sorted by macro F1 descending
    Purpose:
        Produce standard reporting tables for app tab 2 and final reports.
    """
    df_results = pd.DataFrame(results)
    if "f1_macro" in df_results.columns:
        df_results = df_results.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

    config.ensure_directories()
    df_results.to_csv(output_path, index=False)
    logger.info("Saved comparison table (%d models) to: %s", len(df_results), output_path)
    return df_results


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics for multi-class classification.

    Inputs:
        y_true: Ground truth multi-class labels
        y_pred: Predicted multi-class labels
    Outputs:
        Dictionary containing accuracy, precision (macro/weighted), recall (macro/weighted),
        and F1 (macro/weighted) scores.
    Purpose:
        Ensure multi-class evaluations report macro-averaged scores per rules.md.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


def compute_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1 scores, saving to CSV.

    Inputs:
        y_true: Ground truth class labels
        y_pred: Predicted class labels
        classes: Ordered list of attack classes (default: sorted unique in y_true)
        output_path: Destination path (default: reports/per_class_report.csv)
    Outputs:
        DataFrame summarizing per-class performance metrics and sample support
    Purpose:
        Comply with rules.md requirement to report and analyze rare class performance.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if classes is None:
        classes = sorted(list(set(y_t) | set(y_p)))

    report_dict = classification_report(
        y_t,
        y_p,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for cls_name in classes:
        if cls_name in report_dict:
            metrics = report_dict[cls_name]
            rows.append({
                "class": cls_name,
                "precision": round(float(metrics["precision"]), 4),
                "recall": round(float(metrics["recall"]), 4),
                "f1_score": round(float(metrics["f1-score"]), 4),
                "support": int(metrics["support"]),
            })

    df_report = pd.DataFrame(rows)

    if output_path is None:
        config.ensure_directories()
        output_path = config.PER_CLASS_REPORT_PATH

    df_report.to_csv(output_path, index=False)
    logger.info("Saved per-class evaluation report to: %s", output_path)
    return df_report


def plot_multiclass_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    model_name: str = "Best Multi-Class Model",
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save a normalized 10-class confusion matrix heatmap.

    Inputs:
        y_true: Ground truth class labels
        y_pred: Predicted class labels
        classes: Class labels order (default: sorted unique)
        model_name: Name of evaluated algorithm
        save_path: Destination PNG path
    Outputs:
        None (saves figure to disk)
    Purpose:
        Visual identification of misclassification patterns across normal and attack classes.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if classes is None:
        classes = sorted(list(set(y_t) | set(y_p)))

    cm = confusion_matrix(y_t, y_p, labels=classes)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(f"Normalized Confusion Matrix: {model_name}", fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is None:
        config.ensure_directories()
        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = config.FIGURES_DIR / f"cm_multiclass_{clean_name}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved multi-class confusion matrix plot to: %s", save_path)


def plot_model_comparison_bars(
    binary_comparison_path: Optional[Path] = None,
    multiclass_comparison_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generate and save comparative bar charts for binary and multi-class benchmarks.

    Inputs:
        binary_comparison_path: Path to reports/binary_comparison.csv
        multiclass_comparison_path: Path to reports/multiclass_comparison.csv
        save_dir: Destination folder for PNG plots
    Outputs:
        Tuple of (binary_bar_path, multiclass_bar_path)
    Purpose:
        Provide clear visual comparison of models across tasks for reports and tab 2 of web app.
    """
    config.ensure_directories()
    target_dir = save_dir or config.FIGURES_DIR
    bin_path = binary_comparison_path or config.BINARY_COMPARISON_PATH
    multi_path = multiclass_comparison_path or config.MULTICLASS_COMPARISON_PATH

    df_bin = pd.read_csv(bin_path)
    df_multi = pd.read_csv(multi_path)

    # 1. Binary Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_bin))
    width = 0.35

    plt.bar(x - width/2, df_bin["f1_macro"], width, label="Macro-F1", color="#2b5c8f")
    plt.bar(x + width/2, df_bin["recall"], width, label="Attack Recall", color="#888888")

    plt.ylabel("Score")
    plt.title("Binary Classification Performance Across Candidate Algorithms (Test Set)")
    plt.xticks(x, df_bin["model"], rotation=35, ha="right")
    plt.ylim([0.2, 1.05])
    plt.legend(loc="lower left")

    for i, v in enumerate(df_bin["f1_macro"]):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(df_bin["recall"]):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    bin_bar_path = target_dir / "binary_comparison_bar.png"
    plt.savefig(bin_bar_path, dpi=150)
    plt.close()
    logger.info("Saved binary comparison bar chart to: %s", bin_bar_path)

    # 2. Multi-Class Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x_m = np.arange(len(df_multi))

    plt.bar(x_m - width/2, df_multi["f1_macro"], width, label="Macro-F1", color="#2b5c8f")
    plt.bar(x_m + width/2, df_multi["recall_macro"], width, label="Macro-Recall", color="#888888")

    plt.ylabel("Score")
    plt.title("Multi-Class Classification Performance Across Candidate Algorithms (Test Set)")
    plt.xticks(x_m, df_multi["model"], rotation=35, ha="right")
    plt.ylim([0.1, 0.85])
    plt.legend(loc="upper right")

    for i, v in enumerate(df_multi["f1_macro"]):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(df_multi["recall_macro"]):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    multi_bar_path = target_dir / "multiclass_comparison_bar.png"
    plt.savefig(multi_bar_path, dpi=150)
    plt.close()
    logger.info("Saved multi-class comparison bar chart to: %s", multi_bar_path)

    return bin_bar_path, multi_bar_path


def plot_per_class_summary(
    per_class_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot bar chart comparing Recall and F1-score across all 10 attack classes.

    Inputs:
        per_class_path: Path to reports/per_class_report.csv
        save_path: Output PNG figure path
    Outputs:
        Path to generated plot
    Purpose:
        Visual assessment of model capability on rare vs common attack categories.
    """
    config.ensure_directories()
    src_path = per_class_path or config.PER_CLASS_REPORT_PATH
    df_per = pd.read_csv(src_path)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_per))
    width = 0.35

    ax.bar(x - width/2, df_per["recall"], width, label="Recall", color="#2b5c8f")
    ax.bar(x + width/2, df_per["f1_score"], width, label="F1-Score", color="#888888")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance of Winning Multi-Class Model (Random Forest Tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_per["class"], rotation=30, ha="right")
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")

    for i, v in enumerate(df_per["recall"]):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out_path = save_path or (config.FIGURES_DIR / "per_class_metrics_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved per-class metrics bar chart to: %s", out_path)
    return out_path


def plot_feature_importance(
    bin_model_path: Optional[Path] = None,
    multi_model_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    top_n: int = 12,
) -> Path:
    """Plot horizontal bar charts comparing feature importance for best binary and multi-class models.

    Inputs:
        bin_model_path: Path to best binary model (default: config.BEST_BINARY_MODEL_PATH)
        multi_model_path: Path to best multi-class model (default: config.BEST_MULTICLASS_MODEL_PATH)
        save_path: Path to output PNG figure (default: config.FIGURES_DIR / 'feature_importance_best.png')
        top_n: Number of leading features to display per model
    Outputs:
        Path to saved PNG plot
    Purpose:
        Provide transparent model explainability for top-performing tree-based architectures.
    """
    config.ensure_directories()
    b_path = bin_model_path or config.BEST_BINARY_MODEL_PATH
    m_path = multi_model_path or config.BEST_MULTICLASS_MODEL_PATH
    out_path = save_path or (config.FIGURES_DIR / "feature_importance_best.png")

    bin_pipe = joblib.load(b_path)
    multi_pipe = joblib.load(m_path)

    with open(config.SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
        selected_features = selected_data["selected_features"] if isinstance(selected_data, dict) else selected_data

    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in selected_features]

    dt_importances = bin_pipe.named_steps["classifier"].feature_importances_
    rf_importances = multi_pipe.named_steps["classifier"].feature_importances_

    df_dt = pd.DataFrame({"feature": clean_feature_names, "importance": dt_importances})
    df_rf = pd.DataFrame({"feature": clean_feature_names, "importance": rf_importances})

    df_dt_top = df_dt.sort_values(by="importance", ascending=True).tail(top_n)
    df_rf_top = df_rf.sort_values(by="importance", ascending=True).tail(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(df_dt_top["feature"], df_dt_top["importance"], color="#2b5c8f")
    ax1.set_title("Decision Tree Feature Importance (Best Binary)")
    ax1.set_xlabel("Gini Importance")
    for i, v in enumerate(df_dt_top["importance"]):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax1.set_xlim([0, max(df_dt_top["importance"]) * 1.15])

    ax2.barh(df_rf_top["feature"], df_rf_top["importance"], color="#4b8b3b")
    ax2.set_title("Random Forest Feature Importance (Best Multi-Class)")
    ax2.set_xlabel("Mean Impurity Reduction")
    for i, v in enumerate(df_rf_top["importance"]):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax2.set_xlim([0, max(df_rf_top["importance"]) * 1.15])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved feature importance comparison bar chart to: %s", out_path)
    return out_path


def plot_shap_summary(
    bin_model_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    n_samples: int = 500,
) -> Path:
    """Generate SHAP summary beeswarm plot for the winning binary classifier.

    Inputs:
        bin_model_path: Path to best binary model pipeline
        save_path: Output PNG figure path
        n_samples: Sample size from test split for SHAP explanation
    Outputs:
        Path to saved PNG plot
    Purpose:
        Explain directional feature impact (e.g. how high vs low TTL values drive attack decisions).
    """
    config.ensure_directories()
    b_path = bin_model_path or config.BEST_BINARY_MODEL_PATH
    out_path = save_path or (config.FIGURES_DIR / "shap_summary_binary.png")

    from src import data_loader, preprocess

    bin_pipe = joblib.load(b_path)
    prep = preprocess.load_preprocessor()

    with open(config.SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
        selected_features = selected_data["selected_features"] if isinstance(selected_data, dict) else selected_data

    clean_features = [f.replace("num__", "").replace("cat__", "") for f in selected_features]

    _, test_df = data_loader.load_processed()
    sample_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=config.RANDOM_STATE)
    X_test_trans = prep.transform(sample_df[preprocess.load_feature_columns()])

    selector = bin_pipe.named_steps["selector"]
    X_test_sel = selector.transform(X_test_trans)

    explainer = shap.TreeExplainer(bin_pipe.named_steps["classifier"])
    shap_values = explainer.shap_values(X_test_sel)

    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_test_sel, feature_names=clean_features, show=False, max_display=12)
    plt.title("SHAP Feature Attribution: Decision Tree (Attack vs Normal)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved SHAP summary plot to: %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Generating evaluation figures from existing reports and models...")
    if config.BINARY_COMPARISON_PATH.exists() and config.MULTICLASS_COMPARISON_PATH.exists():
        plot_model_comparison_bars()
    if config.PER_CLASS_REPORT_PATH.exists():
        plot_per_class_summary()
    if config.BEST_BINARY_MODEL_PATH.exists() and config.BEST_MULTICLASS_MODEL_PATH.exists():
        plot_feature_importance()
        plot_shap_summary()
    logger.info("Evaluation figures generated successfully.")




