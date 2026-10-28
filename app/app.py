"""Streamlit Web Application for Network Intrusion Detection.

Provides a clean, two-tab interface following rules.md and prd.md:
- Detect: File uploader, validation, cascade predictions, summary metrics,
  category distribution chart, filterable results table, and CSV download.
- Model results: Displays precomputed comparison tables and evaluation figures
  from reports/ with zero computation.
"""

from pathlib import Path
import sys
from typing import Any, List, Optional, Tuple
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, predict

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit per prd.md / architecture.md


@st.cache_resource
def load_cached_artifacts() -> Tuple[Optional[Any], Optional[List[str]], Optional[Any], Optional[Any]]:
    """Load serialized model and preprocessing artifacts once and cache them.

    Inputs: None
    Outputs: Tuple of (preprocessor, feature_columns, best_binary, best_multiclass)
    Purpose: Ensures fast startup and responsive predictions without reloading.
    """
    try:
        return predict.load_artifacts()
    except FileNotFoundError:
        return None, None, None, None


def main() -> None:
    """Main application routine defining UI layout, validation, and rendering."""
    st.set_page_config(
        page_title="Network Intrusion Detection",
        layout="wide",
    )

    tab_detect, tab_results = st.tabs(["Detect", "Model results"])

    preprocessor, feature_cols, best_binary, best_multiclass = load_cached_artifacts()

    # -------------------------------------------------------------
    # TAB 1: DETECT
    # -------------------------------------------------------------
    with tab_detect:
        st.title("Network Intrusion Detection System")
        st.write("Upload a CSV file of network traffic records to detect normal and malicious activities.")

        # Check artifact readiness
        if (
            preprocessor is None
            or feature_cols is None
            or best_binary is None
            or best_multiclass is None
        ):
            st.error("Models not found. Run the training pipeline first: python -m src.train --task all")
            return

        # 1. File Uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # 2. Sample CSV download button for immediate trial
        if config.SAMPLE_INPUT_PATH.exists():
            with open(config.SAMPLE_INPUT_PATH, "rb") as f_sample:
                sample_bytes = f_sample.read()
            st.download_button(
                label="Download sample CSV",
                data=sample_bytes,
                file_name="sample_input.csv",
                mime="text/csv",
                help="Download sample network traffic records to test the detection system.",
            )



        # File validation state
        is_valid_file = False
        upload_df: Optional[pd.DataFrame] = None

        if uploaded_file is not None:
            # Check file size limit
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error("The uploaded file exceeds the 50 MB size limit.")
            elif uploaded_file.size == 0:
                st.error("The uploaded file is empty.")
            else:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    if upload_df.empty or len(upload_df) == 0:
                        st.error("The uploaded file contains no data rows.")
                    else:
                        missing_cols = [col for col in feature_cols if col not in upload_df.columns]
                        if missing_cols:
                            st.error(
                                f"The file is missing these columns: {', '.join(sorted(missing_cols))}."
                            )
                        else:
                            is_valid_file = True
                except Exception:
                    st.error("The uploaded file could not be parsed as a CSV.")
        else:
            st.info("Upload a CSV file to begin.")

        # Track file identity to reset results when a new file is uploaded
        current_file_id = uploaded_file.name if uploaded_file is not None else None
        if st.session_state.get("last_uploaded_file_id") != current_file_id:
            st.session_state["results_df"] = None
            st.session_state["last_uploaded_file_id"] = current_file_id

        # 3. Run detection button (disabled until valid file is uploaded)
        run_detection = st.button("Run detection", disabled=not is_valid_file)

        if run_detection and is_valid_file and upload_df is not None:
            with st.spinner("Processing network traffic records..."):
                try:
                    res_df = predict.predict_dataframe(
                        upload_df,
                        preprocessor=preprocessor,
                        feature_cols=feature_cols,
                        best_binary=best_binary,
                        best_multiclass=best_multiclass,
                    )
                    st.session_state["results_df"] = res_df
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Render detection results if available in session state
        results_df: Optional[pd.DataFrame] = st.session_state.get("results_df")

        if results_df is not None:
            total_rows = len(results_df)
            normal_count = int((results_df["prediction"] == "Normal").sum())
            attack_count = int((results_df["prediction"] == "Attack").sum())

            # 4. Summary: three numbers
            col1, col2, col3 = st.columns(3)
            col1.metric("Total rows", f"{total_rows:,}")
            col2.metric("Normal", f"{normal_count:,}")
            col3.metric("Attack", f"{attack_count:,}")

            # 5. One bar chart of predicted categories
            st.subheader("Predicted category breakdown")
            category_counts = results_df["category"].value_counts()
            st.bar_chart(category_counts)

            # 6. Results table with filter
            st.subheader("Detection results")
            filter_choice = st.radio(
                "Filter table",
                options=["All", "Normal", "Attack"],
                horizontal=True,
            )

            if filter_choice == "Normal":
                filtered_df = results_df[results_df["prediction"] == "Normal"]
            elif filter_choice == "Attack":
                filtered_df = results_df[results_df["prediction"] == "Attack"]
            else:
                filtered_df = results_df

            st.dataframe(filtered_df, width="stretch")

            # 7. Download results button
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results",
                data=csv_data,
                file_name="intrusion_detection_results.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------
    # TAB 2: MODEL RESULTS (Pure Viewing Dashboard - Zero Computation)
    # -------------------------------------------------------------
    with tab_results:
        st.title("Model Evaluation and Comparison")
        st.write("Offline benchmark reports and performance visualizations evaluated on the held-out test set.")

        # Section 1: Binary Classification Comparison
        st.subheader("Binary classification comparison")
        if config.BINARY_COMPARISON_PATH.exists():
            df_bin = pd.read_csv(config.BINARY_COMPARISON_PATH)
            st.dataframe(df_bin, width="stretch")
        else:
            st.write("Binary comparison table not found.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bin_bar_path = config.FIGURES_DIR / "binary_comparison_bar.png"
            if bin_bar_path.exists():
                st.image(str(bin_bar_path), caption="Binary model comparison (Test set)", width="stretch")
        with col_b2:
            roc_path = config.FIGURES_DIR / "roc_binary_comparison.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC curves comparison (Test set)", width="stretch")

        cm_bin_path = config.FIGURES_DIR / "cm_binary_best.png"
        if cm_bin_path.exists():
            st.image(str(cm_bin_path), caption="Confusion matrix: Best binary model (Decision Tree)", width=500)

        # Section 2: Multi-Class Classification Comparison
        st.subheader("Multi-class classification comparison")
        if config.MULTICLASS_COMPARISON_PATH.exists():
            df_multi = pd.read_csv(config.MULTICLASS_COMPARISON_PATH)
            st.dataframe(df_multi, width="stretch")
        else:
            st.write("Multi-class comparison table not found.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            multi_bar_path = config.FIGURES_DIR / "multiclass_comparison_bar.png"
            if multi_bar_path.exists():
                st.image(str(multi_bar_path), caption="Multi-class model comparison (Test set)", width="stretch")
        with col_m2:
            cm_multi_path = config.FIGURES_DIR / "cm_multiclass_best.png"
            if cm_multi_path.exists():
                st.image(str(cm_multi_path), caption="Normalized confusion matrix: Best multi-class model (Random Forest Tuned)", width="stretch")

        # Section 3: Per-Class Detailed Breakdown
        st.subheader("Per-class performance breakdown")
        if config.PER_CLASS_REPORT_PATH.exists():
            df_per = pd.read_csv(config.PER_CLASS_REPORT_PATH)
            st.dataframe(df_per, width="stretch")
        else:
            st.write("Per-class report not found.")

        per_bar_path = config.FIGURES_DIR / "per_class_metrics_bar.png"
        if per_bar_path.exists():
            st.image(str(per_bar_path), caption="Per-class recall and F1-score (Winning multi-class model)", width="stretch")

        # Section 4: Feature Importance and Model Explainability
        st.subheader("Feature importance and explainability")
        feat_imp_path = config.FIGURES_DIR / "feature_importance_best.png"
        if feat_imp_path.exists():
            st.image(
                str(feat_imp_path),
                caption="Top feature importances: Best binary model (Decision Tree) and best multi-class model (Random Forest Tuned)",
                width="stretch",
            )

        shap_path = config.FIGURES_DIR / "shap_summary_binary.png"
        if shap_path.exists():
            st.image(
                str(shap_path),
                caption="SHAP summary plot: Feature attribution and directionality on binary attack detection (Decision Tree)",
                width="stretch",
            )


if __name__ == "__main__":
    main()

