# Network Intrusion Detection Using Machine Learning

**Department of Computer Science & Engineering**  
**Final-Year B.Tech Capstone Project (Group 16)**  
**Future Institute of Engineering and Management**  
**Project Guide:** Prof. Satyam Raha  

---

## 1. Project Overview

This project implements an end-to-end, leak-free Machine Learning system for automated network intrusion detection using the **UNSW-NB15** dataset. It provides an offline training and evaluation pipeline comparing seven machine learning algorithms, along with a production-ready **Streamlit web application** for real-time batch traffic inference.

### Key Capabilities
- **Two-Stage Cascade Inference:** 
  1. High-precision binary classifier rapidly separates `Normal` traffic from `Attack` traffic.
  2. For flagged attacks, a specialized multi-class model attributes the incident to one of 9 distinct attack categories (`Exploits`, `Fuzzers`, `Generic`, `Reconnaissance`, `DoS`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`). Benign packets are never mislabeled with false attack families.
- **Leak-Free ML Engineering:** Preprocessors and feature selectors fit exclusively on training data splits; zero synthetic target leakage.
- **Model Comparison Suite:** Standardized benchmarking across Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, XGBoost, and Support Vector Machines.
- **Interactive Web Dashboard:** Clean, dual-tab Streamlit interface supporting CSV uploads, real-time prediction charts, filtered inspection, report downloads, and offline model evaluation analytics.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.12 and 3.14)
- 8 GB RAM or higher
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "final year project"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # On Windows (Command Prompt):
   python -m venv .venv
   .venv\Scripts\activate.bat

   # On Linux/macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install pinned dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "from src import config; config.ensure_directories(); print('Environment setup successfully!')"
   ```

---

## 3. Dataset Placement

The project uses the official **UNSW-NB15** dataset. Place the raw CSV files directly in `data/raw/`:

```
data/raw/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

*Note: If downloading from an academic repository, ensure the file names match the paths above exactly as defined in `src/config.py`.*

---

## 4. How to Train (Exact Commands in Order)

To execute the entire offline pipeline from raw data to serialized production models and evaluation reports, run the following commands sequentially:

### Step 1: Clean Raw Data and Generate Sample
Loads raw CSVs, strips whitespaces, removes duplicates without the synthetic `id` column, logs cleaned records, saves parquet files to `data/processed/`, and generates `data/sample/sample_input.csv`:
```bash
python -m src.data_loader
```

### Step 2: Fit Preprocessing Transformers
Fits `StandardScaler` on 39 numerical features and `OneHotEncoder(handle_unknown="ignore")` on categorical features (`proto`, `service`, `state`) strictly on training data:
```bash
python -m src.preprocess
```

### Step 3: Run Feature Selection
Computes model-based Gini feature importances on the training split and persists the top 30 features to `models/selected_features.json`:
```bash
python -m src.features
```

### Step 4: Train and Evaluate Models
Execute model training, stratified cross-validation, hyperparameter tuning, and model persistence:

- **Train All Models (Binary and Multi-Class):**
  ```bash
  python -m src.train --task all
  ```

- *Optional Individual Runs:*
  - Train binary models only:
    ```bash
    python -m src.train --task binary
    ```
  - Train multi-class models only:
    ```bash
    python -m src.train --task multiclass
    ```

### Step 5: Regenerate Evaluation Summary Plots
Computes comparative bar charts and per-class metrics visualizations into `reports/figures/`:
```bash
python -m src.evaluate
```

---

## 5. How to Run the Web Application

Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
```

1. Open your browser at `http://localhost:8501`.
2. **Detect Tab:**
   - Upload any network traffic CSV containing the required features (or click **Download sample CSV** to test immediately with `data/sample/sample_input.csv`).
   - Click **Run detection** to execute the two-stage cascade classifier.
   - Inspect total row counts, normal vs attack breakdowns, categorical bar charts, and filtered record tables.
   - Click **Download results** to export the predictions with appended `prediction` and `category` columns.
3. **Model Results Tab:**
   - View test-set benchmark tables for binary and multi-class classification without running offline compute.
   - Inspect saved confusion matrices, ROC curves, per-class detection performance charts, feature importance rankings, and SHAP explainability beeswarm plots.

---

## 6. How to Run Unit Tests

Execute the comprehensive unit test suite covering preprocessor integrity, schema consistency, unseen category resilience, and cascade inference:
```bash
pytest -v
```

All 12 test cases will execute and validate against model artifacts in `models/` and sample data in `data/sample/`.

---

## 7. Folder Guide

```
final-year-project/
├── app/
│   └── app.py                     # Streamlit web application dashboard (two-tab UI)
├── data/
│   ├── raw/                       # Original raw UNSW-NB15 CSV files (git-ignored)
│   ├── processed/                 # Cleaned train and test parquet splits
│   └── sample/
│       └── sample_input.csv       # 200-row stratified sample for testing and app demo
├── models/
│   ├── preprocessor.joblib        # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── feature_columns.json       # Schema contract of expected 42 raw feature columns
│   ├── selected_features.json     # 30 winning features selected by Random Forest
│   ├── best_binary.joblib         # Winning binary classifier pipeline (Decision Tree)
│   └── best_multiclass.joblib     # Winning multi-class classifier pipeline (Random Forest Tuned)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis, distributions, and correlations
│   ├── 02_binary.ipynb            # Stratified CV benchmarking and binary tuning
│   ├── 03_multiclass.ipynb        # Multi-class benchmarking and tuning on attack families
│   └── 04_comparison.ipynb        # Final test evaluation, confusion matrices, and ROC curves
├── reports/
│   ├── binary_comparison.csv      # Held-out test metrics across all binary models
│   ├── multiclass_comparison.csv  # Held-out test metrics across all multi-class models
│   ├── per_class_report.csv       # Precision, Recall, and F1 across all 10 attack classes
│   ├── summary.md                 # Complete technical findings and college capstone report
│   └── figures/                   # Generated evaluation plots (PNG)
├── src/
│   ├── __init__.py                # Package initializer
│   ├── config.py                  # Single source of truth (paths, schemas, targets, seed)
│   ├── data_loader.py             # Data loading, cleaning, duplicate removal, sample generation
│   ├── preprocess.py              # Preprocessing transformer builder and schema serializer
│   ├── features.py                # Feature selection using Gini importance and mutual information
│   ├── models.py                  # Model factory, hyperparameter grids, SampledSVM, XGB wrapper
│   ├── train.py                   # Orchestration of cross-validation, tuning, and pipeline export
│   ├── evaluate.py                # Evaluation metrics, confusion matrices, ROC plots, bar charts
│   └── predict.py                 # Cascading inference engine used by app and testing suites
├── tests/
│   ├── test_preprocess.py         # Unit tests for preprocessing, schema, and unseen values
│   └── test_predict.py            # Unit tests for cascade prediction, empty inputs, column checks
├── requirements.txt               # Pinned dependency requirements
├── prd.md                         # Product Requirements Document
├── architecture.md                # System Architecture & Technical Specifications
├── rules.md                       # Project coding rules and engineering constraints
└── README.md                      # Project documentation and reproduction guide
```

---

## 8. Results Summary

All models were evaluated on the held-out UNSW-NB15 test set consisting of **55,945** unique network flow records.

### 8.1 Binary Classification Benchmark (Held-Out Test Set)

| Model | Accuracy | Precision | Attack Recall | Macro-F1 | Weighted-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Selected)** | **0.8379** | **0.7282** | **0.9299** | **0.8357** | **0.8399** | **0.9207** |
| Decision Tree (Tuned) | 0.8347 | 0.7202 | 0.9397 | 0.8329 | 0.8368 | 0.9367 |
| Random Forest (Tuned) | 0.8315 | 0.7045 | 0.9754 | 0.8306 | 0.8334 | 0.9676 |
| XGBoost (Tuned) | 0.8295 | 0.7019 | 0.9752 | 0.8286 | 0.8313 | 0.9719 |
| XGBoost | 0.8280 | 0.6991 | 0.9784 | 0.8272 | 0.8298 | 0.9726 |
| Random Forest | 0.8269 | 0.6978 | 0.9779 | 0.8261 | 0.8287 | 0.9695 |
| KNN | 0.7964 | 0.6755 | 0.9158 | 0.7949 | 0.7988 | 0.9020 |
| SVM (Sampled) | 0.7403 | 0.6047 | 0.9580 | 0.7403 | 0.7401 | 0.8883 |
| Logistic Regression | 0.7396 | 0.6033 | 0.9637 | 0.7396 | 0.7391 | 0.8882 |
| Naive Bayes | 0.6605 | 0.6068 | 0.3585 | 0.6025 | 0.6363 | 0.7530 |

### 8.2 Multi-Class Classification Benchmark (Held-Out Test Set)

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Tuned (Selected)** | **0.6540** | **0.5042** | **0.6736** | **0.5230** | **0.7098** |
| Random Forest | 0.6448 | 0.4891 | 0.6748 | 0.5086 | 0.7027 |
| XGBoost (Tuned) | 0.6401 | 0.4868 | 0.6973 | 0.5067 | 0.7002 |
| Decision Tree (Tuned) | 0.6471 | 0.4817 | 0.6528 | 0.5046 | 0.7016 |
| XGBoost | 0.6375 | 0.4854 | 0.6970 | 0.5037 | 0.6986 |
| Decision Tree | 0.6119 | 0.4547 | 0.6479 | 0.4801 | 0.6718 |
| KNN | 0.6748 | 0.4562 | 0.4194 | 0.4074 | 0.7060 |
| Logistic Regression | 0.5616 | 0.3255 | 0.4644 | 0.3092 | 0.6310 |
| SVM (Sampled) | 0.5919 | 0.3672 | 0.3241 | 0.2767 | 0.6261 |
| Naive Bayes | 0.4445 | 0.2930 | 0.3633 | 0.2270 | 0.5559 |

### 8.3 Winning Model Per-Class Breakdown (Tuned Random Forest)

| Attack Class | Support | Precision | Recall | F1-Score | Remarks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Normal` | 34,206 | 0.9974 | 0.5969 | 0.7468 | High-precision baseline |
| `Generic` | 3,657 | 0.9683 | 0.8769 | 0.9204 | Distinctive cryptographic packet profile |
| `Exploits` | 7,609 | 0.7929 | 0.7680 | 0.7803 | Strong CVE behavioral match |
| `Reconnaissance`| 2,703 | 0.7605 | 0.8398 | 0.7982 | High sensitivity to port and IP sweep patterns |
| `Worms` | 44 | 0.4643 | 0.8864 | 0.6094 | 39 of 44 rare worm events captured |
| `Shellcode` | 378 | 0.1996 | 0.8228 | 0.3213 | 311 of 378 exploit payloads flagged |
| `DoS` | 1,718 | 0.4481 | 0.3743 | 0.4079 | Confused with burst traffic and Fuzzers |
| `Fuzzers` | 4,838 | 0.2399 | 0.7313 | 0.3613 | High overlap with complex HTTP/web apps |
| `Backdoor` | 346 | 0.1131 | 0.5347 | 0.1867 | Low-frequency beaconing evasion |
| `Analysis` | 446 | 0.0580 | 0.3049 | 0.0974 | Hardest class due to similarity to benign web queries |

### 8.4 Model Explainability & Feature Importance

- **Binary Classification (Decision Tree):** `sttl` accounts for **72.64%** of Gini feature importance, reflecting distinctive time-to-live stack properties in attack traffic generators compared to benign OS traffic. TreeSHAP beeswarm attribution confirms higher TTL values systematically shift predictions toward `Attack`.
- **Multi-Class Classification (Tuned Random Forest):** Feature importance is distributed across packet dynamics: `sbytes` (15.70%), `smean` (11.73%), `sttl` (6.83%), `ct_srv_dst` (5.56%), `ct_dst_src_ltm` (5.23%), and `dbytes` (5.15%), capturing the volume and connection rates needed to separate varied attack categories.

