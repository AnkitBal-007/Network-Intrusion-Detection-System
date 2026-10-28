# Architecture: Network Intrusion Detection Using ML

## 1. System Overview

Two parts:

1. **Offline training pipeline** (Python scripts and notebooks): reads the dataset, preprocesses, trains and evaluates models, saves artifacts.
2. **Web app** (Streamlit): loads the saved artifacts and serves predictions on uploaded CSV files.

The app never trains anything. It only loads files from `models/` and `reports/`.

```
                OFFLINE (run once)                              ONLINE (run app)

 Raw CSVs --> load --> clean --> split --> preprocess            User uploads CSV
                                              |                        |
                                   feature selection                validate
                                              |                        |
                        +---------------------+------------+      preprocess (saved)
                        |                                  |           |
               binary models (6)               multi-class models (6)  binary model
                        |                                  |           |
                 tune + evaluate                    tune + evaluate    Attack rows only
                        |                                  |           |
                        +----------> reports/ <------------+     multi-class model
                        |                                              |
                  models/*.joblib  ------------------------------>  results table,
                                                                    chart, download
```

## 2. Folder Structure

```
nids-ml/
├── data/
│   ├── raw/                  # original dataset CSVs (not committed)
│   ├── processed/            # cleaned splits (parquet or csv)
│   └── sample/sample_input.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_binary.ipynb
│   ├── 03_multiclass.ipynb
│   └── 04_comparison.ipynb
├── src/
│   ├── config.py             # paths, column names, seed, class lists
│   ├── data_loader.py        # load + clean
│   ├── preprocess.py         # build/fit/save ColumnTransformer
│   ├── features.py           # feature selection
│   ├── models.py             # model factory and param grids
│   ├── train.py              # train + tune + save (binary and multiclass)
│   ├── evaluate.py           # metrics, plots, comparison tables
│   └── predict.py            # load artifacts, run cascade prediction
├── app/
│   └── app.py                # Streamlit UI
├── models/                   # joblib artifacts
├── reports/
│   ├── binary_comparison.csv
│   ├── multiclass_comparison.csv
│   ├── per_class_report.csv
│   └── figures/              # confusion matrices, ROC, bar charts
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── requirements.txt
├── README.md
├── prd.md
├── architecture.md
└── rules.md
```

Notebooks are for exploration and presenting results. All reusable logic lives in `src/` and notebooks import from it.

## 3. Components

### 3.1 `config.py`
Single source of truth: dataset paths, `BINARY_TARGET = "label"`, `MULTI_TARGET = "attack_cat"`, `DROP_COLS = ["id"]`, `CATEGORICAL_COLS = ["proto", "service", "state"]`, `RANDOM_STATE = 42`, `TEST_SIZE`, artifact paths.

### 3.2 `data_loader.py`
- `load_raw()` returns train and test DataFrames.
- `clean(df)` drops duplicates, replaces inf with NaN, fills or drops NaN, strips whitespace from string columns.

### 3.3 `preprocess.py`
- `build_preprocessor(numeric_cols, categorical_cols)` returns a `ColumnTransformer`: `StandardScaler` for numeric, `OneHotEncoder(handle_unknown="ignore")` for categorical.
- Fit on training data only. Saved to `models/preprocessor.joblib`.
- The list of expected input columns is saved to `models/feature_columns.json` for app validation.

### 3.4 `features.py`
- Feature selection by Random Forest importance or mutual information, computed on training data only.
- Selected features saved to `models/selected_features.json`.
- Where selection is used, it is part of the saved pipeline so the app applies it identically.

### 3.5 `models.py`
Factory returning the six algorithms with sensible defaults and `class_weight="balanced"` where supported. Parameter grids for tuning. SVM wrapper trains on a stratified sample.

### 3.6 `train.py`
- Trains every model for a given task (binary or multiclass) and records cross-validated scores.
- Tunes the top 2 or 3 by macro-F1.
- Saves the best model per task as `models/best_binary.joblib` and `models/best_multiclass.joblib`.
- Imbalance handling (SMOTE) is placed inside an `imblearn` pipeline so it is applied to training folds only.

### 3.7 `evaluate.py`
- Computes accuracy, precision, recall, F1 (macro and weighted) on the held-out test set.
- Binary: confusion matrix and ROC-AUC with ROC curve.
- Multi-class: per-class report and normalised confusion matrix.
- Writes CSV tables and PNG figures to `reports/`.

### 3.8 `predict.py`
Core inference logic, shared by the app and tests.

```
predict_dataframe(df) -> DataFrame
  1. validate columns against feature_columns.json
  2. drop id / label / attack_cat if present
  3. X = preprocessor.transform(df)
  4. binary_pred = best_binary.predict(X)
  5. for rows where binary_pred == 1:
        probs = best_multiclass.predict_proba(X[rows])
        category = argmax over non-Normal classes
  6. return df with columns: prediction (Normal / Attack), category
```

Rows predicted Normal get category "Normal".

### 3.9 `app/app.py`
Streamlit app with two tabs.

**Detect:** uploader, run button, summary numbers, category chart, filterable results table, download button, sample CSV download.

**Model results:** reads `reports/*.csv` and `reports/figures/*.png` and displays them. No computation.

Models and preprocessor are loaded once with `st.cache_resource`.

## 4. Data Flow and Contracts

| Stage | Input | Output |
|---|---|---|
| Load and clean | raw CSV | clean DataFrame |
| Split | clean DataFrame | train and test, stratified on the target |
| Preprocess | train and test features | numeric matrices (fit on train only) |
| Train | train matrices and labels | fitted models, CV scores |
| Evaluate | test matrices and labels | metrics tables, figures |
| Predict | uploaded DataFrame | DataFrame with `prediction` and `category` |

**Upload contract:** the CSV must contain every column listed in `models/feature_columns.json`. Extra columns are ignored. `id`, `label`, `attack_cat` are optional and ignored for prediction.

## 5. Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, imbalanced-learn |
| Plots | matplotlib, seaborn |
| Persistence | joblib |
| Web app | Streamlit |
| Tests | pytest |

## 6. Key Design Decisions

1. **Cascade inference.** Binary model decides Normal or Attack first; multi-class model only labels flagged rows. This matches the synopsis ("binary as first level").
2. **Single preprocessor artifact.** One fitted preprocessor is shared by both tasks so the app has one consistent input path.
3. **Logic in `src/`, UI in `app/`.** The app contains no ML logic, which keeps it simple and testable.
4. **Config-driven dataset.** Swapping datasets changes `config.py`, not the pipeline.
5. **No retraining in the app.** Fast startup and predictable behaviour for demos.

## 7. Error Handling

- Missing columns: show the list of missing names and stop.
- Empty or unreadable file: show a plain message.
- Uploaded file above the size limit (default 50 MB): show a plain message.
- Unseen categorical values: handled by `handle_unknown="ignore"`.
- Missing artifacts in `models/`: app shows "Models not found. Run the training pipeline first." with the command to run.
