# PRD: Network Intrusion Detection Using Machine Learning

Group 16, Dept. of CSE, Future Institute of Engineering and Management
Guide: Prof. Satyam Raha

## 1. Overview

A machine-learning system that classifies network traffic records as normal or malicious (binary classification) and identifies the specific attack category (multi-class classification). The project compares several ML algorithms, selects the best performers, and exposes them through a simple web app where a user uploads a CSV of traffic records and gets predictions.

## 2. Goals

1. Build a clean, reproducible ML pipeline on a public intrusion-detection dataset.
2. Perform binary classification: Normal vs Attack.
3. Perform multi-class classification: Normal vs specific attack category.
4. Compare at least six algorithms and report accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC (binary).
5. Select the best model for each task using macro-F1 and per-class recall, not accuracy alone.
6. Provide a simple web interface for uploading data and viewing predictions.

## 3. Non-Goals

- Real-time packet capture or live network monitoring (listed as future work only).
- Deep learning (future work only).
- User accounts, authentication, databases, or deployment infrastructure.
- Mobile or desktop apps.

## 4. Dataset

Default: **UNSW-NB15** (official training-set and testing-set CSVs).

- Binary target: `label` (0 = Normal, 1 = Attack)
- Multi-class target: `attack_cat` (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)
- Categorical features: `proto`, `service`, `state`
- Columns to drop before training: `id`, plus both target columns from the feature set

The dataset name, file paths, and column names live in one config file so the dataset can be swapped (e.g. CIC-IDS2017) with minimal code change.

## 5. Users

- **Primary:** a student, examiner, or security-aware user who wants to upload traffic records and see which are malicious.
- **Secondary:** the project team, who use the notebooks and reports to analyse model performance.

## 6. Functional Requirements

### 6.1 ML pipeline
| ID | Requirement |
|---|---|
| F1 | Load dataset, remove duplicates, handle missing and infinite values. |
| F2 | Encode categorical features and scale numeric features using a fitted sklearn preprocessor. |
| F3 | Fit all preprocessors on training data only (no leakage). |
| F4 | Perform feature selection (importance-based or mutual information) and record the chosen features. |
| F5 | Handle class imbalance with class weights or SMOTE applied to training data only. |
| F6 | Train binary models: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost. SVM trained on a stratified sample. |
| F7 | Train the same set of models for multi-class classification. |
| F8 | Tune the top 2 or 3 models per task with cross-validated search. |
| F9 | Evaluate every model with accuracy, precision, recall, F1 (macro and weighted), confusion matrix; add ROC-AUC for binary. |
| F10 | Save a comparison table (CSV) and plots (PNG) to `reports/`. |
| F11 | Save the best binary model, best multi-class model, and preprocessor to `models/` with joblib. |

### 6.2 Web app
| ID | Requirement |
|---|---|
| A1 | User uploads a CSV file of traffic records. |
| A2 | App validates the file: required columns present, file not empty, size within limit. Clear error message if invalid. |
| A3 | App runs the binary model on every row. |
| A4 | For rows predicted as Attack, app runs the multi-class model and assigns the most likely attack category. |
| A5 | App shows: total rows, number of normal rows, number of attack rows. |
| A6 | App shows a results table with the prediction per row and a filter for Normal / Attack. |
| A7 | App shows one chart: count of rows per predicted category. |
| A8 | User can download the results as CSV. |
| A9 | App has a second tab, "Model results", showing the comparison table and key plots from `reports/`. |
| A10 | App offers a sample CSV for download so users can try it immediately. |

## 7. UI Requirements

The UI must be simple and easy to use. Two tabs only: **Detect** and **Model results**.

**Detect tab, top to bottom:**
1. Page title and one line of plain instruction.
2. File uploader.
3. "Run detection" button (disabled until a valid file is uploaded).
4. Summary: three numbers (total, normal, attack).
5. Chart of predicted categories.
6. Results table with filter.
7. Download button.

**Explicit exclusions:**
- No badges, pills, tags, or status chips.
- No emojis or decorative icons.
- No gradients, animations, custom fonts, or marketing copy.
- No sidebar unless it is needed, no extra pages, no settings panels.
- Use Streamlit's default theme with plain text labels.

Error and empty states use plain sentences, for example "Upload a CSV file to begin." and "The file is missing these columns: proto, service."

## 8. Evaluation and Success Criteria

- All six algorithms trained and compared for both tasks.
- Binary best model: macro-F1 reported on the held-out test set. Target is a clear improvement over the Logistic Regression baseline.
- Multi-class: per-class precision, recall, F1 reported; rare classes (Worms, Shellcode, Backdoor, Analysis) discussed explicitly.
- Cross-validation scores reported for the final models.
- App runs end to end on the sample CSV with no errors.
- A fresh clone can be set up from the README and run in under 10 minutes.

## 9. Constraints

- Python 3.10+, runs on a laptop with 8 GB RAM.
- Open-source libraries only.
- Fixed random seed (42) everywhere for reproducibility.

## 10. Deliverables

1. Source code and notebooks.
2. Saved models and preprocessor.
3. Reports folder: comparison tables and plots.
4. Streamlit web app.
5. README with setup and run instructions.
6. Final project report and presentation (written by the team, using outputs from `reports/`).

## 11. Future Work

Real-time traffic monitoring, automated alerts, deep-learning models, deployment in a practical network environment.
