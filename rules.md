# Rules for the AI Agent

These rules apply to every task in this project. Read this file, `prd.md`, and `architecture.md` before writing any code.

## 1. Scope

1. Build only what `prd.md` describes. Do not add features, pages, or dependencies that are not in it.
2. Follow the folder structure and component boundaries in `architecture.md`. If you need to deviate, stop and ask first.
3. Work one phase at a time. When a phase is finished, summarise what was done, list the files changed, and wait for approval before starting the next phase.

## 2. Machine Learning Rules

1. Use `RANDOM_STATE = 42` everywhere randomness appears.
2. Split data before fitting anything. Fit scalers, encoders, feature selectors, and SMOTE on training data only.
3. Use stratified splits and stratified cross-validation.
4. Apply SMOTE or any resampling only inside a pipeline on training folds. Never on the test set.
5. Never report accuracy alone. Always report precision, recall, F1 (macro and weighted), and the confusion matrix. Add ROC-AUC for binary.
6. Select the best model by macro-F1 (multi-class) or F1 plus recall on the Attack class (binary), not accuracy.
7. Never tune hyperparameters on the test set.
8. Handle `inf` and `NaN` explicitly. Do not silently drop large parts of the data. Log how many rows were removed and why.
9. Do not hard-code column names in multiple places. Read them from `src/config.py`.
10. Save every trained artifact with joblib into `models/`. Never retrain inside the app.
11. If a result looks too good (for example 100% accuracy), check for leakage before reporting it.

## 3. Code Rules

1. Python 3.10+. Follow PEP 8. Use type hints on public functions.
2. Every function in `src/` has a short docstring stating inputs, outputs, and purpose.
3. Keep functions small and single-purpose. No logic copied between notebooks and `src/`; notebooks import from `src/`.
4. No hard-coded absolute paths. Use `pathlib` and paths from `config.py`.
5. No unused imports, dead code, or commented-out blocks.
6. Pin dependency versions in `requirements.txt`.
7. Use plain `logging` or print statements for progress. No external logging frameworks.
8. Do not commit the raw dataset, large model files, or virtual environments. Provide a `.gitignore`.
9. Write clear, simple code that students can explain in a viva. Prefer readable over clever.

## 4. UI Rules (Streamlit)

1. Keep the UI simple and easy to use. Two tabs only: **Detect** and **Model results**.
2. Use Streamlit's default theme. No custom CSS, no custom fonts, no gradients, no animations.
3. **No badges, pills, chips, tags, status labels, emojis, or decorative icons.**
4. No marketing text, taglines, hero sections, footers, or "about" sections.
5. Use short, plain labels such as "Upload CSV file", "Run detection", "Download results".
6. Show only these elements in the Detect tab: instruction line, file uploader, run button, three summary numbers, one chart, one results table with a filter, one download button.
7. Error and empty states are single plain sentences that say what is wrong and what to do.
8. Disable the run button until a valid file is uploaded.
9. Do not add settings panels, theme toggles, or sidebars unless a phase explicitly asks for one.
10. Load models once using `st.cache_resource`. The UI must never block on training.

## 5. Dependencies

1. Allowed: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, joblib, streamlit, pytest, jupyter.
2. Anything else requires approval before it is added.

## 6. Testing and Verification

1. After each phase, run the relevant scripts or tests and report the actual output. Do not claim something works without running it.
2. Provide `tests/test_preprocess.py` and `tests/test_predict.py` at minimum. Tests must run with `pytest` from the project root.
3. Before finishing the app phase, run the app on `data/sample/sample_input.csv` and confirm the full flow works.
4. If a command fails, show the error and fix the cause. Do not hide or skip failures.

## 7. Communication

1. Be concise. State what you did, what you found, and what is next.
2. If a requirement is ambiguous or the dataset differs from the assumption in `prd.md`, ask a single clear question instead of guessing.
3. Report real metric values from real runs. Never invent or estimate results.
4. Keep a running `README.md` that documents setup, how to train, how to run the app, and where outputs are saved.

## 8. Definition of Done (per phase)

- Code runs without errors from a clean environment.
- Outputs listed in the phase are present on disk.
- Rules in this file are followed.
- A short summary and the next-phase proposal are given to the user.
