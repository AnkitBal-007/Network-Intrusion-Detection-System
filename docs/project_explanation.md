# Project Guide: Network Intrusion Detection Using Machine Learning

**Department of Computer Science & Engineering**  
**Final-Year B.Tech Capstone Project (Group 16)**  
**Future Institute of Engineering and Management**  
**Project Guide:** Prof. Satyam Raha  

---

## 1. What Problem Are We Solving?

Every second, billions of packets travel across computer networks—streaming videos, transferring files, sending emails, and browsing websites. Hiding inside this legitimate traffic are **cyberattacks**:
* **Denial of Service (DoS):** Flooding a server to make it crash.
* **Exploits & Shellcode:** Injecting malicious code to take control of a computer.
* **Backdoors:** Creating hidden doorways for hackers to remotely access sensitive systems.
* **Reconnaissance:** Scanning ports and IP addresses to find vulnerable entry points.

### Traditional Firewalls vs. Our Machine Learning Approach
| Traditional Signature-Based Firewalls | Our Machine Learning NIDS |
| :--- | :--- |
| Looks for exact, known byte sequences (signatures). | Learns behavioral patterns of traffic (flow duration, byte counts, TTL). |
| Fails completely against new or slightly modified attacks. | Detects novel zero-day attacks based on anomalous behavior. |
| Requires constant manual rule updates by security staff. | Self-contained, trainable mathematical models. |

---

## 2. The Core Concept: The Two-Stage Cascade Architecture

To balance real-time speed and deep analysis, our system uses an **Airport Security Screening Analogy**:

```
                       Incoming Network Traffic
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   STAGE 1: BINARY FILTER     │
                   │       (Decision Tree)        │
                   └──────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
         [Normal / Benign]                 [Attack / Anomaly]
         (99.7% Precision)                         │
                 │                                 ▼
      Instant Pass (No Alert)       ┌──────────────────────────────┐
                                    │ STAGE 2: ATTACK ATTRIBUTION  │
                                    │    (Tuned Random Forest)     │
                                    └──────────────────────────────┘
                                                   │
                                                   ▼
                                    Specific Threat Classification:
                                    - DoS
                                    - Exploits
                                    - Fuzzers
                                    - Reconnaissance
                                    - Generic
                                    - Analysis
                                    - Backdoor
                                    - Shellcode
                                    - Worms
```

* **Stage 1 (Binary Filter — Decision Tree):** Fast triage. It evaluates incoming packets and lets clean normal traffic pass immediately. Normal traffic is classified with **99.7% precision**, ensuring benign operations are never disrupted.
* **Stage 2 (Multi-Class Categorizer — Random Forest):** Only packets flagged as `Attack` are routed to the multi-class model to identify the specific threat family.

---

## 3. The Dataset: UNSW-NB15

The project is built on the **UNSW-NB15** cybersecurity benchmark dataset developed by the Cyber Range Lab of UNSW Canberra:
* **Total Records:** 257,673 raw network flows.
* **Features:** 42 raw flow features plus `id` (dropped to prevent leakage), `label` (binary target: 0 or 1), and `attack_cat` (multi-class target: 10 categories).
* **Feature Categories:**
  * Flow metrics: Duration (`dur`), packet counts (`spkts`, `dpkts`), transaction byte sizes (`sbytes`, `dbytes`), transmission rate (`rate`).
  * Network properties: Source/destination Time-To-Live (`sttl`, `dttl`), window advertisements (`swin`, `dwin`), TCP round-trip times (`tcprtt`).
  * Categorical protocols: Protocol (`proto`: TCP, UDP, etc.), service (`service`: HTTP, DNS, SMTP, FTP), and connection state (`state`: FIN, CON, INT).
  * Behavioral connection counters: Connections to the same destination (`ct_dst_ltm`), connections to the same service (`ct_srv_dst`), state TTL counters (`ct_state_ttl`).

---

## 4. End-to-End Pipeline Walkthrough

### Step 1: Data Cleaning (`src/data_loader.py`)
* Stripped leading and trailing whitespaces across all text columns.
* Replaced infinite values with `NaN` and audited nulls.
* Normalized category typos (e.g., `'Backdoors'` mapped to `'Backdoor'`).
* **Removed duplicates strictly without the artificial `id` column:** Dropping `id` prevents the AI from memorizing row indices instead of real network behavior (**data leakage**). Cleaned sets: 107,740 training records, 55,945 testing records.

### Step 2: Leak-Free Preprocessing (`src/preprocess.py`)
* Applied `StandardScaler` to all 39 numeric features so large numbers (e.g. byte volume) don't dominate small numbers (e.g. duration in seconds).
* Applied `OneHotEncoder(handle_unknown="ignore")` to the 3 categorical features (`proto`, `service`, `state`), expanding categorical combinations into 1s and 0s.
* **Strict Leak-Free Principle:** Fitted preprocessors strictly on the training partition; test data and user uploads are transformed using the saved preprocessor (`models/preprocessor.joblib`).

### Step 3: Feature Selection (`src/features.py`)
* After one-hot encoding, the feature space expanded to **194 columns**.
* Using a 100-tree `RandomForestClassifier` on the training set, we calculated **Gini feature importance**.
* Kept the **Top 30 features** representing **95.31%** of cumulative predictive variance and saved them to `models/selected_features.json`.

### Step 4: Model Comparison Tournament (`src/models.py`, `src/train.py`)
We benchmarked seven algorithms using stratified 5-fold cross-validation and tested them on the held-out test split (55,945 records):
1. **Logistic Regression:** Linear decision boundary baseline.
2. **Decision Tree:** Single tree splitting on Gini impurity.
3. **Random Forest:** Ensemble of 100 decorrelated decision trees.
4. **K-Nearest Neighbors (KNN):** Distance-based instance classifier.
5. **Naive Bayes:** Gaussian probability density estimator.
6. **XGBoost:** Gradient boosted decision trees.
7. **Support Vector Machine (Sampled SVM):** Linear/RBF hyperplane optimizer.

### Step 5: Benchmark Winners & Key Results
* **Binary Winner — Decision Tree:**
  * **Accuracy:** 83.79%
  * **Attack Recall:** **92.99%** (Caught 93 out of every 100 attacks)
  * **Macro-F1:** **0.8357**
  * **ROC-AUC:** 0.9207
* **Multi-Class Winner — Tuned Random Forest:**
  * **Accuracy:** 65.40%
  * **Macro-Recall:** **67.36%**
  * **Macro-F1:** **0.5230**
  * **Weighted-F1:** 0.7098
  * **Rare Attacks Caught:** Worms: **88.64% Recall** (39/44 detected), Shellcode: **82.28% Recall** (311/378 detected).

### Step 6: Model Explainability with SHAP (`src/evaluate.py`)
* **Source Time-To-Live (`sttl`):** Accounts for **72.64%** of Decision Tree feature importance. Attack tools (Metasploit, Nmap scanners) and virtual machines have distinct default TTL values compared to benign client operating systems.
* **SHAP Beeswarm Attribution:** Confirmed that elevated `sttl` values strongly push predictions toward the `Attack` class, while low response bytes (`dbytes`) coupled with repeated connection attempts (`ct_srv_dst`) confirm scanning behavior.

### Step 7: Prediction Engine & Web Dashboard (`src/predict.py`, `app/app.py`)
* Built with **Streamlit** following strict clean UI guidelines.
* **Detect Tab:** Allows uploading CSV files, selecting test scenarios, running cascade predictions, inspecting summary metrics, and downloading flagged results.
* **Model Results Tab:** Pure viewing dashboard displaying all benchmark tables, confusion matrices, ROC curves, and SHAP explainability charts without running offline compute.

---

## 5. Technology Stack & Libraries

| Tool / Library | Version | Purpose in Project |
| :--- | :---: | :--- |
| **Python** | 3.10+ | Core programming language. |
| **Pandas** | 3.0.5 | Data manipulation, CSV ingestion, DataFrame transformations. |
| **NumPy** | 2.4.1 | High-performance numerical and matrix operations. |
| **Scikit-learn**| 1.9.0 | Scalers, encoders, Decision Tree, Random Forest, metrics, cross-validation. |
| **XGBoost** | 3.4.1 | Extreme Gradient Boosting classifier with custom multi-class wrappers. |
| **Imbalanced-learn**| 0.14.2| Balanced class weighting strategies for rare threat categories. |
| **Joblib** | 1.6.0 | Serializing model pipelines and preprocessors (`.joblib` format). |
| **SHAP** | 0.52.0 | TreeSHAP explainability and beeswarm feature attribution. |
| **Matplotlib / Seaborn**| 3.11 / 0.13 | High-resolution publication plots (confusion matrices, ROC, bar charts). |
| **Streamlit**| 1.66.1 | Web application dashboard for online batch traffic detection. |
| **Pytest** | 9.1.1 | Automated unit testing suite covering pipeline resilience and schema checks. |
| **Cloudflare**| Tunnel | Secure, zero-trust edge tunneling exposing localhost to the public internet. |

---

## 6. Viva & Defense Preparation: Frequently Asked Questions

### Q1: "Why not use Deep Learning (e.g. LSTMs or CNNs)?"
> **Answer:** *"For tabular network flow records with statistical features (packet rates, byte sizes, TTL), tree-based ensembles (Random Forest, XGBoost) consistently outperform deep learning models in both speed and accuracy. Deep learning requires millions of parameters and significant GPU compute, whereas our Decision Tree and Random Forest run in milliseconds on standard CPU hardware, making them practical for edge firewall integration. Furthermore, tree-based models offer native explainability via SHAP, which deep neural networks lack."*

### Q2: "What is the difference between Accuracy and Macro-F1, and why does Macro-F1 matter here?"
> **Answer:** *"In intrusion detection, class distributions are heavily imbalanced. For example, in our test set, Normal traffic has 34,206 records, while Worms has only 44 records. If a model predicted 'Normal' for every single packet, it would achieve over 61% accuracy while completely failing to stop all attacks. Macro-F1 gives equal weight to all classes regardless of size, ensuring the model is penalized if it ignores rare, high-severity attacks."*

### Q3: "What were the hardest attack categories to detect and why?"
> **Answer:** *"Analysis attacks (F1: 0.0974) and Backdoors (F1: 0.1867) were the hardest. Analysis attacks consist of web vulnerability scanning and SQL injection probes that closely mimic regular user browser behavior (HTTP GET/POST requests). Backdoors use low-and-slow command beacons designed specifically to blend in with normal background keep-alives. Without Deep Packet Inspection (DPI) of payload bodies, flow-level metadata alone makes them challenging to separate from benign traffic."*

### Q4: "How does the Streamlit app communicate with the models?"
> **Answer:** *"The app never trains anything. During offline development, our training script exported `preprocessor.joblib` (feature scaler/encoder), `feature_columns.json` (schema validator), `best_binary.joblib` (Stage 1 Decision Tree), and `best_multiclass.joblib` (Stage 2 Random Forest). The web app loads these pre-computed artifacts once into RAM using `st.cache_resource`, ensuring instantaneous predictions when a user uploads a CSV."*
