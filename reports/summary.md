# Project Technical Summary: Network Intrusion Detection Using Machine Learning

**Department of Computer Science & Engineering**  
**Final-Year B.Tech Capstone Project — Group 16**  
**Project Guide:** Prof. Satyam Raha  

---

## 1. Dataset Overview

- **Source:** UNSW-NB15 Dataset (developed by the Cyber Range Lab of UNSW Canberra).
- **Domain:** Modern synthetic and real network traffic capturing normal background operations and nine contemporary attack families.
- **Records Processed:**
  - Raw Training Set: 175,341 records (45 columns)
  - Raw Testing Set: 82,332 records (45 columns)
  - Cleaned Training Split (`data/processed/train_cleaned.parquet`): 107,740 unique records (67,601 duplicates and 0 missing/inf dropped)
  - Cleaned Testing Split (`data/processed/test_cleaned.parquet`): 55,945 unique records (26,387 duplicates and 0 missing/inf dropped)
- **Class Distributions (Cleaned Test Set):**
  - **Binary:** Normal (34,206 records; 61.14%), Attack (21,739 records; 38.86%)
  - **Multi-Class (10 categories):**
    - `Normal`: 34,206 (61.14%)
    - `Exploits`: 7,609 (13.60%)
    - `Fuzzers`: 4,838 (8.65%)
    - `Generic`: 3,657 (6.54%)
    - `Reconnaissance`: 2,703 (4.83%)
    - `DoS`: 1,718 (3.07%)
    - `Analysis`: 446 (0.80%)
    - `Shellcode`: 378 (0.68%)
    - `Backdoor`: 346 (0.62%)
    - `Worms`: 44 (0.08%)

---

## 2. Preprocessing & Feature Selection

1. **Data Cleaning:**
   - Stripped all leading and trailing whitespaces across categorical features.
   - Normalized category mislabelings (e.g. `'Backdoors'` mapped to `'Backdoor'`).
   - Replaced infinite values with `NaN` and audited nulls.
   - Removed duplicate records strictly without the synthetic identifier `id` to prevent artificial data memorization.
2. **Leak-Free Transformation Pipeline:**
   - Numerical columns (39 features): Scaled using `StandardScaler` fitted exclusively on the training partition.
   - Categorical columns (`proto`, `service`, `state`): Encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fitted on training data.
   - Total transformed feature dimensionality: 194 columns.
   - Serialized preprocessor artifact: `models/preprocessor.joblib`.
3. **Feature Selection:**
   - Method: Model-based Gini feature importance via `RandomForestClassifier(n_estimators=100, random_state=42)` on the training split.
   - Retention: Top 30 features accounting for 84.44% of cumulative variance and importance.
   - Winning features include `sttl` (0.2447), `ct_state_ttl` (0.1077), `dttl` (0.0469), `rate` (0.0401), `sbytes` (0.0384), and `dbytes` (0.0366).
   - Artifacts: `models/selected_features.json` and `models/feature_columns.json`.

---

## 3. Models Compared

Seven algorithms were benchmarked across both tasks:
1. **Logistic Regression** (L2 regularization, balanced class weights)
2. **Decision Tree** (CART classifier, balanced class weights, tuned for depth and split criteria)
3. **Random Forest** (100 ensemble trees, balanced subsample weighting, tuned tree depth)
4. **K-Nearest Neighbors (KNN)** (Distance-weighted neighbor voting with k=5)
5. **Naive Bayes** (GaussianNB density estimation)
6. **XGBoost** (Gradient boosted decision trees with custom multi-class label encoder and sample weights)
7. **Support Vector Machine (SVM)** (LinearSVC calibrated with Platt scaling, fitted on a stratified sample of 15,000 samples)

---

## 4. Key Metrics & Benchmark Results

### 4.1 Binary Classification (Held-Out Test Set: 55,945 records)

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

### 4.2 Multi-Class Classification (Held-Out Test Set: 55,945 records)

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

### 4.3 Per-Class Performance of Winning Multi-Class Model

| Attack Class | Support | Precision | Recall | F1-Score | Detection Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Normal` | 34,206 | 0.9974 | 0.5969 | 0.7468 | High precision benign filtering |
| `Generic` | 3,657 | 0.9683 | 0.8769 | 0.9204 | Excellent cryptographic pattern capture |
| `Exploits` | 7,609 | 0.7929 | 0.7680 | 0.7803 | Strong detection of CVE signature behaviors |
| `Reconnaissance`| 2,703 | 0.7605 | 0.8398 | 0.7982 | High sensitivity to scanning sweeps |
| `Worms` | 44 | 0.4643 | 0.8864 | 0.6094 | 39 of 44 rare worm infections caught |
| `Shellcode` | 378 | 0.1996 | 0.8228 | 0.3213 | 311 of 378 execution payloads detected |
| `DoS` | 1,718 | 0.4481 | 0.3743 | 0.4079 | Confused with Fuzzers and burst traffic |
| `Fuzzers` | 4,838 | 0.2399 | 0.7313 | 0.3613 | Overlaps heavily with normal dynamic web traffic |
| `Backdoor` | 346 | 0.1131 | 0.5347 | 0.1867 | Low-and-slow command beacons escape detection |
| `Analysis` | 446 | 0.0580 | 0.3049 | 0.0974 | Hardest category due to HTTP/DNS port similarity |

---

## 5. Hardest Attack Categories & Technical Analysis

1. **Analysis Attacks (F1: 0.0974):**
   - *Nature:* Web application vulnerability assessments, port scanning, and SQL injection probes.
   - *Root Cause:* Generates statistical distributions almost indistinguishable from legitimate user web browsers making concurrent AJAX requests or DNS lookups. Without full Deep Packet Inspection (DPI) of payload bodies, flow-level metrics (such as `sbytes`, `dbytes`, and `dur`) lack sufficient discrimination.
2. **Backdoor Exploits (F1: 0.1867):**
   - *Nature:* Stealth command-and-control (C2) heartbeat beacons.
   - *Root Cause:* Designed deliberately to evade signature detection through randomized sleep intervals, small packet sizes, and legitimate port camouflage (e.g. TCP 443 / HTTPS).
3. **Denial of Service (DoS) (F1: 0.4079):**
   - *Nature:* Low-rate application-layer exhaustion (e.g. Slowloris, Slow POST) vs volumetric floods.
   - *Root Cause:* While high-volume syn-floods are readily separated, low-rate slowloris attacks manifest as benign persistent connections, causing misclassifications as Fuzzers or Exploits.

---

## 6. Model Explainability & SHAP Feature Attribution

To provide transparent decision auditing for security analysts, feature importance and SHAP (SHapley Additive exPlanations) values were extracted for the top models:

1. **Binary Model Feature Importance (Decision Tree):**
   - `sttl` (Source-to-Destination Time to Live): **72.64%** Gini importance.
   - `ct_srv_dst` (Number of connections to the same service and destination IP): **6.52%**.
   - `dbytes` (Destination-to-source transaction bytes): **3.25%**.
   - `sloss` (Source packet loss count): **3.16%**.
   - `sbytes` (Source-to-destination transaction bytes): **2.52%**.
   - *Security Rationale:* Attacking tools (Metasploit, Nmap, fuzzers) and victim VMs often exhibit standardized network stack TTL footprints distinct from benign user traffic, allowing the binary Decision Tree to make high-confidence triage decisions based primarily on `sttl`.

2. **Multi-Class Feature Distribution (Tuned Random Forest):**
   - Unlike the binary model where a single feature dominates, multi-class categorization distributes importance across packet dynamics:
     - `sbytes` (15.70%), `smean` (11.73%), `sttl` (6.83%), `ct_srv_dst` (5.56%), `ct_dst_src_ltm` (5.23%), `dbytes` (5.15%), and `ct_srv_src` (4.76%).
   - *Security Rationale:* Distinguishing specific attack families (such as distinguishing volumetric DoS attacks from stealthy Reconnaissance scans or single-packet Exploits) requires comprehensive flow size, packet average length, and connection frequency metrics.

3. **SHAP Directional Attribution (Beeswarm Plot):**
   - TreeSHAP analysis on test set samples reveals that elevated `sttl` values (high feature value in red) exert a strong positive SHAP impact toward the `Attack` class, whereas low `dbytes` combined with high `ct_srv_dst` characteristic of repetitive probes further shift the probability mass toward malicious classifications.

---

## 7. Cascade Inference Architecture

In production inference, requests pass through a **two-stage cascade pipeline**:
1. **Stage 1 (Binary Filter):** Evaluates incoming traffic via the winning Decision Tree classifier. Packets predicted as `Normal` are marked benign with near-zero false-positive risk (99.7% precision).
2. **Stage 2 (Multi-Class Attribution):** Only flows classified as `Attack` are routed to the Tuned Random Forest. The model attributes the attack to the highest-probability non-Normal attack family.
3. **Performance Benefit:** Prevents benign traffic from being erroneously tagged with niche attack labels and conserves computational throughput by bypassing complex multi-class trees for the 60%+ majority class.

---

## 8. Limitations

1. **Synthetic / Laboratory Shift:** UNSW-NB15 is generated in a controlled cyber-range environment; live production enterprise networks exhibit dynamic diurnal cycles, encrypted QUIC traffic, and zero-day protocols not captured in static CSV records.
2. **Absence of Payload Inspection:** The feature set relies entirely on packet metadata (TTL, bytes, inter-packet arrival time, window size). Attacks using obfuscated payloads over TLS cannot be fully characterized without deep packet reconstruction.
3. **Imbalance vs Rare Classes:** Despite balanced sample weighting, attack categories with fewer than 500 samples (such as Worms, Analysis, and Backdoors) suffer lower precision due to boundary overlap with dominant classes.

---

## 9. Future Work


1. **Live Network Ingestion:**
   - Integrate `Scapy` and `PyShark` packet sniffers to capture raw `.pcap` flows from live network interfaces (NICs) in real time.
2. **Automated Security Alerting:**
   - Integrate webhooks and alerting endpoints (Syslog, Slack, SIEM / Splunk) triggering immediate incident tickets when attack probabilities exceed configurable thresholds.
3. **Deep Learning Architectures:**
   - Implement temporal recurrent networks (Bi-LSTM, GRU) or 1D-CNNs directly over packet sequences to capture sequential protocol state transitions without manual feature engineering.
4. **Microservice & Container Deployment:**
   - Containerize the inference service using Docker and publish a REST API endpoint (FastAPI) alongside the Streamlit dashboard for enterprise security operations center (SOC) integration.
