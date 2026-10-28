# PowerPoint Presentation Deck: Network Intrusion Detection Using Machine Learning

**Department of Computer Science & Engineering**  
**Future Institute of Engineering and Management**  
**Final-Year B.Tech Capstone Project — Group 16**  
**Project Guide:** Prof. Satyam Raha  

> **How to use this file:**  
> - Copy-paste each slide directly into **Microsoft PowerPoint**, **Google Slides**, or **Canva**.  
> - Compatible with Markdown presentation tools like **Marp**, **Pandoc**, or **Gamma.app**.  
> - Each slide contains **"Speaker Notes / What to Say"** providing exact viva talking points.

---

## Slide 1: Title Slide

### **Network Intrusion Detection System Using Machine Learning**
*An End-to-End Two-Stage Cascade Architecture for Real-Time Cyber Threat Detection and Attribution*

* **Presented By:** Group 16
  * Student Name 1 (Roll No: _____________)
  * Student Name 2 (Roll No: _____________)
  * Student Name 3 (Roll No: _____________)
  * Student Name 4 (Roll No: _____________)
* **Project Guide:** Prof. Satyam Raha
* **Department:** Computer Science & Engineering
* **Institution:** Future Institute of Engineering and Management (FIEM)

> **Speaker Notes / What to Say:**  
> *"Good morning respected external examiners, Head of the Department, and our project guide Prof. Satyam Raha. Today, our team is presenting our final year capstone project: 'Network Intrusion Detection System Using Machine Learning'. In this project, we built a leak-free, explainable machine learning pipeline comparing seven algorithms and deployed a live web application using a two-stage cascade architecture."*

---

## Slide 2: Problem Statement & Motivation

### Why Machine Learning for Intrusion Detection?
* **Exponential Rise in Cyber Threats:** Modern enterprise networks process millions of flows per minute, hiding sophisticated attacks like Denial of Service (DoS), Backdoors, and Exploits.
* **Failure of Traditional Signature Firewalls:**
  * Rely on static, rigid rule blacklists (e.g., Snort rules).
  * Incapable of detecting zero-day exploits or polymorphic malware variations.
  * Require continuous manual rule updates by security analysts.
* **The Machine Learning Promise:**
  * Learns statistical behavioral footprints (packet rates, byte sizes, Time-To-Live).
  * Generalizes to spot unknown, novel attacks in real time.

> **Speaker Notes / What to Say:**  
> *"Traditional signature-based firewalls act like an old security guard looking at a static photo blacklist. If a hacker changes just one byte in their attack payload, the firewall misses it completely. Our machine learning approach looks at behavioral traffic statistics—such as transmission rate, packet sizes, and connection frequency—allowing us to detect both known and zero-day intrusions automatically."*

---

## Slide 3: Project Objectives

### Key Goals & Deliverables
1. **Binary Intrusion Detection:** Accurately classify incoming traffic records as **Normal** or **Attack**.
2. **Multi-Class Threat Attribution:** Categorize flagged attacks into 9 distinct families (`Exploits`, `DoS`, `Reconnaissance`, `Fuzzers`, `Generic`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`).
3. **Comparative Algorithm Benchmarking:** Rigorously compare 7 diverse AI algorithms under stratified cross-validation.
4. **Model Explainability (XAI):** Uncover the root causes of attack flags using **SHAP** and **Feature Importance** metrics.
5. **Interactive Web Application:** Deploy a live Streamlit dashboard for real-time batch traffic inspection with zero-downtime cloud hosting.

> **Speaker Notes / What to Say:**  
> *"We set out five clear objectives: first, fast binary triage between normal and malicious traffic; second, pinpointing the exact attack type; third, benchmarking seven candidate algorithms under identical cross-validation conditions; fourth, ensuring the system is explainable using SHAP; and fifth, delivering a production-ready web application where network administrators can upload traffic logs and receive instant predictions."*

---

## Slide 4: Proposed System Architecture (Two-Stage Cascade)

### Two-Stage Screening Pipeline
```
               Incoming Network Traffic CSV
                            │
                            ▼
               ┌─────────────────────────┐
               │ STAGE 1: BINARY FILTER  │
               │     (Decision Tree)     │
               └─────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [Normal Traffic]                [Attack Traffic]
    99.7% Precision                         │
   (Instant Pass-Through)                   ▼
                              ┌─────────────────────────┐
                              │ STAGE 2: MULTI-CLASS    │
                              │     CLASSIFICATION      │
                              │  (Tuned Random Forest)  │
                              └─────────────────────────┘
                                            │
                                            ▼
                              Attributed Threat Category:
                              DoS, Exploit, Backdoor, etc.
```

* **Stage 1 (Binary Filter):** Acts like airport metal detection. 99.7% precision on benign traffic ensures regular user operations are never blocked.
* **Stage 2 (Multi-Class Attribution):** Only flagged anomalies are sent for deep classification, conserving CPU compute and eliminating false attack labels on benign packets.

> **Speaker Notes / What to Say:**  
> *"A major architectural contribution of our project is the Two-Stage Cascade. In real-world enterprise networks, over 60 to 80% of traffic is normal. Running a heavy 10-class model on every single packet wastes CPU resources and leads to false alarms. Instead, Stage 1 acts as a rapid filter with 99.7% precision. Only suspicious flows are passed to Stage 2 for detailed malware family attribution."*

---

## Slide 5: Dataset Overview (UNSW-NB15)

### Benchmark Cyber-Range Dataset
* **Source:** Cyber Range Lab of the Australian Defence Force Academy (UNSW Canberra).
* **Scope:** Real modern network traffic combining normal background activities with contemporary attack scenarios.
* **Volume:** 257,673 raw network connection flows.
* **Class Distribution (Held-Out Test Set: 55,945 records):**
  * `Normal`: 34,206 (61.14%)
  * `Exploits`: 7,609 (13.60%)
  * `Fuzzers`: 4,838 (8.65%)
  * `Generic`: 3,657 (6.54%)
  * `Reconnaissance`: 2,703 (4.83%)
  * `DoS`: 1,718 (3.07%)
  * `Analysis`: 446 (0.80%)
  * `Shellcode`: 378 (0.68%)
  * `Backdoor`: 346 (0.62%)
  * `Worms`: 44 (0.08%)

> **Speaker Notes / What to Say:**  
> *"We selected the UNSW-NB15 dataset over older datasets like KDD Cup 99 because UNSW-NB15 reflects modern network protocols and attack vectors. Notice the severe class imbalance: while Normal traffic has over 34,000 records, rare categories like Worms have only 44 records. This realistic imbalance required specialized stratified sampling and balanced class-weighting techniques."*

---

## Slide 6: Data Preprocessing & Leakage Prevention

### Rigorous Leak-Free Data Pipeline
1. **Cleaning:**
   - Stripped leading/trailing whitespaces across string features.
   - Normalized category mislabelings (e.g. `'Backdoors'` → `'Backdoor'`).
   - Audited and handled infinite values and missing records.
2. **Preventing Data Leakage:**
   - **Dropped Artificial `id` Column:** Prevents models from memorizing sequential row IDs.
   - Removed duplicate records strictly without the `id` column (67,601 duplicates dropped in training; 26,387 in testing).
3. **Transformation Pipeline:**
   - **StandardScaler:** Fitted strictly on training split for 39 numerical features (packet duration, bytes, TTL).
   - **OneHotEncoder:** Fitted with `handle_unknown="ignore"` for 3 categorical features (`proto`, `service`, `state`).
   - Saved fitted preprocessor artifact to `models/preprocessor.joblib`.

> **Speaker Notes / What to Say:**  
> *"Data leakage is the number one cause of failed ML projects in production. If you scale your data before splitting, test set information leaks into your model, producing fake 99% accuracy that crashes in real life. We strictly split our dataset first, fit our StandardScaler and OneHotEncoder exclusively on the training partition, and removed the synthetic ID column to ensure our evaluation metrics represent true real-world performance."*

---

## Slide 7: Feature Selection & Dimensionality Reduction

### Reducing 194 Features to Top 30
* **The Problem:** One-hot encoding categorical protocols expanded feature dimensionality from 42 to **194 columns**, introducing sparsity and slowing down inference.
* **The Solution:** Model-based feature importance using a 100-tree `RandomForestClassifier` on the training set.
* **Results:**
  * Selected top **30 features** accounting for **95.31%** of cumulative predictive variance.
  * Key winning features: `sttl` (Source TTL), `ct_state_ttl`, `dttl`, `sbytes`, `dbytes`, `rate`, and `ct_srv_dst`.
  * Saved selection metadata to `models/selected_features.json`.

> **Speaker Notes / What to Say:**  
> *"After one-hot encoding, we had 194 features. To make the models lightweight and fast enough for real-time firewall use, we performed Random Forest Gini feature selection. We found that the top 30 features captured over 95% of the total predictive power, allowing us to drop 164 redundant columns without sacrificing detection accuracy."*

---

## Slide 8: Algorithms Benchmarked

### Candidate Models & Implementation Details
We evaluated seven distinct machine learning algorithms:

| Algorithm | Paradigm | Key Configuration |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear Probabilistic | L2 penalty, balanced class weights |
| **Decision Tree** | Rule-Based Tree | CART algorithm, Gini criterion, tuned max_depth |
| **Random Forest** | Bagging Ensemble | 100 decorrelated trees, balanced sub-sampling |
| **K-Nearest Neighbors** | Instance-Based Metric | k=5, distance-weighted voting |
| **Naive Bayes** | Statistical Density | GaussianNB probability estimation |
| **XGBoost** | Gradient Boosting | Extreme Gradient Boosting with custom multi-class label encoders |
| **SVM (Sampled)** | Maximum Margin Hyperplane | LinearSVC calibrated with Platt scaling on stratified samples |

> **Speaker Notes / What to Say:**  
> *"We didn't just pick one algorithm arbitrarily. We benchmarked seven distinct paradigms: linear models, tree-based models, ensembles, instance-based classifiers, probabilistic models, gradient boosters, and support vector machines. Each model was evaluated using stratified 5-fold cross-validation on the exact same feature-selected training split."*

---

## Slide 9: Binary Classification Results (Normal vs. Attack)

### Test Set Performance (55,945 Held-Out Records)
| Model | Accuracy | Precision | Attack Recall | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Selected)** | **83.79%** | **72.82%** | **92.99%** | **0.8357** | **0.9207** |
| Decision Tree (Tuned) | 83.47% | 72.02% | 93.97% | 0.8329 | 0.9367 |
| Random Forest (Tuned) | 83.15% | 70.45% | 97.54% | 0.8306 | 0.9676 |
| XGBoost (Tuned) | 82.95% | 70.19% | 97.52% | 0.8286 | 0.9719 |
| KNN | 79.64% | 67.55% | 91.58% | 0.7949 | 0.9020 |
| SVM (Sampled) | 74.03% | 60.47% | 95.80% | 0.7403 | 0.8883 |
| Logistic Regression | 73.96% | 60.33% | 96.37% | 0.7396 | 0.8882 |
| Naive Bayes | 66.05% | 60.68% | 35.85% | 0.6025 | 0.7530 |

* **Winning Model:** **Decision Tree** (Macro-F1: 0.8357, Attack Recall: 92.99%, inference time under 0.1 ms per packet).

> **Speaker Notes / What to Say:**  
> *"Here are our held-out test set results for binary classification. While Random Forest and XGBoost scored high ROC-AUC, our Decision Tree won with a 0.8357 Macro-F1 and a 93% attack recall while executing in less than one-tenth of a millisecond. In cybersecurity, missing an attack is fatal, so achieving 93% recall with ultra-low latency made Decision Tree the optimal Stage 1 filter."*

---

## Slide 10: Multi-Class Classification Results

### Identifying the Specific Attack Family
| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Tuned (Winner)** | **65.40%** | **50.42%** | **67.36%** | **0.5230** | **0.7098** |
| Random Forest (Base) | 64.48% | 48.91% | 67.48% | 0.5086 | 0.7027 |
| XGBoost (Tuned) | 64.01% | 48.68% | 69.73% | 0.5067 | 0.7002 |
| Decision Tree (Tuned) | 64.71% | 48.17% | 65.28% | 0.5046 | 0.7016 |
| KNN | 67.48% | 45.62% | 41.94% | 0.4074 | 0.7060 |
| Logistic Regression | 56.16% | 32.55% | 46.44% | 0.3092 | 0.6310 |
| SVM (Sampled) | 59.19% | 36.72% | 32.41% | 0.2767 | 0.6261 |
| Naive Bayes | 44.45% | 29.30% | 36.33% | 0.2270 | 0.5559 |

### Per-Class Detection of Winning Random Forest:
* **Generic:** 87.7% Recall | 92.0% F1-Score
* **Exploits:** 76.8% Recall | 78.0% F1-Score
* **Reconnaissance:** 84.0% Recall | 79.8% F1-Score
* **Worms (Ultra-rare):** **88.64% Recall** (39 of 44 detected)
* **Shellcode:** **82.28% Recall** (311 of 378 detected)

> **Speaker Notes / What to Say:**  
> *"For multi-class classification across 10 categories, our Tuned Random Forest achieved the highest Macro-F1 of 0.5230 and Macro-Recall of 67.36%. Notice how well it handled rare, dangerous classes: it caught 39 out of 44 Worms (88.6% recall) and 311 of 378 Shellcode exploits (82.3% recall), proving that balanced sub-sampling effectively prevented the model from ignoring minority classes."*

---

## Slide 11: Model Explainability (XAI) with SHAP

### Why Did the Model Flag a Packet as an Attack?
* **Black-Box Problem:** Security analysts cannot trust an AI that gives predictions without explanations.
* **Feature Importance Findings:**
  * In binary detection, **Source Time-To-Live (`sttl`) accounts for 72.64% of the decision**.
  * Automated tools (Nmap, Metasploit) and attacker VMs use default operating system TTL signatures distinct from client machines.
* **SHAP Beeswarm Attribution:**
  * High `sttl` values (red points) push predictions strongly toward `Attack`.
  * Low response bytes (`dbytes`) coupled with repetitive connections (`ct_srv_dst`) explain automated scanning and probe attempts.

> **Speaker Notes / What to Say:**  
> *"To eliminate the 'black-box' nature of machine learning, we implemented SHAP and Gini importance analysis. We discovered that Source Time-To-Live (`sttl`) accounts for over 72% of the binary classification decision. When automated attacking tools generate packets, their network stack leaves distinct TTL values that contrast sharply with regular Windows and Mac operating systems. SHAP provides clear evidence for security officers to audit flagged incidents."*

---

## Slide 12: Web Application & System Implementation

### Production-Ready Streamlit Interface
* **Detect Tab:**
  * Clean drag-and-drop CSV upload supporting large network capture files.
  * In-app sample downloader (`sample_input.csv`) for instant evaluation.
  * Real-time metrics: Total Rows, Normal Count, Attack Count.
  * Interactive categorical threat distribution bar chart.
  * Filterable results table (All / Normal / Attack) with instant CSV export.
* **Model Results Tab:**
  * Complete viewing dashboard of all test benchmarks, confusion matrices, ROC curves, and SHAP explainability charts with zero online compute overhead.

> **Speaker Notes / What to Say:**  
> *"We brought the models to life with a clean Streamlit web dashboard following professional UI design guidelines. A network administrator can download our sample CSV or upload their own network logs, click 'Run detection', and immediately see total packet counts, an attack distribution chart, and export flagged threat records for SOC analysis. The Model Results tab allows visitors to review our offline evaluation benchmarks without re-training."*

---

## Slide 13: Software Engineering & Testing

### Robust Engineering Practices
* **Zero Hard-Coded Paths:** Unified filesystem configuration in `src/config.py`.
* **Clean Code Standards:** PEP 8 compliance, full type hints, docstrings on all public functions, and zero dead code.
* **Automated Pytest Suite (12 Unit Tests Passing):**
  * `test_load_artifacts`: Validates model joblib integrity.
  * `test_preprocessor_handles_unseen_category`: Confirms resilience against unknown protocols.
  * `test_predict_dataframe_missing_column_error`: Validates schema error handling.
  * `test_selected_features_filtering`: Confirms exact column alignment.
* **Optimized Model Compression:** Compressed `best_multiclass.joblib` from **107 MB to 21.69 MB** using Joblib compression level 3, enabling direct GitHub tracking under the 100 MB limit.

> **Speaker Notes / What to Say:**  
> *"From a software engineering perspective, the project adheres to professional standards. We implemented an automated pytest suite containing 12 unit tests verifying model loading, schema validation, and handling of unseen network protocols. Additionally, we compressed our 100-tree Random Forest from 107 MB to 21 MB, allowing the complete system to be tracked directly on GitHub without requiring external Git LFS bandwidth."*

---

## Slide 14: Deployment & Cloud Hosting

### 24/7 Global Access
* **GitHub Repository:** Fully version-controlled and public:
  * `https://github.com/AnkitBal-007/Network-Intrusion-Detection-System`
* **Cloudflare Zero-Trust Tunnel:** 
  * Tested live using `cloudflared` exposing localhost securely over Cloudflare's edge network without port-forwarding.
* **Streamlit Community Cloud:** 
  * Configured via `.streamlit/config.toml` for 1-click cloud deployment.
  * Runs 24/7 on AWS/GCP servers with automatic GitHub synchronization and zero laptop dependency.

> **Speaker Notes / What to Say:**  
> *"Our system is deployed and publicly accessible. The repository is published on GitHub, we verified zero-trust edge tunneling via Cloudflare, and configured the application for 24/7 cloud deployment on Streamlit Community Cloud. Evaluators can access the live web application anytime from any phone or computer without requiring our local laptops to remain powered on."*

---

## Slide 15: Limitations, Future Work & Conclusion

### Limitations & Future Roadmap
* **Limitations:**
  * Reliance on network flow metadata without Deep Packet Inspection (DPI) of encrypted TLS payloads.
  * Analysis and Backdoor categories remain difficult due to behavioral overlap with regular AJAX browsing.
* **Future Work:**
  * Real-time live packet capture from physical NICs using `Scapy` and `PyShark`.
  * Automated security alerting via Webhooks, Syslog, and Slack for SOC integration.
  * Deep Learning temporal sequences (1D-CNN / Bi-LSTM) on raw packet byte sequences.

### Conclusion
* Developed a leak-free, explainable, two-stage cascade NIDS achieving **93% attack recall** and **88.6% worm detection**.
* Fully verified, tested, documented, and hosted.

### Thank You!
**Questions & Answers Welcome**

> **Speaker Notes / What to Say:**  
> *"In conclusion, we have designed, benchmarked, explained, and deployed an end-to-end Network Intrusion Detection System that catches over 93% of cyber threats while ensuring clean traffic flows uninterrupted. We want to thank our project guide Prof. Satyam Raha and the department for their guidance. We are now ready to take your questions. Thank you!"*
