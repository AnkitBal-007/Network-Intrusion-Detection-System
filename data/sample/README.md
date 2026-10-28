# Sample Input Datasets Guide

This directory contains test network traffic CSV files extracted from the held-out test split of the **UNSW-NB15** dataset. Each file strictly complies with the **42-feature input schema** (without `id`, `label`, or `attack_cat`) required by `src/predict.py` and the Streamlit web application.

---

## Available Test Scenarios

### 1. `sample_input.csv` (Default Balanced Baseline)
- **Rows:** 200
- **Composition:** Stratified proportional sample across all 10 traffic categories (`Normal`, `Exploits`, `Fuzzers`, `Generic`, `Reconnaissance`, `DoS`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`).
- **Expected Results:**
  - Normal: ~98 rows (49%)
  - Attack: ~102 rows (51%)
  - Top Attack Categories: `Fuzzers`, `Exploits`, `Generic`, `Reconnaissance`.
- **Testing Purpose:** General baseline test demonstrating multi-class detection across diverse attack vectors.

---

### 2. `sample_benign_only.csv` (Normal Baseline)
- **Rows:** 100
- **Composition:** 100% legitimate background traffic records (HTTP browsing, DNS requests, SMTP mail exchanges, FTP sessions).
- **Expected Results:**
  - Normal: ~78%
  - Attack: ~22% (minimal false-positive alarms on dynamic web traffic)
- **Testing Purpose:** Demonstrates that the model maintains low false-alarm rates when handling regular everyday corporate network traffic.

---

### 3. `sample_heavy_attacks.csv` (High-Intensity Attack Campaign)
- **Rows:** 100
- **Composition:** 100% malicious traffic flows sampled proportionally across all 9 attack families (zero normal records).
- **Expected Results:**
  - Attack: **96%** (96 of 100 attacks caught)
  - Normal: 4%
  - Top Attack Categories: `Exploits` (32), `Generic` (14), `Backdoor` (13), `Fuzzers` (12), `Reconnaissance` (10).
- **Testing Purpose:** Demonstrates high sensitivity and detection recall during an active multi-vector cyberattack incident.

---

### 4. `sample_dos_recon_flood.csv` (SOC Flood Incident)
- **Rows:** 100
- **Composition:** 
  - 40 Denial of Service (`DoS`) flows
  - 40 Reconnaissance (`Reconnaissance`) port sweeps / IP scans
  - 20 Normal background connections
- **Expected Results:**
  - Attack: **85%**
  - Normal: 15%
  - Dominant Categories: `Reconnaissance` (35) and `DoS` (16).
- **Testing Purpose:** Recreates a typical enterprise Security Operations Center (SOC) scenario where network scanners and DoS floods attempt to overwhelm edge infrastructure.

---

### 5. `sample_stealth_threats.csv` (Advanced Persistent Threats & Rare Exploits)
- **Rows:** 80
- **Composition:**
  - 20 `Backdoor` stealth beacons
  - 20 `Shellcode` execution payloads
  - 20 `Analysis` web penetration probes
  - 10 `Worms` automated self-propagating infections
  - 10 `Normal` background sessions
- **Expected Results:**
  - Attack: **69%**
  - Normal: 11%
  - Dominant Categories: `Backdoor` (24) and `Shellcode` (18).
- **Testing Purpose:** Demonstrates the model's ability to catch stealthy, low-and-slow Advanced Persistent Threats (APTs) and rare malware payloads that traditional signature firewalls miss.

---

## How to Test in the Web App

1. Launch or open the web dashboard (`http://localhost:8501` or your live URL).
2. On the **Detect** tab, choose any scenario from the **"Choose sample scenario"** dropdown.
3. Click the download button to save the file.
4. Drag and drop the downloaded CSV into the file uploader.
5. Click **"Run detection"** to inspect live metrics, breakdown charts, and filtered tables.
