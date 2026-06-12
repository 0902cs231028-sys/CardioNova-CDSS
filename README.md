# 🫀 SPDN-CardioNova CDSS

### Clinical Decision Support System for Heart Failure Detection & Automated Care Management

**Live Demo:** [https://cardionova.streamlit.app/](https://cardionova.streamlit.app/)

---

## 1. Overview

**SPDN-CardioNova CDSS** (*Spandan — meaning heartbeat*) is a Clinical Decision Support System designed to bridge machine learning inference with real-world cardiology decision-making.

Unlike traditional black-box prediction systems, this platform transforms model outputs into clinically interpretable actions through:

- Risk stratification for heart failure prediction
- Explainable AI insights using SHAP-based attribution
- Guideline-driven treatment logic aligned with ACC/AHA principles

The system is designed for **point-of-care decision assistance**, not standalone diagnosis.

---

## 2. Project Structure

```text
SPDN-CardioNova-CDSS/
├── app/                        # Streamlit frontend (point-of-care UI)
│   ├── assets/                 # CSS and JS for the Streamlit app
│   │   ├── style.css
│   │   └── main.js
│   └── main.py                 # Application entrypoint
│
├── artifacts/                  # Trained models & serialized assets
│   ├── clinical_knn_imputer.pkl
│   ├── feature_schema_lock.json
│   └── heart_failure_catboost_core.cbm
│
├── core/                       # ML inference + clinical intelligence layer
│   ├── config.py               # Paths, thresholds, constants
│   ├── pipeline.py             # Schema alignment + imputation
│   ├── engine.py               # CatBoost inference wrapper
│   ├── explainability.py       # SHAP-based explanations
│   └── clinical_rules.py       # Rule-based GDMT engine
│
├── Data/                       # Datasets used for training
│   ├── CAD.csv
│   └── heart_disease_uci.csv
│
├── Docs/                       # Project documentation
│   ├── Heart_Failure_Detection_and_Management_System.pdf
│   └── Heart_Failure_System_Presentation.pptx
│
├── Notebook/                   # Jupyter notebooks for experimentation
│   ├── heart-disease-detection.ipynb
│   └── novaheart-disease-detection.ipynb
│
├── train.py                    # Script for training the model
├── requirements.txt            # Project dependencies
└── README.md                   # Project README file
```

---

## 3. Core Features

### 🧠 Defensive ML Pipeline

* Schema-locked feature alignment using `feature_schema_lock.json`
* Robust missing value handling via serialized `KNNImputer`
* Safe inference even with incomplete clinical inputs

---

### 🎯 Clinical Safety Optimization

* Threshold tuned to **0.5723** instead of default 0.5
* Designed to prioritize recall over precision
* Targets ≥95% sensitivity for high-risk patient detection

---

### 🩺 Clinical Decision Engine

* Automatically classifies:

  * HFrEF
  * HFmrEF
  * HFpEF
* Generates guideline-aligned care suggestions:

  * Diuretics
  * ARNI therapy
  * Risk monitoring recommendations

---

### 🔍 Explainable AI Layer

* SHAP-based feature attribution
* Visual breakdown of risk contributors
* Improves clinician trust and interpretability

---

### 🎨 Clinical UI (Streamlit)

* Glassmorphism-inspired dashboard
* Sidebar intake form for vitals & labs
* Real-time inference + explanation panel

---

## 4. Installation

### Prerequisites

* Python 3.12+

---

### 1. Clone Repository

```bash
git clone https://github.com/ayushmorbar/SPDN-CardioNova-CDSS.git
cd SPDN-CardioNova-CDSS
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. (Optional) Train Model & Generate Artifacts

```bash
python train.py
```

This will:

* Train CatBoost model
* Fit preprocessing pipeline
* Export artifacts to `/artifacts`

---

### 4. Run Application

```bash
streamlit run app/main.py
```

---

## 5. Clinical Safety Design

Standard ML systems default to a **0.50 decision threshold**, which is unsafe in medical screening contexts.

This system instead uses:

* Optimized threshold: **0.5723**
* Goal: minimize false negatives
* Design priority: clinical recall ≥ 95%

This ensures high-risk patients are not missed during early screening.

---

## 6. Project Structure Philosophy

This system follows:

* Separation of concerns
* Inference isolation
* Deterministic clinical logic layer
* Explainable AI instead of black-box predictions

This makes the system suitable for:

* Academic evaluation
* Clinical prototype demonstration
* Future hospital-scale extension

---

## 7. Author
**Ayush Morbar**

---

## ⚠️ Disclaimer

This system is a Clinical Decision Support System (CDSS) intended for educational and research purposes only. It does not replace professional medical judgment or diagnosis. Always consult a qualified healthcare provider for any medical concerns. The authors are not liable for any clinical decisions made based on this system.
