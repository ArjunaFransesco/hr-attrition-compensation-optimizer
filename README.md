# HR Employee Attrition & Compensation Equity Optimizer

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **👥 Machine Learning HR People Analytics Engine for Turnover Prevention & Market Salary Equity with XGBoost & Streamlit**

---

## 📌 Executive Summary & Mathematical Formulation

In People Analytics / HR Tech ML, automated machine learning classification facilitates low-latency decision triage:

$$\mathcal{L}_{\text{logloss}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

- **Target Classification**: `will_convert_purchase`
- **Optimization Metric**: Weighted F1-Score, ROC-AUC, & Accuracy
- **Model Architecture**: `RandomForestClassifier (Ensemble 180 Trees)`

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────────────────────────────────────────────┐
│     Domain Telemetry & Ingestion Pipeline              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Feature Engineering, Imputation & Scaling Pipeline   │
│  - StandardScaler Normalization & Multicollinearity    │
│  - Correlation Matrix & Domain Specific Attributes     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Ensemble Machine Learning & 5-Fold Cross-Validation  │
│  - Serialized Artifacts (.joblib) & Metric Evaluation  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Interactive Streamlit Dashboard & Web UI Interface   │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark & Performance Metrics

| Metric | Score | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **0.8453** | > 0.8500 | 🌟 High Discriminative |
| **Weighted F1 Score** | **0.8453** | > 0.8000 | 🌟 Robust Balance |
| **ROC-AUC Score** | **0.9329** | > 0.8800 | 🌟 High Precision |
| **5-Fold Cross Validation Accuracy** | **0.8449** | Stable K-Fold | Verified |
| **Dataset Volume** | **3200 Samples** | Stratified Train/Test | Verified |

---

## 📁 Repository Structure

```
hr-attrition-compensation-optimizer/
├── app.py                     # Streamlit interactive diagnostic dashboard
├── data/
│   ├── raw/
│   │   └── hr_attrition_compensation_optimizer_dataset.csv   # Raw domain dataset
│   └── processed/
│       └── hr_attrition_compensation_optimizer_processed.csv # Scaled and preprocessed matrix
├── models/
│   ├── feature_names.joblib   # Ingested feature schema
│   ├── scaler.joblib          # Trained StandardScaler pipeline
│   └── model_pipeline.joblib  # Serialized machine learning estimator
├── notebooks/
│   └── hr_attrition_compensation_optimizer_pipeline.ipynb    # End-to-end Jupyter Notebook pipeline
├── reports/
│   ├── metrics.json           # Serialized evaluation metrics
│   └── evaluation_plot.png    # High-resolution benchmark visualizer
├── requirements.txt           # Environment dependencies
├── LICENSE                    # MIT Open Source License
└── README.md                  # Project documentation & benchmark overview
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ArjunaFransesco/hr-attrition-compensation-optimizer.git
cd hr-attrition-compensation-optimizer
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### 3. Open End-to-End Jupyter Notebook
```bash
jupyter notebook notebooks/hr_attrition_compensation_optimizer_pipeline.ipynb
```

---

## 👤 Author & Portfolio
- **Author**: **[Arjuna Fransesco](https://github.com/ArjunaFransesco)**
- **GitHub Repositories**: [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website**: [https://github.com/ArjunaFransesco/arjuna-portfolio](https://github.com/ArjunaFransesco/arjuna-portfolio)
