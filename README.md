# 🛡️ FraudShield AI — Credit Card Fraud Detection System

> **End-to-end machine learning system for real-time credit card fraud detection**  
> Trained on 1.29M transactions · 99.51% AUC-ROC · 98.43% Accuracy · 93.60% Recall


🚀 Live Demo:
https://ai-fraud-detection-bzpdrbwx2d7f6j54bwzbbd.streamlit.app/

> 📂 GitHub Repository:
https://github.com/alekhya-git-star/Ai-Fraud-Detection

> 
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Project Overview

FraudShield AI is a production-grade fraud detection system that uses a **Random Forest classifier** to identify fraudulent credit card transactions in real time. The system includes a full ML pipeline, a Streamlit web application, SHAP explainability, ROC/PR curves, and a Power BI dashboard.

---

## 🎯 Model Results

| Metric | Score |
|---|---|
| **AUC-ROC** | **99.51%** |
| **Accuracy** | **98.43%** |
| **Recall** | **93.60%** |
| **F1 Score** | **40.81%** |
| **Precision** | **26.09%** |
| Training Samples | 1,296,675 |
| Fraud Cases | 7,506 (0.58%) |
| Features Engineered | 19 |

> **Why Recall matters most:** In fraud detection, missing a real fraud is far more costly than a false alarm. 93.60% recall means the model catches 93.6 out of every 100 real fraud transactions.

---

## 🗂️ Project Structure

```
FraudShield-AI/
│
├── app.py                      # Streamlit web app (5 tabs)
├── train_pipeline.py           # Full ML training pipeline
│
├── fraud_model.pkl             # Trained Random Forest model
├── feature_columns.pkl         # 19 feature names
├── categories.pkl              # 14 transaction categories
├── label_encoders.pkl          # Fitted encoders for categorical features
├── model_metrics.pkl           # AUC, accuracy, F1, confusion matrix
│
├── shap_plot.png               # SHAP feature importance chart
├── eda_plots.png               # EDA charts from 1.29M transactions
├── roc_pr_curves.png           # ROC and Precision-Recall curves
├── powerbi_data_sample.csv     # Sample 1000 rows for Power BI preview
│
├── dataset/                    # Place raw CSV here for retraining
│   └── .gitkeep
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes large files
└── README.md                   # This file
```

> **Note:** `powerbi_data.csv` (286MB full export) and raw `dataset/` files are excluded from GitHub due to size limits. See instructions below.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FraudShield-AI.git
cd FraudShield-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. (Optional) Retrain from scratch
```bash
# Place dataset at:
# dataset/credit_card_transactions.csv
# Then run:
python train_pipeline.py
```

---

## 📊 App Features — 5 Tabs

| Tab | Content |
|---|---|
| 🔍 **Fraud Checker** | Real-time prediction, risk gauge, probability bars, risk signals |
| 📊 **Analytics** | Fraud by category, hour, amount distribution, day of week |
| 🧠 **Model Insights** | Confusion matrix, ROC curve, PR curve, SHAP, feature importance |
| 📈 **EDA** | Full exploratory data analysis from 1.29M transactions |
| 📋 **History** | Session transaction history with risk trend chart |

---

## ⚙️ Feature Engineering (19 Features)

| Feature | Description |
|---|---|
| `amt` | Transaction amount |
| `hour` | Hour of transaction (0–23) |
| `day` | Day of month |
| `month` | Month |
| `day_of_week` | Day of week (0=Mon) |
| `age` | Cardholder age (derived from DOB) |
| `distance` | Cardholder–merchant distance (lat/long delta) |
| `category_enc` | Encoded merchant category |
| `gender_enc` | Encoded gender |
| `city_enc` | Encoded city |
| `state_enc` | Encoded state |
| `job_enc` | Encoded job title |
| `merchant_enc` | Encoded merchant name |
| `city_pop` | City population |
| `zip` | ZIP code |
| `lat / long` | Cardholder coordinates |
| `merch_lat / merch_long` | Merchant coordinates |

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Max Depth | 20 |
| Class Imbalance | RandomUnderSampler (strategy=0.5) |
| Class Weight | balanced |
| Train/Test Split | 80/20 stratified |
| Explainability | SHAP TreeExplainer |

---

## 📊 Power BI Dashboard

Import `powerbi_data_sample.csv` (or full `powerbi_data.csv`) into Power BI.

**Recommended visuals:**
1. **KPI Cards** — Total Transactions, Fraud Count, Fraud Rate, Avg Fraud Amount
2. **Bar Chart** — Fraud Cases by Category
3. **Line Chart** — Fraud Transactions by Hour
4. **Donut Chart** — Fraud by Gender
5. **Histogram** — Fraud Amount Distribution
6. **Slicers** — Category, State, Risk Level, Gender, Month

**Key columns for Power BI:**
- `fraud_prob` — model fraud probability (0–1)
- `risk_level` — Low / Medium / High
- `fraud_label` — Legitimate / Fraudulent
- `predicted_label` — model's prediction label

---

## 📁 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection by kartik2112](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

- 1,296,675 transactions
- Time period: 2019–2020
- 23 original features

To retrain, download and place at `dataset/credit_card_transactions.csv`

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
plotly>=5.15.0
shap>=0.43.0
matplotlib>=3.7.0
```

---

## 👤 Author

**Your Name**:Alekhya Viswanadhapalli
- 📧 Email:viswanadhapallialekhya13@gmail.com
