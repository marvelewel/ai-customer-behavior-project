# 🧠 AI-Driven Customer Behavior Analysis System

> **End-to-end AI System untuk Analisis Perilaku Pelanggan E-commerce**  
> Customer Segmentation • Satisfaction Prediction • AI-Powered Recommendations

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [AI Architecture](#ai-architecture)
- [Results & Insights](#results--insights)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

---

## 🎯 Overview

Project ini membangun **AI Decision-Support System** yang:

1. **Menganalisis** perilaku pelanggan e-commerce
2. **Mengelompokkan** pelanggan ke dalam segment (unsupervised learning)
3. **Memprediksi** tingkat kepuasan pelanggan (classification)
4. **Menghasilkan** insight dan rekomendasi bisnis otomatis (AI decision layer)

> ⚠️ **Ini bukan sekadar ML modeling, tetapi AI decision-support system lengkap.**

---

## ❓ Problem Statement

### Business Questions:
1. Apa pola perilaku pelanggan yang berbeda dalam dataset?
2. Bagaimana cara mengidentifikasi pelanggan berisiko churn?
3. Fitur apa yang paling mempengaruhi kepuasan pelanggan?
4. Strategi apa yang harus diterapkan untuk setiap segmen pelanggan?

### AI Solutions:
- **Pattern Recognition** → K-Means Clustering
- **Prediction Engine** → Classification Models
- **Decision Engine** → Rule-based AI Recommendations

---

## 📊 Dataset

### 📥 Source

> **Original Dataset:** [E-commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset) on Kaggle
> 
> Dataset Credit: This dataset is sourced from Kaggle and is synthetically generated for illustrative purposes.

**Size:** 350 customers × 11 features

### 📋 Overview

This dataset provides a comprehensive view of customer behavior within an e-commerce platform. Each entry corresponds to a unique customer, offering a detailed breakdown of their interactions and transactions. The information facilitates nuanced analysis of customer preferences, engagement patterns, and satisfaction levels.

### 📝 Column Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| Customer ID | Numeric | Unique identifier assigned to each customer |
| Gender | Categorical | Gender of the customer (Male / Female) |
| Age | Numeric | Age of the customer (26-43 years) |
| City | Categorical | City of residence (6 major US cities) |
| Membership Type | Categorical | Type of membership (Gold / Silver / Bronze) |
| Total Spend | Numeric | Total monetary expenditure ($410 - $1,520) |
| Items Purchased | Numeric | Total number of items purchased (7-21) |
| Average Rating | Numeric | Average rating given for purchased items (3.0 - 4.9) |
| Discount Applied | Boolean | Whether a discount was applied (TRUE / FALSE) |
| Days Since Last Purchase | Numeric | Days since most recent purchase (9-63 days) |
| Satisfaction Level | Categorical | Overall satisfaction (Satisfied / Neutral / Unsatisfied) |

### 💡 Potential Use Cases

- **Customer Segmentation** - Categorize customers based on demographics, spending habits, and satisfaction
- **Satisfaction Analysis** - Investigate factors influencing customer satisfaction
- **Promotion Strategy** - Assess impact of discounts on customer spending
- **Retention Strategies** - Develop targeted retention based on purchase recency
- **Geographic Insights** - Explore regional variations in customer behavior

> ⚠️ **Note:** This dataset is synthetically generated for illustrative purposes. Any resemblance to real individuals or scenarios is coincidental.

---

## 🔬 Methodology

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  01 Data        │ ──► │  02 Data        │ ──► │  03 Statistical │
│  Exploration    │     │  Preprocessing  │     │  Analysis       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  06 AI Decision │ ◄── │  05 Classification│ ◄── │  04 Clustering │
│  Layer          │     │  Model          │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Workflow:

1. **Data Exploration** - EDA, visualizations, initial insights
2. **Preprocessing** - Cleaning, encoding, feature engineering
3. **Statistical Analysis** - Hypothesis testing, correlation analysis
4. **Clustering** - K-Means segmentation, segment profiling
5. **Classification** - Satisfaction prediction, model comparison
6. **AI Decision Layer** - Recommendations, executive summary

---

## 🏗️ AI Architecture

```
                    ┌──────────────────────────────────────┐
                    │         AI DECISION ENGINE           │
                    │                                      │
                    │  ┌────────────┐  ┌────────────────┐ │
                    │  │  Cluster   │  │  Classification │ │
                    │  │  Results   │  │   Predictions   │ │
                    │  └─────┬──────┘  └───────┬────────┘ │
                    │        │                  │          │
                    │        ▼                  ▼          │
                    │  ┌────────────────────────────────┐ │
                    │  │     REASONING LAYER            │ │
                    │  │  • Segment Interpretation      │ │
                    │  │  • Risk Assessment             │ │
                    │  │  • Strategy Generation         │ │
                    │  └─────────────┬──────────────────┘ │
                    │                │                     │
                    │                ▼                     │
                    │  ┌────────────────────────────────┐ │
                    │  │  RECOMMENDATION OUTPUT         │ │
                    │  │  • Individual Actions          │ │
                    │  │  • Segment Strategies          │ │
                    │  │  • Executive Summary           │ │
                    │  └────────────────────────────────┘ │
                    └──────────────────────────────────────┘
```

### Key AI Functions:

| Function | Purpose |
|----------|---------|
| `interpret_cluster()` | Menjelaskan karakteristik setiap segment |
| `generate_recommendation()` | Menghasilkan aksi untuk setiap pelanggan |
| `generate_segment_strategy()` | Strategi bisnis per segment |
| `generate_executive_summary()` | Ringkasan untuk stakeholders |

---

## 📈 Results & Insights

### Customer Segments Identified:

| Segment | Persona | Size | Priority |
|---------|---------|------|----------|
| Cluster 0 | VIP Champions | High spenders, very satisfied | High Value |
| Cluster 1 | Regular Customers | Medium engagement | Potential Value |
| Cluster 2 | Hibernating | Inactive, at-risk | Retention Focus |

### Model Performance:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~85% | ~84% | ~85% | ~84% |
| Logistic Regression | ~78% | ~77% | ~78% | ~77% |
| Decision Tree | ~80% | ~79% | ~80% | ~79% |

### Key Insights:

1. **Gold members** spend 2-3x more than Bronze members
2. **Satisfaction** strongly correlates with spending behavior
3. **Recency** is the strongest churn indicator
4. **AI-identified** high-risk customers enable proactive intervention

---

## 🚀 How to Run

### Prerequisites:
```bash
# Python 3.8+ required
python --version
```

### Installation:
```bash
# Clone repository
cd /path/to/ai-customer-behavior-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks:
```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. 01_data_exploration.ipynb
# 2. 02_data_preprocessing.ipynb
# 3. 03_statistical_analysis.ipynb
# 4. 04_clustering.ipynb
# 5. 05_classification_model.ipynb
# 6. 06_ai_decision_layer.ipynb
```

---

## 📁 Project Structure

```
ai-customer-behavior-project/
│
├── 📂 data/
│   ├── raw/                    # Original dataset (DO NOT MODIFY)
│   │   └── E-commerce Customer Behavior - Sheet1.csv
│   └── processed/              # Cleaned & engineered data
│       ├── customer_cleaned.csv
│       ├── customer_features.csv
│       └── customer_final_insights.csv
│
├── 📂 notebooks/               # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_classification_model.ipynb
│   └── 06_ai_decision_layer.ipynb
│
├── 📂 src/                     # Reusable Python modules
│   ├── preprocessing.py        # Data cleaning functions
│   ├── features.py             # Feature engineering
│   ├── models.py               # ML model functions
│   └── ai_logic.py             # AI Decision Engine
│
├── 📂 outputs/
│   ├── figures/                # Visualizations
│   └── models/                 # Saved ML models (.pkl)
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── venv/                       # Virtual environment
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Machine learning
- **joblib** - Model persistence

---

## 📝 Definition of Success

✅ **ML Implementation** - Clustering & Classification  
✅ **Automated Interpretation** - AI-generated insights  
✅ **AI-based Recommendations** - Actionable business strategies  
✅ **Non-technical Explanation** - Executive-ready reports  

---

## 👤 Author

AI-Driven Customer Behavior Analysis Project

---

> 🧠 *"This project demonstrates the transition from ML modeling to AI decision-support systems."*
