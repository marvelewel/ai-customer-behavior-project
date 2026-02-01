# data/processed/README.md
# Processed Data Directory

This directory contains cleaned and transformed datasets ready for analysis.

## 📄 Files

| File | Description | Created By |
|------|-------------|------------|
| `customer_cleaned.csv` | Cleaned data with fixed types | `02_data_preprocessing.ipynb` |
| `customer_features.csv` | Data with engineered features | `02_data_preprocessing.ipynb` |
| `customer_clustered.csv` | Data with cluster assignments | `04_clustering.ipynb` |
| `customer_final_insights.csv` | Complete data with all insights | `06_ai_decision_layer.ipynb` |
| `segment_strategies.csv` | Strategy recommendations per segment | `06_ai_decision_layer.ipynb` |

## 🔄 Data Pipeline

```
raw/E-commerce Customer Behavior.csv
    │
    ▼ (cleaning)
customer_cleaned.csv
    │
    ▼ (feature engineering)
customer_features.csv
    │
    ▼ (clustering)
customer_clustered.csv
    │
    ▼ (AI insights)
customer_final_insights.csv
```

## 📝 Notes

- These files are regenerated when running the notebooks
- Do not edit manually - rerun notebooks instead
- Used for faster analysis without reprocessing
