# outputs/README.md
# Output Files

This directory contains all generated outputs from the analysis.

## 📁 Structure

```
outputs/
├── figures/     # Visualization images (PNG)
└── models/      # Trained ML models (PKL)
```

## 📊 Figures

| File | Description | Notebook |
|------|-------------|----------|
| `01_age_distribution.png` | Age histogram | 01 |
| `02_spend_distribution.png` | Spending histogram | 01 |
| `03_categorical_distributions.png` | Category bar charts | 01 |
| `04_correlation_matrix.png` | Feature correlations | 01 |
| `05_pairplot_membership.png` | Pairplot by membership | 01 |
| `06_spend_by_segments.png` | Box plots by segment | 01 |
| `07_recency_satisfaction.png` | Recency vs satisfaction | 01 |
| `08_engineered_features.png` | New feature distributions | 02 |
| `09_correlation_detailed.png` | Detailed correlations | 03 |
| `10_anova_membership.png` | ANOVA test results | 03 |
| `11_chisquare_satisfaction.png` | Chi-square visualization | 03 |
| `12_regression_analysis.png` | Regression plots | 03 |
| `13_optimal_k.png` | Elbow & silhouette plots | 04 |
| `14_cluster_profiles.png` | Cluster characteristics | 04 |
| `15_cluster_scatter.png` | 2D cluster visualization | 04 |
| `16_model_comparison.png` | Model accuracy comparison | 05 |
| `17_confusion_matrix.png` | Best model confusion matrix | 05 |
| `18_feature_importance.png` | Feature importance chart | 05 |
| `19_risk_dashboard.png` | Risk assessment dashboard | 06 |

## 🤖 Models

| File | Description | Type |
|------|-------------|------|
| `kmeans_model.pkl` | K-Means clustering model | Clustering |
| `random_forest_model.pkl` | Random Forest classifier | Classification |
| `best_classifier.pkl` | Best performing classifier | Classification |
| `clustering_scaler.pkl` | Scaler for clustering | Preprocessing |
| `classification_scaler.pkl` | Scaler for classification | Preprocessing |
| `label_encoder.pkl` | Label encoder for target | Preprocessing |

## 🔄 Regenerating Outputs

To regenerate all outputs, run all notebooks in sequence (01 → 06).
