# notebooks/README.md
# Analysis Notebooks

This directory contains the Jupyter notebooks for the complete analysis pipeline.

## 📓 Notebook Sequence

Run notebooks in numerical order:

| # | Notebook | Purpose | Duration |
|---|----------|---------|----------|
| 01 | `01_data_exploration.ipynb` | EDA, visualizations, initial insights | ~5 min |
| 02 | `02_data_preprocessing.ipynb` | Data cleaning, feature engineering | ~3 min |
| 03 | `03_statistical_analysis.ipynb` | Hypothesis testing, correlations | ~5 min |
| 04 | `04_clustering.ipynb` | K-Means segmentation | ~5 min |
| 05 | `05_classification_model.ipynb` | Satisfaction prediction models | ~5 min |
| 06 | `06_ai_decision_layer.ipynb` | AI recommendations & insights | ~3 min |

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start Jupyter
jupyter notebook

# Run notebooks in order (01 → 06)
```

## 📝 Notes

- Each notebook can be run independently if processed data exists
- Outputs are saved to `outputs/figures/` and `outputs/models/`
- All notebooks import from `src/` module for reusability
