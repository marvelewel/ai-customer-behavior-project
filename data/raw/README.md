# Raw Data Directory

This directory contains the original, unmodified datasets.

## ⚠️ Important
- **DO NOT** modify files in this directory
- Raw data should remain unchanged for reproducibility
- All transformations should output to `data/processed/`

---

## 📥 Data Source

> **Original Dataset:** [E-commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset) on Kaggle
> 
> This dataset is synthetically generated for illustrative purposes. Any resemblance to real individuals or scenarios is coincidental.

---

## 📄 Files

| File | Description | Size |
|------|-------------|------|
| `E-commerce Customer Behavior - Sheet1.csv` | Original customer behavior dataset | 350 rows × 11 columns |

---

## 📊 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Customer ID | Numeric | Unique identifier assigned to each customer |
| Gender | Categorical | Gender of the customer (Male / Female) |
| Age | Numeric | Age of the customer (26-43 years) |
| City | Categorical | City of residence for each customer |
| Membership Type | Categorical | Type of membership (Gold / Silver / Bronze) |
| Total Spend | Numeric | Total monetary expenditure ($410 - $1,520) |
| Items Purchased | Numeric | Total number of items purchased (7-21) |
| Average Rating | Numeric | Average rating given for purchased items (3.0 - 4.9) |
| Discount Applied | Boolean | Whether a discount was applied (TRUE / FALSE) |
| Days Since Last Purchase | Numeric | Days since most recent purchase (9-63 days) |
| Satisfaction Level | Categorical | Overall satisfaction (Satisfied / Neutral / Unsatisfied) |

---

## 💡 Potential Use Cases

- **Customer Segmentation** - Categorize customers based on demographics, spending habits, and satisfaction
- **Satisfaction Analysis** - Investigate factors influencing customer satisfaction
- **Promotion Strategy** - Assess impact of discounts on customer spending
- **Retention Strategies** - Develop targeted retention based on purchase recency
- **Geographic Insights** - Explore regional variations in customer behavior
