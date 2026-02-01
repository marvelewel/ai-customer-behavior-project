# data/raw/README.md
# Raw Data Directory

This directory contains the original, unmodified datasets.

## ⚠️ Important
- **DO NOT** modify files in this directory
- Raw data should remain unchanged for reproducibility
- All transformations should output to `data/processed/`

## 📄 Files

| File | Description | Size |
|------|-------------|------|
| `E-commerce Customer Behavior - Sheet1.csv` | Original customer behavior dataset | 350 rows × 11 columns |

## 📊 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Customer ID | Integer | Unique customer identifier |
| Gender | String | Male / Female |
| Age | Integer | Customer age (26-43) |
| City | String | Customer location |
| Membership Type | String | Gold / Silver / Bronze |
| Total Spend | Float | Total spending amount ($410-$1520) |
| Items Purchased | Integer | Number of items bought (7-21) |
| Average Rating | Float | Average product rating (3.0-4.9) |
| Discount Applied | Boolean | Whether discount was used |
| Days Since Last Purchase | Integer | Recency in days (9-63) |
| Satisfaction Level | String | Satisfied / Neutral / Unsatisfied |
