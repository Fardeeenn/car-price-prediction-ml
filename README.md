# Car Price Prediction

End-to-end machine learning pipeline that predicts the resale price of used cars
in India, built on real listings data from CarDekho. Covers the full workflow:
data cleaning, exploratory data analysis, feature engineering, and model building.

## Overview

| | |
|---|---|
| **Problem type** | Regression |
| **Target variable** | Selling price (INR) |
| **Dataset size** | ~15,400 listings, 14 raw features |
| **Source** | [CarDekho](https://www.cardekho.com/) used car listings, via Kaggle |
| **Status** | Data pipeline complete — modeling in progress |

## Project structure

├── data/ raw and processed datasets
├── notebooks/ analysis and modeling notebook
├── visualizations/ exported charts
└── README.md


## Methodology

### 1. Data Cleaning
Raw data had duplicate records, inconsistent category labels, and several
data-entry errors. Key fixes:

- Removed 167 duplicate rows and a redundant index column
- Standardized inconsistent brand/model labels (e.g. `ISUZU` → `Isuzu`)
- Corrected invalid values (0-seat listings, a hybrid car mislabeled "Electric")
- Removed implausible entries (odometer readings in the millions, a price
  value mistakenly entered into the mileage field, a 29-year-old listing)

**Result:** 15,411 → 15,227 rows, zero missing values, zero duplicates.

### 2. Exploratory Data Analysis
Univariate, bivariate, and multivariate analysis to understand price drivers.

**Key findings:**
- Selling price is heavily right-skewed (skew ≈ 10), driven by a small number
  of luxury listings — log-transforming the target reduces skew to ≈ 0.57
- `max_power` (r = 0.75) and `engine` (r = 0.59) are the strongest numeric
  predictors; `km_driven` alone is weak (r = −0.12)
- Automatic transmission carries roughly a 2x price premium over manual,
  consistent across vehicle age — not just a byproduct of automatics being newer
- A handful of "top-priced" brands (Ferrari, Rolls-Royce, Bentley) have only
  1–3 listings each, too few to represent a reliable brand-level trend
- CNG/LPG vehicles cluster almost entirely in the low-power range, suggesting
  fuel type partly proxies for vehicle segment rather than acting independently

| | |
|---|---|
| ![Price distribution](visualizations/price_distribution.png) | ![Correlation heatmap](visualizations/correlation_heatmap.png) |

Full chart set available in [`visualizations/`](visualizations/).

### 3. Feature Engineering
All new features are derived from input columns only, to avoid target leakage.

| Feature | Description |
|---|---|
| `log_price` | Log-transformed target; corrects severe skew |
| `log_km_driven` | Log-transformed mileage; reduces skew |
| `brand_grouped` | Brands with <10 listings merged into `Other` |
| `fuel_type_grouped` | Hybrid and LPG (low sample count) merged into `Other` |
| `km_per_year` | Usage intensity (`km_driven / vehicle_age`); found to be partly confounded by age itself — older cars sell for less regardless of how lightly they were used |

### 4. Modeling — *in progress*
- [ ] Train/test split and preprocessing pipeline
- [ ] Baseline model: Linear Regression
- [ ] Random Forest, Gradient Boosting
- [ ] Evaluation: R², MAE, RMSE with cross-validation
- [ ] Hyperparameter tuning
- [ ] Feature importance analysis

## Tech Stack

`Python` · `pandas` · `NumPy` · `matplotlib` · `seaborn` · `scikit-learn`

## Running locally

```bash
git clone https://github.com/Fardeeenn/car-price-prediction-ml.git
cd car-price-prediction-ml
pip install -r requirements.txt
jupyter notebook notebooks/01_cardekho_price_analysis.ipynb
```

## Author

**Fardeen** — [GitHub](https://github.com/Fardeeenn)