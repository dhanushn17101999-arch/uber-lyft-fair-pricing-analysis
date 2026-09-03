# Fairness-Aware Dynamic Pricing Analysis

## Project Overview

This project investigates the trade-off between profitability and fairness in algorithmic dynamic pricing using Uber and Lyft ride-pricing data from Boston, Massachusetts.

The analysis develops an Ordinary Least Squares (OLS) regression model to examine factors associated with ride prices and subsequently uses the fitted model to simulate different surge-pricing fairness constraints.

The analysis evaluates how increasingly restrictive surge caps affect:

- Predicted revenue
- Average ride price
- Route-level price variation
- The potential fairness implications of reduced price variation

The complete analytical workflow is implemented in Python.

---

## Dataset

The analysis uses the Uber and Lyft Boston ride-pricing dataset obtained from Kaggle.

**Original dataset:** 693,071 ride observations, 47 variables

**Cleaned dataset:** 538,845 observations, 9 variables

Variables included in analysis:
- Price
- Surge multiplier
- Distance
- Hour of day
- Temperature
- Cab type
- Source location
- Destination location

The cleaned dataset was generated locally but is not included in this repository due to file size (49.9 MB).

---

## Analytical Workflow

### Stage 1 — Data Cleaning
Processed original dataset using Python and Pandas:
- Loaded dataset and inspected structure
- Selected required variables
- Handled missing values
- Checked for duplicates and invalid values
- Converted variables to appropriate data types
- Result: 538,845 observations, 9 variables

### Stage 2 — Exploratory Data Analysis
Conducted EDA on cleaned dataset:
- Descriptive statistics (mean, median, SD, range)
- Distribution analysis (histograms, density plots)
- Scatter plots (continuous predictors vs price)
- Price variation by cab type and hour
- Correlation analysis
- Route-level analysis (72 unique routes)

### Stage 3 — Categorical Encoding
Converted categorical variables to dummy variables:
- Cab type (Uber vs Lyft reference)
- Hour of day (23 dummies, hour 0 reference)
- Source location (11 dummies, Back Bay reference)
- Destination location (11 dummies, Back Bay reference)

### Stage 4 — Train-Test Split
80:20 random split with seed = 42:
- Training set: 431,076 observations
- Test set: 107,769 observations

### Stage 5 — OLS Regression & Model Evaluation
Fitted OLS model with 49 predictors on training data.

**Model performance:**
- R² = 0.177
- Adjusted R² = 0.177
- F-statistic = 1932, p < 0.001

**Key coefficients:**
- Surge multiplier: β = $22.12 per unit (p < 0.001)
- Distance: β = $2.96 per mile (p < 0.001)
- Temperature: β = $0.0018 (p = 0.385, not significant)

**Test set evaluation:**
- Test R² = 0.1785
- RMSE = $8.70
- MAE = $7.16
- MAPE = 58.04%

**Diagnostic tests:**
- Breusch-Pagan: LM = 12,135.10 (p < 0.001) — Heteroscedastic
- Durbin-Watson: 2.01 — No autocorrelation
- Shapiro-Wilk: W = 0.959 (p < 0.001) — Non-normal residuals

### Stage 6 — Fairness Scenario Simulation & Hypothesis Testing
Simulated four surge-pricing scenarios on test data:

| Scenario | Surge Cap | Revenue | Revenue Loss | Variance Reduction |
|----------|-----------|---------|--------------|-------------------|
| Baseline | None | $1,863,330.89 | — | — |
| Cap 1.20 | 20% | $1,838,418.54 | 1.34% | 16.15% |
| Cap 1.10 | 10% | $1,829,264.58 | 1.83% | 16.15% |
| Cap 1.05 | 5% | $1,824,687.60 | 2.07% | 16.15% |

**Route-level fairness:**
- 71 of 72 routes experienced price variance reduction
- Mean price standard deviation: $4.08 (baseline) → $3.43 (Cap 1.20)
- Variance reduction range: 6.59%–20.44% across routes

**Hypothesis testing:**
- **H1 (Revenue Cost):** SUPPORTED — all scenarios showed revenue loss
- **H2 (Fairness Improvement):** SUPPORTED — robust 16.15% variance reduction
- **H3 (Net Strategic Value):** PROVISIONALLY SUPPORTED — fairness mechanism confirmed; retention gains inferred from literature

---

## Key Findings

**Surge Variation Limitation:**
Only 3.84% of observations had surge > 1.0; 96.16% at baseline surge = 1.0. This constrains empirical magnitude of surge-cap impacts.

**Price Determinants:**
- Surge multiplier: strongest predictor ($22.12 per unit)
- Distance: second-strongest predictor ($2.96 per mile)
- Location (source/destination): significant negative effects for university and Fenway locations
- Cab type: minimal effect ($0.07 for Uber)
- Hour: minimal effects (only hour 18 significant at −$0.18)
- Temperature: not significant

**Fairness-Profitability Trade-off:**
- Moderate surge caps (10% cap) produce modest revenue loss (1.83%)
- Fairness gains are robust and consistent (16.15% variance reduction)
- Revenue losses could be offset by retention gains (10–15% literature-based estimate)


---

## Large Files Not Included

The following large files were generated locally but are not included in this repository due to file size constraints:

- `cleaned_uber_lyft_dataset.csv` — 49.9 MB (Google sheet - https://docs.google.com/spreadsheets/d/1ZDChK6W3kM8trGr0kfWl1FXpmKbEq-lusYgJ3F0WqP4/edit?usp=sharing)
- `stage2_X_train.csv` — 48.1 MB
- Original raw dataset (Kaggle source - https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)

Python scripts and analytical outputs are included to provide transparency into the complete analytical workflow.

---

## Reproducibility

The analysis uses a fixed random seed of **42** for the train-test split to ensure reproducibility.

Complete analytical workflow:
1. Data cleaning (`clean_uber_lyft_data.py`)
2. Exploratory Data Analysis (`eda_analysis.py`)
3. Categorical encoding (`stage1_encoding.py`)
4. 80:20 train-test split (`stage2_train_test_split.py`)
5. OLS regression & evaluation (`stage3_ols_regression_and_evaluation.py`)
6. Fairness simulation & hypothesis testing (`stage4_fairness_hypothesis.py`)

To reproduce the analysis, run the Python scripts in order. Each script loads outputs from the previous stage and generates new outputs for the next stage.

---

## Technologies Used

- Python 3.x
- Pandas (data manipulation)
- NumPy (numerical computing)
- Statsmodels (OLS regression, diagnostics)
- Scikit-learn (train-test split, metrics)
- Matplotlib (visualizations)
- Seaborn (statistical graphics)
- SciPy (statistical tests)

---

## Project Purpose

This repository supports a Master's thesis in Business Analytics investigating fairness-aware dynamic pricing in ride-sharing platforms. It documents the complete analytical workflow from data cleaning through hypothesis testing, with emphasis on the trade-off between revenue optimization and price fairness.

---

## License

This project uses publicly available data from Kaggle (Uber and Lyft Boston dataset) under the terms of the Kaggle Data License.
