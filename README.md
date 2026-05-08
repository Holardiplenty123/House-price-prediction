# 🏠 House Price Prediction

Predicting residential property sale prices using numerical features, linear regression, and Ridge regularisation.

---

## Project Overview

This project builds a machine learning pipeline to predict house sale prices from physical and structural property measurements. The constraint was deliberate: numerical features only, no categorical encoding. Every cleaning and engineering decision is justified in the notebook.

The final model scores **RMSE 0.1287** on log(SalePrice), beating the target of 0.16.

---

## Dataset

- **Source:** Ames Housing Dataset
- **Training rows:** 1,460 (1,458 after outlier removal)
- **Test rows:** 1,459
- **Raw features:** 81 columns
- **Numerical features kept:** 38 (including target)


---

## Workflow

### Part 1 — Exploratory Data Analysis
- Filtered 81 columns down to 38 numerical features
- Computed Pearson correlations to identify top predictors
- Built correlation heatmap for top 10 features
- Created scatter plots for top 3 features vs SalePrice
- Analysed SalePrice distribution, skew = 1.88

**Top 5 features by correlation with SalePrice:**

| Feature | r |
|---|---|
| OverallQual | 0.791 |
| GrLivArea | 0.709 |
| GarageCars | 0.640 |
| GarageArea | 0.623 |
| TotalBsmtSF | 0.614 |

---

### Part 2 — Data Cleaning

**Missing values (column-specific strategy, no blanket fill):**

| Column | Missing | Strategy | Reason |
|---|---|---|---|
| LotFrontage | 259 | Median (68 ft) | Unmeasured, not absent |
| GarageYrBlt | 81 | 0 | No garage exists |
| MasVnrArea | 8 | 0 | No veneer present |
| Test set (rest) | Various | Train medians | Prevent data leakage |

**Multicollinearity resolved (pairs with r > 0.8):**

| Pair | r | Dropped |
|---|---|---|
| GarageArea vs GarageCars | 0.882 | GarageArea |
| TotalBsmtSF vs 1stFlrSF | 0.820 | 1stFlrSF |
| GrLivArea vs TotRmsAbvGrd | 0.826 | TotRmsAbvGrd |

**Outliers:**
- Rows 523 and 1298 dropped: houses over 4,600 sqft selling under $200K, almost certainly non-arm's-length transactions
- 5 houses with 4-car garages kept: legitimate luxury properties

**Scaling:**
StandardScaler applied, fitted on training data only, applied to test set to prevent leakage.

---

### Part 3 — Feature Engineering

Four new features created from domain knowledge:

| Feature | Formula | Why |
|---|---|---|
| TotalSF | TotalBsmtSF + GrLivArea | Total usable space |
| HouseAge | YrSold - YearBuilt | Depreciation at point of sale |
| TotalBathrooms | FullBath + 0.5 x HalfBath | Weighted bathroom count |
| YearsSinceRemodel | YrSold - YearRemodAdd | Freshness of finishes |

SalePrice log-transformed using `np.log1p()`, reducing skew from **1.88 to 0.12** (93.6% reduction).

---

### Part 4 — Modelling

5-fold cross-validation used throughout. No single train/test split.

**Linear Regression baseline:**

| Fold | RMSE |
|---|---|
| Fold 1 | 0.1328 |
| Fold 2 | 0.1191 |
| Fold 3 | 0.1408 |
| Fold 4 | 0.1326 |
| Fold 5 | 0.1188 |
| **Mean** | **0.1288** |
| Std | 0.0086 |

**Ridge Regression (alpha = 10, selected by RidgeCV):**

| Metric | Value |
|---|---|
| Mean RMSE | 0.1287 |
| Std RMSE | 0.0084 |
| Improvement vs Linear | +0.0001 |
| Target (< 0.16) | PASS |

Both models beat the success criterion of 0.16.

---

### Part 5 — Reflection

**Hardest decision:** Dropping rows 523 and 1298. Large houses at very low prices that would distort the GrLivArea regression slope for every other property in the test set.

**What the model gets wrong:**
- Underestimates luxury homes (neighbourhood, finish quality, all categorical, all excluded)
- Overestimates distressed properties (flood damage, structural issues, invisible in numerical columns)

**What comes next:**
1. Add categorical features via target encoding, expected to drop RMSE by 0.03 to 0.05
2. Try Lasso and ElasticNet for automatic feature selection
3. Train XGBoost or LightGBM to capture non-linear interactions

---

## Results

| Model | CV RMSE | Target | Status |
|---|---|---|---|
| Linear Regression | 0.1288 | < 0.16 | PASS |
| Ridge Regression | 0.1287 | < 0.16 | PASS |

**Submission prediction stats:**

| Metric | Value |
|---|---|
| Min | $52,996 |
| Median | $158,310 |
| Mean | $178,635 |
| Max | $1,745,834 |
