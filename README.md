# [DSC630] Predicting Beer Strength (ABV) from Sensory & Beer Characteristics

## Overview
This project builds and compares regression models to predict **beer strength (ABV)** using **sensory review scores** and **beer characteristics**. Multiple predictor variables—such as **Style**, **Body**, **Malty**, and **IBU range**—are used to model ABV. Two regression models are compared:
- **Random Forest Regression** (non-linear ensemble approach)
- **Lasso Regression** (regularized linear approach)

## Dataset
Dataset: *Beer Profiles and Ratings* (Ruthgn, 2022) on Kaggle.  
The dataset contains ~3,200 beer entries aggregated from beer review sources.

**Sensory variables (examples):**
- Malty, Hoppy, Bitter, Sweet, Sour, Body
- (Excluded to prevent leakage): Alcohol (perceived alcohol presence)

**Beer characteristics:**
- Style (categorical)
- Min IBU, Max IBU (continuous)

## Target Variable & Feature Selection
- **Target (y):** `ABV`
- **Predictors (X):**
  - Numeric: `Min IBU`, `Max IBU`, `Body`, `Bitter`, `Hoppy`, `Malty`
  - Categorical: `Style`

### Leakage prevention decisions
- `ABV` is used **only** as the target and removed from predictors.
- The `Alcohol` sensory score is **excluded** because it directly correlates with measured ABV and would constitute data leakage when predicting ABV.

## Target Variable Transformation
ABV exhibited right skew (approximately +1.2). Two transformations were evaluated:
- **Log transform:** reversed skewness (−1.2) and was deemed unsuitable.
- **Square-root transform:** reduced skewness to approximately −0.034, producing a near-normal distribution that improves linear model assumptions and helps stabilize variance for tree-based models.

For interpretability, all final predictions and error metrics are **back-transformed** to the original ABV percentage scale (by squaring) after modeling on the transformed scale.

## Methods of Analysis
This is a supervised regression problem (continuous target prediction).

### Model 1 — Lasso Regression
Lasso regression (alpha = 0.1) serves as the linear baseline. L1 regularization shrinks less-important coefficients toward zero, effectively performing feature selection and highlighting which sensory attributes contribute most to ABV prediction.

### Model 2 — Random Forest Regressor
A Random Forest Regressor with **100 estimators** was selected to capture non-linear relationships and feature interactions without requiring explicit feature engineering.

Random Forest is appropriate because:
1. It handles categorical variables well via tree-based splits (the beer `Style` has dozens of categories).
2. It reduces noise by averaging predictions across many trees (useful given reviewer subjectivity in sensory ratings).

## Preprocessing Pipeline
A scikit-learn `ColumnTransformer` was used to handle mixed feature types within each model pipeline:
- **Numeric features:** standardized with `StandardScaler`
- **Categorical feature:** `Style` one-hot encoded with `handle_unknown='ignore'`

## Validation Strategy
- **Train/Test Split:** 80/20 with fixed random state (**42**) for reproducibility
- **Cross-Validation:** 5-fold cross-validation on the training set for both models

## Evaluation Metrics
Models were evaluated using:
- **RMSE**: penalizes larger errors more strongly (same units as ABV after back-transformation)
- **MAE**: average absolute difference between predicted and actual ABV values (more directly interpretable)

## Data Preparation
The preprocessing pipeline used a multi-step approach:

1. **Initial audit:** columns separated into categorical vs. continuous; frequency counts (categorical) and summary stats (continuous).
2. **Standardizing categorical values:** `"India Pale Ale"` and `"IPA"` combined into `"IPA"`.
3. **Managing unrealistic ABV values:** ABV max of 57.5% capped at 20% to prevent distortion (3 records affected).
4. **Imputing placeholder zeros:** implausible zeros replaced with the **median** of non-zero values for each affected column.
5. **Dropping the `Salty` column:** removed due to near-zero variance (1,895 zeros out of ~3,200).
6. **Removing invalid records:** rows where `ABV == 0` removed (12 records).
7. **Grouping rare styles:** styles appearing fewer than 5 times grouped into `"Other"` (resulting in 109 unique styles).
8. **Outlier capping (99th percentile Winsorization):** applied to `Alcohol`, `Body`, `Astringency`, `Malty`, `Sweet`, and `number_of_reviews`.

## Results & Findings
### Cross-Validation Performance (Square-root transformed target)
5-fold cross-validation was used to compare model performance on the **square-root transformed ABV scale**.

**Table 1 — 5-fold CV (sqrt-transformed scale)**  
Random Forest outperformed Lasso Regression with lower errors.

| Model | MAE (± SD) | RMSE (± SD) |
|---|---:|---:|
| Lasso Regression | 0.2884 (±0.0128) | 0.4041 (±0.0232) |
| Random Forest | 0.1699 (±0.0060) | 0.2426 (±0.0121) |

### Additional sqrt-scale results
**Table 2 — Random Forest vs Lasso (sqrt-transformed scale)**

| Model | MAE | RMSE |
|---|---:|---:|
| Lasso Regression | 0.2727 | 0.3783 |
| Random Forest | 0.1602 | 0.2245 |

### Performance on original ABV percentage scale
After back-transforming predictions to the original ABV percentage scale:

**Table 3 — Original ABV scale**

| Model | MAE | RMSE | Prediction Error |
|---|---:|---:|---|
| Lasso Regression | 1.40% ABV | 1.96% ABV | ±1.40% ABV |
| Random Forest | 0.84% ABV | 1.22% ABV | ±0.84% ABV |

### Interpretation (Actual vs Predicted + by style)
The **Actual vs. Predicted** scatter plot shows Random Forest predictions cluster more tightly around the perfect prediction line than Lasso. Lasso underestimates at the high end of ABV, consistent with limitations of linear-model assumptions.

Prediction accuracy varied across beer styles:
- **High-variance styles** (e.g., strong ales, barleywines) were harder to predict.
- **Lower-variance styles** (e.g., standard lagers, wheat beers) were predicted more accurately.

This suggests that style-specific modeling or additional features (e.g., **original gravity**) could improve performance for high-variance categories.

### Features learned by the model
The selected features—**IBU range, body, bitterness, hoppiness, and maltiness**—align with brewing science:
- Higher ABV beers require more fermentable sugars (malt),
- which often produces a fuller body,
- and typically involves more hops to balance sweetness.
The model appears to have learned these relationships from the dataset.

## Conclusion
A **Random Forest ensemble** reliably predicted **beer ABV** from sensory and style data, outperforming linear regression. Results indicate that **complex, non-linear interactions** among attributes (e.g., body, maltiness, bitterness, hoppiness) drive ABV prediction. Performance was robust under 5-fold cross-validation, but accuracy varied by beer style—especially for high-variance styles such as strong ales and barleywines.

## How to Run

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter

Start Jupyter Notebook from the project root (the folder that contains beer_profile_and_ratings.csv).
Open and run: notebooks/Spalding_DSC630_FinalCode.ipynb
