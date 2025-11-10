# Assignment 8 (Ensemble Learning for Complex Regression Modeling)

## Student Info
**Name:** Shashank Satish Adsule  
**Roll No.:** DA25M005  

## Dataset Used
- **Dataset:** [Bike Sharing Demand Dataset (Hourly)](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- **Source:** UCI Machine Learning Repository  
- **Files:** `hour.csv`  
- **Size:** ~17,000 hourly records  
- **Target Variable:** `cnt` total count of rented bikes (continuous regression target)

### Dataset Features
The dataset contains information about hourly bike rentals and environmental conditions, including:
- **Temporal features:** season, month, hour, weekday, workingday, holiday  
- **Weather features:** temperature (`temp`), adjusted temperature (`atemp`), humidity (`hum`), windspeed (`windspeed`)  
- **Target variable:** total rental count (`cnt`)

## Objective

The objective of this assignment is to apply and compare three major **ensemble learning techniques** to solve a **complex regression problem**:

1. **Bagging (Bootstrap Aggregating)** – to reduce model variance  
2. **Boosting (Gradient Boosting)** – to reduce model bias  
3. **Stacking (Meta-Ensemble)** – to optimally combine diverse base learners  

The goal is to evaluate how these ensemble techniques improve generalization compared to baseline models (**Linear Regression** and **Decision Tree Regressor**).

## Models Implemented

A total of **five regression models** were trained and compared:

| Model No. | Algorithm | Type | Purpose |
|:--:|:--|:--|:--|
| 1 | **Linear Regression** | **Baseline** | Linear model with high bias and low variance |
| 2 | **Decision Tree Regressor** | Benchmark | Non-linear model with low bias and high variance |
| 3 | **Bagging (Decision Tree base)** | Ensemble | Averages multiple high-variance trees to reduce variance |
| 4 | **Gradient Boosting Regressor** | Ensemble | Reduces bias via sequential residual correction |
| 5 | **Stacking Regressor (KNN + Bagging + GBR, meta: Ridge)** | Meta-Ensemble | Combines multiple models for optimal bias–variance balance |


## Methodology

### Part A: Data Preprocessing and Baseline
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`
- Split data into **train (70%)** and **test (30%)**
- Trained baseline models:
  - **Linear Regression (with standardization)**  
  - **Decision Tree Regressor** (`max_depth=6`) for non-linear benchmarking  
- The **Linear Regression** model performed better on the test set and was selected as the primary **baseline reference**.


### Part B: Ensemble Techniques

1. **Bagging Regressor (Decision Tree base)**  
   - Targets **variance reduction** by training multiple Decision Trees on bootstrapped samples.  
   - Averaging their predictions stabilizes variance and reduces overfitting.  
   - However, because the dataset’s relationships are largely **linear and smooth**, Bagging’s variance reduction had limited impact resulting in slightly **worse performance than Linear Regression**.

2. **Gradient Boosting Regressor**  
   - Targets **bias reduction** through sequential learning.  
   - Each tree learns from the residuals of the previous model, progressively capturing non-linear relationships.  
   - This iterative correction significantly improved model accuracy, lowering both RMSE and bias.


### Part C: Stacking for Optimal Performance
- **Base Learners (Level-0):**  
  `KNN`, `Bagging (Decision Trees)`, `Gradient Boosting Regressor`
- **Meta-Learner (Level-1):**  
  `Ridge Regression`
- Stacking combines the strengths of all base learners — the **low bias** of Boosting, **variance control** of Bagging, and **local adaptivity** of KNN leading to the **best overall generalization**.


## Evaluation Metrics

For all models, the following regression metrics were computed:

| Metric | Description | Optimization |
|:--|:--|:--|
| **RMSE (Root Mean Squared Error)** | Measures prediction deviation magnitude | Lower = Better |
| **R² (Coefficient of Determination)** | Measures variance explained by the model | Higher = Better |
| **MAPE (Mean Absolute Percentage Error)** | Measures percentage deviation between predictions and true values | Lower = Better |


## Model Performance Summary

| Model | RMSE ↓ | R² ↑ | MAPE ↓ |
|--------|--------|------|-------------|
| **Decision Tree Regressor** | 119.84 | 0.5329 | 4.4128 |
| **Linear Regressor** | 100.02 | 0.6747 | 2.9407 |
| **Bagging (Decision Tree)** | 114.52 | 0.5732 | 4.365 |
| **Gradient Boosting Regressor** | 78.39 | 0.8001 | 1.6557 |
| **Stacking Regressor** | **51.709** | **0.9130** | **0.6668** |

![model_eval](./assests/model_eval.png)

> **Best Model:** *Stacking Regressor* achieved the lowest RMSE and MAPE, and the highest R² score.  
> **Observation:** *Linear Regression* slightly outperformed *Bagging (Decision Trees)*, showing that the dataset’s dominant relationships are mostly linear.

## Bias–Variance Trade-off Discussion

| Model | Bias | Variance | Learning Pattern |
|:--|:--|:--|:--|
| **Linear Regression (Baseline)** | **High Bias** | **Low Variance** | Captures linear patterns but underfits non-linear data |
| **Decision Tree Regressor** | **Low Bias** | **High Variance** | Learns complex, non-linear relationships but prone to overfitting |
| **Bagging (Decision Trees)** | **Low Bias** | **Reduced Variance** | Averages multiple trees, stabilizing predictions but limited gain due to dataset linearity |
| **Gradient Boosting Regressor** | **Reduced Bias** | Moderate Variance | Sequentially corrects residuals, capturing complex non-linear trends |
| **Stacking Regressor** | **Balanced** | **Balanced** | Combines model diversity for optimal bias–variance trade-off |

- **Bagging** reduced the variance of Decision Trees but showed **no major improvement** since the dataset’s variance was already low.  
- **Boosting** significantly reduced bias and captured more complex trends.  
- **Stacking** achieved the best performance by integrating the complementary strengths of all models.


## Python Dependencies

```bash
pandas                  # data manipulation and analysis
numpy                   # numerical computation
matplotlib              # visualization
seaborn                 # enhanced plotting
scikit-learn            # core ML framework
    ├── model_selection (train_test_split)
    ├── metrics (mean_squared_error, r2_score)
    ├── linear_model (LinearRegression, Ridge)
    ├── ensemble (BaggingRegressor, GradientBoostingRegressor, StackingRegressor)
    ├── neighbors (KNeighborsRegressor)
```

## Conclusion

- The **Stacking Regressor** achieved the **best generalization performance** among all models.
- It outperformed both baseline and other ensembles by effectively balancing bias and variance.
- The Ridge meta-learner successfully learned how to weight KNN, Bagging, and Boosting predictions optimally.
- Ensemble diversity and bias–variance optimization are key to its superior predictive power.
  
> **Stacking Regressor** is the best model for this regression problem, achieving the lowest prediction error and highest explanatory power.

sumbition link: https://docs.google.com/forms/d/e/1FAIpQLSdyUZz9ugLIqIMfUipxoCCOuNmJ-Kml6O0TGxU8N7RJgHvR_A/viewform
