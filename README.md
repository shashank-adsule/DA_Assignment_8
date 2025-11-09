# Assignment 8 (Ensemble Learning for Complex Regression Modeling)

## Student Info
**Name:** Shashank Satish Adsule  
**Roll No.:** DA25M005  

## Dataset Used
- **Dataset:** [Bike Sharing Demand Dataset (Hourly)](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- **Source:** UCI Machine Learning Repository  
- **Files:** `hour.csv`  
- **Size:** ~17,000 hourly records  
- **Target Variable:** `cnt` — total count of rented bikes (continuous regression target)

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

The goal is to evaluate how these ensemble techniques improve generalization compared to the **baseline Linear Regression model**.

## Models Implemented

A total of **five regression models** were trained and compared:

| Model No. | Algorithm | Type | Purpose |
|:--:|:--|:--|:--|
| 1 | Decision Tree Regressor | Benchmark | Non-linear model reference |
| 2 | **Linear Regression** | **Baseline** | Linear model with high bias and low variance |
| 3 | **Bagging (Linear Regression base)** | Ensemble | Reduces variance slightly by averaging multiple linear models |
| 4 | **Gradient Boosting Regressor** | Ensemble | Reduces bias via sequential residual correction |
| 5 | **Stacking Regressor (KNN + Bagging + GBR, meta: Ridge)** | Meta-Ensemble | Combines bias and variance reduction for optimal generalization |

## Methodology

### Part A: Data Preprocessing and Baseline
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`
- Split data into **train (70%)** and **test (30%)**
- Trained baseline models:
  - **Linear Regression (with standardization)**  
  - Decision Tree Regressor (`max_depth=6`) as a comparison model  
- The **Linear Regression** model was chosen as the **baseline** since it offers a high-bias, low-variance reference point.

### Part B: Ensemble Techniques

1. **Bagging Regressor (Linear Regression base)**  
   - Targets **variance reduction** by training multiple Linear Regressors on bootstrapped samples.  
   - Each model captures slightly different data characteristics, and their averaged predictions produce a more stable output.

2. **Gradient Boosting Regressor**  
   - Targets **bias reduction** by sequentially training weak learners (Linear Regressors or shallow trees) on residuals from prior models.  
   - Each iteration corrects the systematic errors of the previous ensemble, leading to progressively improved predictions.

### Part C: Stacking for Optimal Performance
- **Base Learners (Level-0):**  
  `KNN`, `Bagging (Linear Regression)`, `Gradient Boosting Regressor`
- **Meta-Learner (Level-1):**  
  `Ridge Regression`
- Stacking learns how to best combine the outputs of diverse models, minimizing both bias and variance simultaneously.

## Evaluation Metrics

For all models, the following regression metrics were computed:

| Metric | Description | Optimization |
|:--|:--|:--|
| **RMSE (Root Mean Squared Error)** | Measures prediction deviation magnitude | Lower = Better |
| **R² (Coefficient of Determination)** | Measures variance explained by the model | Higher = Better |
| **MAPE (Mean Absolute Percentage Error)** | Measures prediction accuracy as percentage error | Lower = Better |

## Model Performance Summary

| Model | RMSE ↓ | R² ↑ | MAPE ↓ |
|--------|--------|------|-------------|
| **Decision Tree Regressor** | 119.84 | 0.5329 | 4.4128 |
| **Linear Regressor (Baseline)** | 100.02 | 0.6747 | 2.9407 |
| **Bagging (Linear Regressor)** | 100.00 | 0.6748 | 2.9397 |
| **Gradient Boosting Regressor** | 78.39 | 0.8001 | 1.6557 |
| **Stacking Regressor** | **65.04** | **0.8624** | **1.2133** |

![model_eval](./assests/model_eval.png)

> **Best Model:** *Stacking Regressor* achieved the lowest RMSE and MAPE, and the highest R² score.


## Bias–Variance Trade-off Discussion

| Model | Bias | Variance | Learning Pattern |
|:--|:--|:--|:--|
| **Linear Regression (Baseline)** | **High** | **Low** | Assumes linear relationships; underfits complex patterns |
| **Decision Tree Regressor** | Low | High | Learns non-linear boundaries but overfits easily |
| **Bagging (Linear Regressor)** | Similar Bias | **Slightly Lower Variance** | Averages predictions from multiple Linear Regressors trained on bootstrapped samples |
| **Gradient Boosting Regressor** | **Reduced Bias** | Moderate Variance | Sequentially fits residuals to improve bias and capture non-linearities |
| **Stacking Regressor** | Balanced | Balanced | Combines diverse learners and meta-learning for optimal bias–variance trade-off |

- **Bagging** slightly reduces the variance of the Linear Regression baseline.  
- **Boosting** significantly reduces bias by iteratively correcting residuals.  
- **Stacking** achieves the best overall generalization by combining the strengths of all approaches through a meta-learner.


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
