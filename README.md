# air-quality-gradient-boosting
An advanced machine learning pipeline comparing XGBoost, LightGBM, and CatBoost to predict Carbon Monoxide levels. Built with Optuna hyperparameter tuning.
# Air Quality Prediction: High-Performance Gradient Boosting Architecture

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Optimization](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple.svg)](https://optuna.org/)

An advanced, end-to-end machine learning pipeline designed to predict atmospheric Carbon Monoxide ($CO$) concentrations (`co_column`) utilizing spatial, temporal, and cyclical feature engineering. This project processes a massive dataset (~18.4 million records), evaluates multiple state-of-the-art gradient boosting frameworks, and optimizes the champion model via Bayesian optimization.

---

## 🚀 Features & Pipeline

* **Massive Data Scale:** Safely handles, merges, and cleans over **18.4 million rows** spanning years 2024 and 2025.
* **Feature Engineering Matrix:** 
  * *Temporal & Cyclical:* Extracts features like weekend flags, quarters, and encodes cyclical trends using Sine/Cosine transformations for months and days.
  * *Spatial Dynamics:* Tracks coordinates, squared spatial interactions, and calculates geometric distances from the geographical dataset center.
* **Framework Benchmarking:** Evaluates performance across **XGBoost, LightGBM, CatBoost, and HistGradientBoosting**.
* **Hyperparameter Optimization:** Utilizes **Optuna** to execute a 25-trial search space to maximize the $R^2$ score.

---

## 📊 Model Evaluation & Benchmarking

The initial baseline evaluation highlighted **XGBoost** as the top-performing model. The architectures were compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$):

| Model Rank | Model | MAE | RMSE | $R^2$ Score | Training Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **XGBoost** | **0.002025** | **0.002669** | **0.884791** | 662.26 |
| 2 | LightGBM | 0.002290 | 0.003034 | 0.851085 | 342.61 |
| 3 | CatBoost | 0.002333 | 0.003105 | 0.844056 | 980.73 |
| 4 | HistGradientBoosting | 0.002549 | 0.003396 | 0.813450 | 371.43 |

### 🛠️ Hyperparameter Tuning (Optuna)
By running a sub-sampled optimization space via Optuna, the optimal parameters for the champion `XGBRegressor` were discovered:
```python
{
    'n_estimators': 492, 
    'max_depth': 9, 
    'learning_rate': 0.0495, 
    'subsample': 0.8483, 
    'colsample_bytree': 0.7593, 
    'min_child_weight': 3
}
