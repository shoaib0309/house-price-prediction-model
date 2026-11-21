# 🏠 Ames House Price Prediction (Supervised Machine Learning)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Capstone](https://img.shields.io/badge/Capstone_Project-Honourmand_Punjab-success?style=for-the-badge)

## 📌 Executive Summary
Developed as the **Capstone Project for the Data Science & Machine Learning Program at Honourmand Punjab**, this repository implements an end-to-end supervised regression pipeline on the **Ames Housing Dataset** (sourced from Kaggle). The objective is to accurately predict residential property sale prices (`SalePrice`) using structured feature preprocessing, missing-value imputation (`LotFrontage`), and multi-metric model evaluation.

---

## 🛠️ Tech Stack & Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions, skewness, and correlations with `SalePrice`.
2. **Data Cleaning & Imputation:** Handled missing values systematically—including neighborhood-aware/statistical imputation for `LotFrontage` and categorical null handling.
3. **Feature Engineering & Preprocessing:** Encoded categorical variables and scaled numerical predictors for linear modeling.
4. **Model Training & Evaluation:** Trained a **Linear Regression** model using `Scikit-Learn` and evaluated performance across multiple error metrics.

---

## 📊 Model Evaluation Metrics

| Metric | Score / Value | Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | `[0.90]` | Variance in house prices explained by the model |
| **RMSE** | `$[0.1287]` | Root Mean Squared Error (penalizes large errors) |
| **MAE** | `$[0.0901]` | Average absolute prediction error in USD |
| **MSE** | `[0.0165]` | Mean Squared Error |

---

## 📈 Key Insights
* **Top Price Drivers:** Overall material/finish quality (`OverallQual`), above-grade living area (`GrLivArea`), and garage capacity strongly correlate with higher sale prices.
* **Data Quality Impact:** Proper imputation of `LotFrontage` and removal of extreme outliers significantly stabilized linear regression residuals.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/shoaib0309/house-price-prediction-model.git](https://github.com/shoaib0309/house-price-prediction-model.git)
cd house-price-prediction-model

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Launch Jupyter Notebook
jupyter notebook
