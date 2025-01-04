# Health Insurance Cost Prediction using Machine Learning

This project aims to predict health insurance costs using machine learning techniques applied to a dataset from Kaggle. The goal is to evaluate the performance of different models and identify the significant factors influencing medical charges.

## Overview

Health insurance companies need reliable models to estimate the insurance costs for individuals based on various factors like age, smoking habits, BMI, and more. In this project, we use machine learning algorithms to build a predictive model for insurance costs.

- **Dataset**: The dataset was sourced from Kaggle and contains data on individuals' demographics and medical costs.
- **Models Used**:
  - **Linear Regression**: A simple yet effective model for predicting continuous values.
  - **Decision Tree Regressor**: A non-linear model used for regression tasks.
  - **Random Forest Regressor**: An ensemble model that combines multiple decision trees for improved performance.

## Key Features Identified

Through feature importance analysis, we identified the following features as the most influential in predicting insurance costs:
- Age
- BMI (Body Mass Index)
- Smoker status
- Children (number of dependents)

## Model Evaluation

- **Random Forest Regressor**:
  - Training R²: 0.973
  - Test R²: 0.859
  - RMSE: 0.385
  - Best overall model in terms of performance.

- **Linear Regression**:
  - Training R²: 0.741
  - Test R²: 0.781
  - RMSE: 0.482
  - Good performance, but less accurate than Random Forest.

- **Decision Tree Regressor**:
  - Training R²: 0.998
  - Test R²: 0.732
  - RMSE: 0.532
  - Overfitting on the training data, but still provides reasonable predictions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/health-insurance-cost-prediction.git
   cd health-insurance-cost-prediction
