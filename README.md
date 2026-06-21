# Advertising Click Prediction with Machine Learning

This project analyzes and predicts whether users will click on online advertisements using demographic, behavioral, and temporal features.

The project was developed as part of a Data Science / Artificial Intelligence academic report and compares multiple machine learning models, including Logistic Regression, Random Forest, XGBoost, and a TensorFlow Neural Network.

## Project Overview

The dataset contains 1,000 user records with features such as:

- Daily time spent on site
- Daily internet usage
- Age
- Area income
- Gender
- Timestamp
- Clicked on Ad target variable

The objective is to predict whether a user clicked on an advertisement and to understand which features influence click behavior.

## Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- TensorFlow Neural Network

## Feature Engineering

The project includes several preprocessing and feature engineering steps:

- Timestamp conversion into day/night activity
- QuantileTransformer normalization for skewed features
- PCA for correlated engagement features
- Feature selection
- Standardization for neural network training

## Results

The best-performing models were:

| Model | Accuracy | F1-score | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.99 | 0.99 | >0.99 | >0.99 |
| TensorFlow Neural Network | 0.99 | 0.99 | >0.99 | >0.99 |
| Random Forest | 0.98 | 0.98 | - | - |
| XGBoost | 0.97 | 0.97 | - | - |

The results show that the dataset is highly structured and nearly linearly separable. Logistic Regression performed as well as the TensorFlow neural network, suggesting that simpler interpretable models can be highly effective when features are well engineered.

## Key Findings

- Older users were more likely to click ads.
- Heavy online users were less likely to click.
- Higher-income areas showed lower click-through rates.
- Gender and time of day had limited independent influence.
- False negatives occurred mainly on rare user behavior profiles.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib
- Seaborn
- PCA
- QuantileTransformer

## Dataset

Dataset source:

Advertisement Click on Ad dataset, Kaggle.

## Future Work

Possible improvements:

- Add text embeddings from advertisement content
- Use more advanced neural network architectures
- Add SHAP or permutation feature importance
- Apply cost-sensitive learning for rare misclassification cases
- Build a small Streamlit or Django dashboard for interactive predictions
