# Real-Estate-Price-Prediction-Dataset-Analysis


## Overview
This repository contains code and data for exploring and analyzing a real estate dataset aimed at predicting housing prices. The dataset comprises various features such as house specifications (bedrooms, bathrooms, square footage, etc.) and location-related details. The analysis involves exploratory data analysis (EDA), data preprocessing, feature engineering, outlier detection, and scaling.

## Contents
- `Data.csv`: Raw dataset file containing housing-related information.
- `Analysis.ipynb`: Jupyter Notebook file containing the Python code for data analysis and exploration.
- `README.md`: This README file explaining the contents and purpose of the repository.

## Analysis
The primary objective of this analysis is to predict housing prices based on various features available in the dataset. The analysis includes:

- **Data Loading and Overview**: Read the dataset, display basic information (shape, columns, data types), and check for missing values.
- **Exploratory Data Analysis (EDA)**: Explore the dataset's distribution, correlations, and visualizations of different features to understand their relationships with house prices.
- **Outlier Detection and Removal**: Identify outliers using IQR (Interquartile Range) and remove them from the dataset.
- **Feature Engineering**: Transform categorical variables into numerical representations using one-hot encoding.
- **Scaling**: Scale numerical features using MinMaxScaler to bring them within a specific range.
- **Modeling**: Build and train machine learning models for predicting house prices. Common regression models such as Linear Regression, Ridge, Random Forest Regressor and ANN
- **Evaluation**: Assess the performance of each model using evaluation metrics like R-squared (R²), Mean Absolute Error (MAE), and Mean Squared Error (MSE).
- **Visualization**: Create visualizations to understand feature importance, distributions, and relationships among variables.
- **Conclusion**: Summarize findings, limitations, and potential areas for further improvement.

## Usage
- **Environment Setup**: Ensure Python and necessary libraries (NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn) are installed.
- **Data Import**: Load the dataset using Pandas from the provided 'data.csv' file.
- **Code Execution**: Run the 'Analysis.ipynb' Jupyter Notebook or Python script to execute the analysis step-by-step.
- **Understanding the Code**: Comments and markdown cells within the notebook explain each step and its purpose.
- **Modification and Extension**: Modify the code for additional analysis, feature engineering, or model tuning as needed.

## Requirements
- **Python (3.x)**
- **Libraries**: NumPy, Pandas, Seaborn, Matplotlib, Scikit-learnn, pickel, Keras, Flask, 

Model Evaluation
Linear Regression using Ridge 
Accuracy-66%
Random Forest using Random Seactrh CV
Accuracy-69.5
ANN
Accuracy-59.3%

## Authors
- Anargha Manoj-Dataset Selection,Data Preprocessing, EDA
- Simranjeet Kaur-EDA, Model Selection, Model Evaluation
- Sharon Victor-Model Evaluation, Hyperparameter Tunnig, Pickel Files
- Lakshmi Kumari-Pickel Files, Flask Web Application
- Navneet Kaur-Flask Web Application, GitHub Repository, Deployment


Feel free to contribute, modify, or use this codebase for your projects!

