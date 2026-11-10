
# Stock Price Prediction Using Machine Learning

## Project Description

This project develops a machine learning system that
predicts the next trading day's stock closing price
using historical stock market data.

The system uses Linear Regression to learn relationships
between historical stock prices and future closing prices.

## Objectives

- Collect historical stock market data.
- Perform data preprocessing.
- Extract useful features.
- Train a Linear Regression model.
- Predict future closing prices.
- Evaluate model performance.
- Visualize actual and predicted prices.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Yahoo Finance API (yfinance)
- Matplotlib
- Seaborn
- Streamlit

## Machine Learning Algorithm

Linear Regression is used to predict the next trading
day's closing price based on historical features.

Features include:
- Previous closing prices
- 5-day moving average
- 10-day moving average
- Daily returns
- Trading volume

Target:
- Next trading day's closing price

## Model Evaluation

The model is evaluated using:

1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. R-squared (R²)

## Installation

Clone or download the project.

Install dependencies:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

Open the local URL displayed in your terminal.

## Expected Output

- Historical stock price chart
- Predicted next closing price
- Actual vs predicted comparison
- Model performance metrics
- Feature correlation heatmap
- Downloadable prediction results

## Disclaimer

This project is intended for educational purposes.
Stock market predictions are uncertain and should not
be treated as financial advice.
# -Stock-Price-Predictor
