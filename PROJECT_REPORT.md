
# STOCK PRICE PREDICTION USING MACHINE LEARNING

## Internship Project Report

**Project Title:** Stock Price Prediction Using Machine Learning

**Domain:** Data Science and Machine Learning

**Programming Language:** Python

**Organization:** CODEC Internship

**Year:** 2026

---

# 1. INTRODUCTION

Stock market prediction is an important application
of data science and machine learning.

Stock prices fluctuate due to several factors,
including market demand, economic conditions,
company performance and investor sentiment.

The objective of this project is to develop a
machine learning application that analyzes historical
stock market data and predicts the next trading
day's closing price.

The project uses Linear Regression to identify
relationships between historical stock prices
and future closing prices.

An interactive dashboard is developed using
Streamlit to display predictions, charts and
model performance.

# 2. PROBLEM STATEMENT

Predicting stock market prices is challenging
because financial markets are affected by many
uncertain factors.

Manual analysis of historical prices can be
time-consuming.

This project aims to create an automated system
that uses historical stock market data and
machine learning techniques to estimate future
closing prices.

# 3. OBJECTIVES

The main objectives are:

1. Collect historical stock market data.
2. Clean and preprocess the collected data.
3. Extract important features from historical prices.
4. Develop a Linear Regression model.
5. Predict the next trading day's closing price.
6. Evaluate the model using performance metrics.
7. Visualize historical and predicted stock prices.
8. Develop an interactive dashboard using Streamlit.

# 4. TECHNOLOGIES USED

## Programming Language
Python

## Libraries

- Pandas: Data manipulation and analysis.
- NumPy: Numerical computations.
- Scikit-learn: Machine learning model development.
- Matplotlib: Data visualization.
- Seaborn: Statistical visualization.
- yfinance: Historical stock market data collection.
- Streamlit: Interactive web application development.

## Development Environment

Visual Studio Code.

# 5. DATA COLLECTION

Historical stock market information is collected
using the Yahoo Finance library (yfinance).

The application allows users to enter stock symbols
such as:

- AAPL - Apple
- MSFT - Microsoft
- TSLA - Tesla
- RELIANCE.NS - Reliance Industries
- TCS.NS - Tata Consultancy Services
- INFY.NS - Infosys

The collected dataset contains:

- Open Price
- High Price
- Low Price
- Closing Price
- Trading Volume

Historical data is automatically saved in CSV
format inside the data folder.

# 6. DATA PREPROCESSING

Data preprocessing is performed to prepare the
historical stock information for machine learning.

The following features are created:

1. Previous Closing Price
2. Previous Two Days Closing Price
3. Previous Three Days Closing Price
4. Five-Day Moving Average
5. Ten-Day Moving Average
6. Daily Percentage Return
7. Trading Volume

The target variable is the next trading day's
closing price.

Missing values generated during feature
engineering are removed before model training.

# 7. MACHINE LEARNING ALGORITHM

## Linear Regression

Linear Regression is a supervised machine learning
algorithm used to predict continuous numerical values.

In this project, the algorithm learns the relationship
between historical stock features and the next
trading day's closing price.

The general equation is:

Y = b0 + b1X1 + b2X2 + ... + bnXn

Where:

- Y = Predicted closing price
- b0 = Intercept
- b1, b2 ... bn = Model coefficients
- X1, X2 ... Xn = Input features

The dataset is divided chronologically:

- 80% Training Data
- 20% Testing Data

The model is trained using the training dataset
and evaluated on unseen testing data.

# 8. MODEL EVALUATION

The model is evaluated using three performance
metrics.

## Mean Absolute Error (MAE)

Measures the average absolute difference between
actual and predicted prices.

## Root Mean Squared Error (RMSE)

Measures prediction error while giving greater
weight to larger errors.

## R-Squared (R²)

Measures how much variation in the target variable
is explained by the model.

These metrics help assess model performance
on historical test data.

# 9. SYSTEM FEATURES

The application provides the following features:

1. Automatic historical stock data collection.
2. Multiple stock symbol selection.
3. Historical stock price analysis.
4. Linear Regression model training.
5. Next-day closing price estimation.
6. Actual versus predicted price visualization.
7. MAE, RMSE and R² evaluation.
8. Feature correlation heatmap.
9. Historical stock data table.
10. CSV download functionality.

# 10. SYSTEM WORKFLOW

The system follows these steps:

1. User enters a stock symbol.
2. Historical data is downloaded.
3. Data is saved in CSV format.
4. Data preprocessing is performed.
5. Features and target variables are created.
6. Data is divided into training and testing sets.
7. Linear Regression model is trained.
8. Model performance is evaluated.
9. Next trading day's price is estimated.
10. Results are displayed on the dashboard.

# 11. EXPECTED OUTPUT

The application displays:

- Latest closing stock price.
- Predicted next closing price.
- Expected percentage change.
- Historical closing price chart.
- Actual versus predicted price chart.
- Model performance metrics.
- Feature correlation heatmap.
- Historical stock data.
- Downloadable prediction results.

# 12. ADVANTAGES

- Easy-to-use interactive dashboard.
- Automatic historical data collection.
- Supports multiple stock symbols.
- Simple machine learning implementation.
- Visual representation of predictions.
- Provides downloadable CSV files.
- Demonstrates practical applications of Python.

# 13. LIMITATIONS

- Stock markets are affected by unpredictable events.
- Linear Regression assumes a linear relationship.
- Historical data alone cannot capture all market factors.
- Predictions are not guaranteed to be accurate.
- The project does not include news or sentiment analysis.

# 14. FUTURE ENHANCEMENTS

The project can be improved by:

1. Implementing Random Forest Regression.
2. Implementing Decision Tree Regression.
3. Comparing multiple machine learning algorithms.
4. Adding technical indicators such as RSI and MACD.
5. Including market news sentiment analysis.
6. Improving dashboard design.
7. Deploying the application online.

# 15. CONCLUSION

The Stock Price Prediction project demonstrates
the application of machine learning in financial
data analysis.

Using Python, historical stock market information
is collected, processed and analyzed.

A Linear Regression model is trained to estimate
the next trading day's closing price.

The Streamlit dashboard provides an interactive
interface for visualizing stock information and
model performance.

This project helped develop practical knowledge
of Python programming, data preprocessing,
feature engineering, machine learning,
model evaluation and data visualization.

# 16. REFERENCES

1. Python Documentation:
   https://docs.python.org/3/

2. Pandas Documentation:
   https://pandas.pydata.org/docs/

3. Scikit-learn Documentation:
   https://scikit-learn.org/

4. Streamlit Documentation:
   https://docs.streamlit.io/

5. Yahoo Finance:
   https://finance.yahoo.com/

6. yfinance Documentation:
   https://ranaroussi.github.io/yfinance/

# DISCLAIMER

This project is developed for educational and
internship purposes only.

The predicted stock prices are estimates and
should not be considered guaranteed financial
forecasts or investment recommendations.
