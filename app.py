
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction System")

st.markdown(
    "### Machine Learning Based Stock Market Analysis"
)

st.write(
    "This application uses historical stock market data "
    "and Linear Regression to predict the next trading day's "
    "closing price."
)

st.divider()

# ==========================================
# 2. CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs("data", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ==========================================
# 3. SIDEBAR SETTINGS
# ==========================================

st.sidebar.title("Prediction Settings")

ticker = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
).strip().upper()

period = st.sidebar.selectbox(
    "Select Historical Period",
    ["1y", "2y", "5y", "10y"]
)

st.sidebar.info(
    "Examples: AAPL, MSFT, TSLA, "
    "RELIANCE.NS, TCS.NS, INFY.NS"
)

# ==========================================
# 4. DOWNLOAD HISTORICAL STOCK DATA
# ==========================================

@st.cache_data(ttl=3600)
def get_stock_data(symbol, selected_period):

    stock = yf.Ticker(symbol)

    data = stock.history(
        period=selected_period,
        auto_adjust=True
    )

    return data.dropna()


if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

if st.sidebar.button("Predict Stock Price"):
    st.session_state.run_prediction = True

if st.sidebar.button("Reset"):
    st.session_state.run_prediction = False

# ==========================================
# 5. MAIN APPLICATION
# ==========================================

if st.session_state.run_prediction:

    if not ticker:
        st.error("Please enter a valid stock symbol.")
        st.stop()

    with st.spinner("Downloading stock market data..."):

        try:
            df = get_stock_data(ticker, period)

        except Exception as error:
            st.error(f"Unable to download stock data: {error}")
            st.stop()

    if df.empty or len(df) < 50:

        st.error(
            "Insufficient stock data. "
            "Please check the stock symbol."
        )

        st.stop()

    # ======================================
    # 6. SAVE DATA AUTOMATICALLY IN CSV
    # ======================================

    csv_path = f"data/{ticker}_stock_data.csv"

    df.to_csv(csv_path)

    st.success(
        f"Historical stock data saved successfully: {csv_path}"
    )

    # ======================================
    # 7. DATA PREPROCESSING
    # ======================================

    df = df.copy()

    df["Previous_Close"] = df["Close"].shift(1)

    df["Previous_2_Close"] = df["Close"].shift(2)

    df["Previous_3_Close"] = df["Close"].shift(3)

    df["MA_5"] = df["Close"].rolling(5).mean()

    df["MA_10"] = df["Close"].rolling(10).mean()

    df["Daily_Return"] = df["Close"].pct_change()

    # Predict next trading day's closing price

    df["Target"] = df["Close"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)

    features = [
        "Previous_Close",
        "Previous_2_Close",
        "Previous_3_Close",
        "MA_5",
        "MA_10",
        "Daily_Return",
        "Volume"
    ]

    # Training data has known target values

    training_data = df.dropna(
        subset=features + ["Target"]
    )

    # Latest row is used for next-day prediction

    latest_data = df.dropna(
        subset=features
    ).tail(1)

    if len(training_data) < 30 or latest_data.empty:

        st.error("Not enough valid data for prediction.")

        st.stop()

    X = training_data[features]

    y = training_data["Target"]

    # ======================================
    # 8. TRAIN-TEST SPLIT
    # ======================================

    split = int(len(training_data) * 0.8)

    X_train = X.iloc[:split]

    X_test = X.iloc[split:]

    y_train = y.iloc[:split]

    y_test = y.iloc[split:]

    # ======================================
    # 9. LINEAR REGRESSION MODEL
    # ======================================

    model = LinearRegression()

    model.fit(X_train, y_train)

    # Test predictions

    predictions = model.predict(X_test)

    # ======================================
    # 10. MODEL EVALUATION
    # ======================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # ======================================
    # 11. NEXT DAY PRICE PREDICTION
    # ======================================

    final_model = LinearRegression()

    final_model.fit(X, y)

    next_price = final_model.predict(
        latest_data[features]
    )[0]

    current_price = float(
        df["Close"].iloc[-1]
    )

    price_change = next_price - current_price

    percentage_change = (
        price_change / current_price
    ) * 100

    # ======================================
    # 12. DISPLAY PREDICTION RESULTS
    # ======================================

    st.header("Stock Price Prediction")

    currency = (
        "₹" if ticker.endswith(".NS")
        or ticker.endswith(".BO") else "$"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"{currency}{current_price:,.2f}"
    )

    col2.metric(
        "Predicted Next Closing Price",
        f"{currency}{next_price:,.2f}",
        f"{currency}{price_change:+,.2f}"
    )

    col3.metric(
        "Expected Percentage Change",
        f"{percentage_change:+.2f}%"
    )

    st.caption(
        "The prediction refers to the next trading session, "
        "not necessarily the next calendar day."
    )

    st.divider()

    # ======================================
    # 13. MODEL PERFORMANCE
    # ======================================

    st.header("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.4f}")

    c2.metric("RMSE", f"{rmse:.4f}")

    c3.metric("R² Score", f"{r2:.4f}")

    st.write(
        "MAE and RMSE measure prediction errors. "
        "R² measures how much variation in the test "
        "target is explained by the model."
    )

    st.divider()

    # ======================================
    # 14. HISTORICAL STOCK PRICE CHART
    # ======================================

    st.header("Historical Closing Prices")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df.index,
        df["Close"],
        color="blue",
        label="Closing Price"
    )

    ax.set_title(
        f"{ticker} Historical Stock Prices"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(f"Price ({currency})")

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ======================================
    # 15. ACTUAL VS PREDICTED CHART
    # ======================================

    st.header("Actual vs Predicted Prices")

    comparison = pd.DataFrame(
        {
            "Actual Price": y_test,
            "Predicted Price": predictions
        },
        index=y_test.index
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        comparison.index,
        comparison["Actual Price"],
        label="Actual Price",
        color="blue"
    )

    ax2.plot(
        comparison.index,
        comparison["Predicted Price"],
        label="Predicted Price",
        color="red"
    )

    ax2.set_title(
        "Actual vs Predicted Closing Prices"
    )

    ax2.set_xlabel("Date")

    ax2.set_ylabel(f"Price ({currency})")

    ax2.legend()

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

    plt.close(fig2)

    st.divider()

    # ======================================
    # 16. CORRELATION HEATMAP
    # ======================================

    st.header("Feature Correlation Heatmap")

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df[features + ["Target"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax3
    )

    ax3.set_title(
        "Correlation Between Features"
    )

    st.pyplot(fig3)

    plt.close(fig3)

    st.divider()

    # ======================================
    # 17. HISTORICAL DATA TABLE
    # ======================================

    st.header("Historical Stock Market Data")

    st.dataframe(
        df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ].tail(15),
        use_container_width=True
    )

    # ======================================
    # 18. DOWNLOAD PREDICTION RESULTS
    # ======================================

    st.header("Download Results")

    csv = comparison.to_csv().encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name=f"{ticker}_predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Historical Stock Data",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv"
    )

    st.success(
        "Stock analysis and prediction completed successfully!"
    )

    st.warning(
        "Disclaimer: This project is for educational purposes "
        "only. Stock market predictions are uncertain and "
        "should not be used as guaranteed investment advice."
    )

# ==========================================
# 19. INITIAL SCREEN
# ==========================================

else:

    st.info(
        "Enter a stock symbol in the sidebar and click "
        "'Predict Stock Price' to start."
    )

    st.subheader("Available Stock Examples")

    st.markdown(
        """
        | Company | Symbol |
        |---|---|
        | Apple | AAPL |
        | Microsoft | MSFT |
        | Tesla | TSLA |
        | Amazon | AMZN |
        | Reliance Industries | RELIANCE.NS |
        | Tata Consultancy Services | TCS.NS |
        | Infosys | INFY.NS |
        """
    )

    st.subheader("Project Technologies")

    st.write(
        "Python | Pandas | NumPy | Scikit-learn | "
        "Yahoo Finance | Matplotlib | Seaborn | Streamlit"
    )
