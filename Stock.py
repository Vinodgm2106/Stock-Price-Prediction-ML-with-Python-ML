# 📈 Stock Price Prediction System (Production-Safe Version)
# LSTM-based Time Series Forecasting using Yahoo Finance Data

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction System")
st.subheader("LSTM-based Time Series Forecasting using Yahoo Finance Data")

# ============================
# USER INPUT
# ============================
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", "AAPL").strip().upper()
start_date = st.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.date_input("End Date", datetime.today())

# ============================
# CACHED DATA LOADER
# ============================
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
        progress=False,
        threads=False
    )
    return df

# ============================
# ROBUST DATA LOADING
# ============================
def load_data(ticker, start_date, end_date):
    try:
        if not ticker:
            st.error("❌ Please enter a valid ticker symbol.")
            return None

        today = datetime.today().date()
        if end_date >= today:
            end_date = today - timedelta(days=1)

        if start_date >= end_date:
            st.error("❌ Start date must be earlier than end date.")
            return None

        with st.spinner("Fetching stock data..."):
            df = fetch_data(ticker, start_date, end_date)

        # Retry once if empty
        if df.empty:
            time.sleep(1)
            df = fetch_data(ticker, start_date, end_date)

        # Fallback method
        if df.empty:
            stock = yf.Ticker(ticker)
            df = stock.history(period="5y")

        if df.empty:
            st.error("❌ No data found. Check ticker symbol (e.g., AAPL, TSLA, INFY.NS)")
            return None

        if "Close" not in df.columns:
            st.error("❌ Close price column missing in dataset.")
            return None

        df = df.dropna()

        if len(df) < 100:
            st.warning("⚠ Not enough data for reliable LSTM training.")
            return None

        return df

    except Exception as e:
        st.error("⚠ Data fetching failed.")
        st.exception(e)
        return None

# ============================
# MAIN EXECUTION
# ============================
if st.button("Run Prediction"):

    df = load_data(ticker, start_date, end_date)

    if df is None:
        st.stop()

    st.success("✅ Data Loaded Successfully")
    st.write(df.tail())

    # ============================
    # CANDLESTICK CHART
    # ============================
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # PREPROCESSING
    # ============================
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Close"]])

    sequence_length = 60
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ============================
    # MODEL
    # ============================
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # ============================
    # PREDICTIONS
    # ============================
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    st.write(f"📊 RMSE: {rmse:.2f}")
    st.write(f"📊 MAE: {mae:.2f}")

    # Plot Actual vs Predicted
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=actual.flatten(), name="Actual"))
    fig2.add_trace(go.Scatter(y=predictions.flatten(), name="Predicted"))
    fig2.update_layout(title="Actual vs Predicted Prices")
    st.plotly_chart(fig2, use_container_width=True)

    # Next Day Prediction
    last_60_days = scaled_data[-sequence_length:]
    last_60_days = np.reshape(last_60_days, (1, sequence_length, 1))
    next_day_scaled = model.predict(last_60_days)
    next_day_price = scaler.inverse_transform(next_day_scaled)

    st.success(f"🔮 Predicted Next Day Closing Price: {next_day_price[0][0]:.2f}")

st.markdown("---")
st.caption("Production-Ready | Error-Handled | Cached | Retry Mechanism | Candlestick Visualization")
