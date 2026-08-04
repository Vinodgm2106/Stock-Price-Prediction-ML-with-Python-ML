import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from keras.models import load_model
except Exception:
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        load_model = None

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Stock Analytics Dashboard",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MIDNIGHT LUXURY GOLD THEME CONFIGURATION
# ==========================================
bg_color = "#0a0a0c"
card_bg = "#141419"
border_color = "#ffd700"
text_color = "#f5f5f7"
accent_gradient = "linear-gradient(135deg, #ffd700 0%, #ff8c00 100%)"
plotly_bg = "#121217"
plotly_paper = "#141419"
grid_color = "#23232c"
primary_line = "#ffd700"
secondary_line = "#ff8c00"
pred_line = "#ff3366"

# Dynamic CSS Injection
st.markdown(f"""
    <style>
    /* Background & Fonts */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Hero Gradient Title */
    .hero-title {{
        background: {accent_gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    
    /* Custom Glass Cards for Metrics */
    div[data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color}33 !important;
        border-top: 3px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(255, 215, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: {text_color} !important;
    }}
    
    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: {card_bg};
        padding: 8px;
        border-radius: 12px;
        border: 1px solid {grid_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        border-radius: 8px;
        padding: 0px 20px;
        font-weight: 700;
        color: {text_color};
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: {accent_gradient} !important;
        color: #000000 !important;
    }}
    
    /* Download Button Styling */
    .stDownloadButton button {{
        background: {accent_gradient} !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 800 !important;
        transition: opacity 0.2s ease;
    }}
    .stDownloadButton button:hover {{
        opacity: 0.9;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚜️ Stock Navigator")

# Stock Selection Controls
st.sidebar.subheader("📌 Stock Ticker")
popular_stocks = ["GOOG", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "INFY.NS"]
selected_popular = st.sidebar.selectbox("Quick Select Stock", ["Custom"] + popular_stocks)

default_ticker = selected_popular if selected_popular != "Custom" else "GOOG"
stock_input = st.sidebar.text_input("Symbol", default_ticker)
stock = stock_input.strip().upper()

# Date Range Controls
st.sidebar.subheader("📅 Date Range")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.sidebar.date_input("Start Date", datetime(2013, 1, 1))
with col_d2:
    end_date = st.sidebar.date_input("End Date", datetime(2023, 12, 30))

# Indicators Controls
st.sidebar.subheader("📊 Indicators")
show_sma = st.sidebar.multiselect("Moving Averages (SMA)", [20, 50, 100, 200], default=[100, 200])
show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# Forecast Control
st.sidebar.subheader("🔮 Forecasting")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)

# ==========================================
# 1. LOAD PRE-TRAINED LSTM MODEL
# ==========================================
@st.cache_resource
def load_prediction_model():
    if load_model is None:
        return None
    
    candidates = [
        'Stock_Predictions_Model.keras',
        'Stock Predictions Model.keras',
        os.path.join(os.path.dirname(__file__), 'Stock_Predictions_Model.keras') if '__file__' in globals() else '',
        os.path.join(os.path.dirname(__file__), 'Stock Predictions Model.keras') if '__file__' in globals() else '',
        os.path.join('.ipynb_checkpoints', 'Stock Predictions Model.keras')
    ]
    
    for path in candidates:
        if path and os.path.exists(path):
            try:
                m = load_model(path)
                if m is not None:
                    return m
            except Exception:
                pass

    # Fallback directory scan
    base_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    for search_dir in ['.', base_dir]:
        if search_dir and os.path.exists(search_dir):
            try:
                for fname in os.listdir(search_dir):
                    if fname.endswith('.keras') or fname.endswith('.h5'):
                        try:
                            m = load_model(os.path.join(search_dir, fname))
                            if m is not None:
                                return m
                        except Exception:
                            pass
            except Exception:
                pass
    return None

model = load_prediction_model()

# ==========================================
# 2. DATA FETCHING ENGINE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_symbol, s_date, e_date):
    try:
        s_str = s_date.strftime('%Y-%m-%d')
        e_str = e_date.strftime('%Y-%m-%d')
        
        df = yf.download(ticker_symbol, start=s_str, end=e_str, progress=False)
        
        if df.empty:
            time.sleep(0.5)
            t_obj = yf.Ticker(ticker_symbol)
            df = t_obj.history(start=s_str, end=e_str)
            
        if df.empty:
            t_obj = yf.Ticker(ticker_symbol)
            df = t_obj.history(period="10y")
            
        return df
    except Exception:
        return pd.DataFrame()

with st.spinner(f"Fetching market data for {stock}..."):
    data = fetch_stock_data(stock, start_date, end_date)

if model is None:
    st.error("❌ Pre-trained LSTM model file `Stock Predictions Model.keras` not found.")
    st.stop()

if data is None or data.empty or len(data) < 100:
    st.error(f"❌ No market data found for ticker **'{stock}'**. Please verify the symbol or adjust dates.")
    st.stop()

# Helper to Extract Columns Safely
def extract_single_column(df, col_name, ticker_sym):
    if isinstance(df.columns, pd.MultiIndex):
        if col_name in df.columns.levels[0]:
            col_df = df[col_name]
            if ticker_sym in col_df.columns:
                return col_df[ticker_sym]
            return col_df.iloc[:, 0]
        return df.iloc[:, 0]
    else:
        if col_name in df.columns:
            return df[col_name]
        return df.iloc[:, 0]

close_series = extract_single_column(data, 'Close', stock).dropna()
open_series = extract_single_column(data, 'Open', stock).dropna()
high_series = extract_single_column(data, 'High', stock).dropna()
low_series = extract_single_column(data, 'Low', stock).dropna()
volume_series = extract_single_column(data, 'Volume', stock).dropna()

close_data = pd.DataFrame(close_series)

# Dataframe of Technical Indicators
df_indicators = pd.DataFrame(index=close_series.index)
df_indicators['Close'] = close_series
df_indicators['Open'] = open_series
df_indicators['High'] = high_series
df_indicators['Low'] = low_series
df_indicators['Volume'] = volume_series

for ma in [20, 50, 100, 200]:
    df_indicators[f'SMA_{ma}'] = close_series.rolling(window=ma).mean()

# RSI
delta = close_series.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_indicators['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp12 = close_series.ewm(span=12, adjust=False).mean()
exp26 = close_series.ewm(span=26, adjust=False).mean()
df_indicators['MACD'] = exp12 - exp26
df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']

# Bollinger Bands
sma20 = close_series.rolling(20).mean()
std20 = close_series.rolling(20).std()
df_indicators['BB_Upper'] = sma20 + (std20 * 2)
df_indicators['BB_Lower'] = sma20 - (std20 * 2)

# ==========================================
# HERO HEADER & METRICS
# ==========================================
st.markdown(f'<div class="hero-title">⚜️ {stock} Stock Intelligence</div>', unsafe_allow_html=True)
st.caption("AI-Powered Time-Series Forecasting & Deep Technical Analysis Engine (Midnight Luxury Gold Edition)")

current_price = close_series.iloc[-1]
prev_price = close_series.iloc[-2] if len(close_series) > 1 else current_price
daily_change = current_price - prev_price
daily_change_pct = (daily_change / prev_price) * 100

high_52w = close_series.tail(252).max() if len(close_series) >= 252 else close_series.max()
low_52w = close_series.tail(252).min() if len(close_series) >= 252 else close_series.min()
latest_vol = volume_series.iloc[-1]

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(label="Current Price", value=f"${current_price:,.2f}", delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)")
with kpi2:
    st.metric(label="52-Wk High", value=f"${high_52w:,.2f}")
with kpi3:
    st.metric(label="52-Wk Low", value=f"${low_52w:,.2f}")
with kpi4:
    st.metric(label="Volume", value=f"{latest_vol:,.0f}")
with kpi5:
    st.metric(label="Data Points", value=f"{len(close_series):,} Days")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price & Candlesticks",
    "📈 Technical Indicators",
    "🤖 LSTM AI Predictions",
    "🔮 Future N-Day Forecast",
    "📥 Data Export"
])

# Plotly Layout Helper
def apply_plotly_style(fig, title_text=""):
    fig.update_layout(
        title=title_text,
        paper_bgcolor=plotly_paper,
        plot_bgcolor=plotly_bg,
        font=dict(color=text_color, family="sans-serif"),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# ------------------------------------------
# TAB 1: OVERVIEW & CANDLESTICK
# ------------------------------------------
with tab1:
    st.subheader(f"Price History & Moving Averages - {stock}")
    
    fig_candle = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{stock} Candlestick Chart', 'Trading Volume'),
        row_width=[0.2, 0.8]
    )

    fig_candle.add_trace(
        go.Candlestick(
            x=df_indicators.index,
            open=df_indicators['Open'],
            high=df_indicators['High'],
            low=df_indicators['Low'],
            close=df_indicators['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    sma_colors = {20: '#f59e0b', 50: '#06b6d4', 100: primary_line, 200: secondary_line}
    for ma in show_sma:
        fig_candle.add_trace(
            go.Scatter(
                x=df_indicators.index,
                y=df_indicators[f'SMA_{ma}'],
                name=f'SMA {ma}',
                line=dict(width=1.8, color=sma_colors.get(ma, '#ffd700'))
            ),
            row=1, col=1
        )

    fig_candle.add_trace(
        go.Bar(
            x=df_indicators.index,
            y=df_indicators['Volume'],
            name='Volume',
            marker_color='rgba(255, 215, 0, 0.35)'
        ),
        row=2, col=1
    )

    fig_candle.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        paper_bgcolor=plotly_paper,
        plot_bgcolor=plotly_bg,
        font=dict(color=text_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

# ------------------------------------------
# TAB 2: TECHNICAL INDICATORS
# ------------------------------------------
with tab2:
    st.subheader("Technical Indicator Suite")

    if show_rsi:
        st.markdown("##### 🟣 Relative Strength Index (RSI - 14 Days)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['RSI'], name='RSI', line=dict(color='#c084fc', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
        fig_rsi = apply_plotly_style(fig_rsi)
        fig_rsi.update_layout(height=320)
        st.plotly_chart(fig_rsi, use_container_width=True)

    if show_macd:
        st.markdown("##### 🔵 MACD & Signal Line")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD'], name='MACD', line=dict(color='#ffd700', width=2)))
        fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD_Signal'], name='Signal', line=dict(color='#ff8c00', width=2)))
        fig_macd.add_trace(go.Bar(x=df_indicators.index, y=df_indicators['MACD_Hist'], name='Histogram', marker_color='rgba(255, 215, 0, 0.4)'))
        fig_macd = apply_plotly_style(fig_macd)
        fig_macd.update_layout(height=320)
        st.plotly_chart(fig_macd, use_container_width=True)

    if show_bb:
        st.markdown("##### 🟢 Bollinger Bands (20-day SMA)")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['Close'], name='Close Price', line=dict(color=primary_line, width=2)))
        fig_bb.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['BB_Upper'], name='Upper Band', line=dict(color='#ff8c00', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['BB_Lower'], name='Lower Band', line=dict(color='#ff8c00', dash='dash')))
        fig_bb = apply_plotly_style(fig_bb)
        fig_bb.update_layout(height=380)
        st.plotly_chart(fig_bb, use_container_width=True)

# ------------------------------------------
# TAB 3: LSTM AI PREDICTIONS & EVALUATION
# ------------------------------------------
with tab3:
    st.subheader("🤖 LSTM Deep Learning Neural Network Evaluation")
    
    train_size = int(len(close_data) * 0.80)
    data_train = pd.DataFrame(close_data.iloc[0:train_size])
    data_test = pd.DataFrame(close_data.iloc[train_size:len(close_data)])

    scaler = MinMaxScaler(feature_range=(0, 1))
    pas_100_days = data_train.tail(100)
    data_test_combined = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test_combined)

    x_test, y_test = [], []
    for i in range(100, data_test_scaled.shape[0]):
        x_test.append(data_test_scaled[i-100:i])
        y_test.append(data_test_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    with st.spinner("Executing LSTM neural network predictions..."):
        y_predicted = model.predict(x_test, verbose=0)

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_actual = y_test * scale_factor

    test_dates = close_data.index[train_size:]

    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae = mean_absolute_error(y_actual, y_predicted)
    mape = np.mean(np.abs((y_actual - y_predicted.flatten()) / y_actual)) * 100
    r2 = r2_score(y_actual, y_predicted)

    st.markdown("##### Accuracy & Error Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="RMSE", value=f"${rmse:.2f}")
    with m2:
        st.metric(label="MAE", value=f"${mae:.2f}")
    with m3:
        st.metric(label="MAPE", value=f"{mape:.2f}%")
    with m4:
        st.metric(label="R² Accuracy", value=f"{r2:.4f}")

    st.markdown("##### Original Price vs. LSTM Model Prediction")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test_dates, y=y_actual, name='Actual Price', line=dict(color=primary_line, width=2.2)))
    fig_pred.add_trace(go.Scatter(x=test_dates, y=y_predicted.flatten(), name='Predicted Price', line=dict(color=pred_line, width=2.2)))
    fig_pred = apply_plotly_style(fig_pred)
    fig_pred.update_layout(height=520, xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_pred, use_container_width=True)

# ------------------------------------------
# TAB 4: FUTURE FORECASTING
# ------------------------------------------
with tab4:
    st.subheader(f"🔮 Multi-Step Future Price Projection (Next {forecast_days} Days)")
    
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler_full.fit_transform(close_data)

    current_input = scaled_full[-100:].reshape(1, 100, 1)
    future_preds_scaled = []

    with st.spinner(f"Calculating {forecast_days}-day future prediction loop..."):
        curr_step = current_input.copy()
        for _ in range(forecast_days):
            pred = model.predict(curr_step, verbose=0)
            future_preds_scaled.append(pred[0, 0])
            curr_step = np.append(curr_step[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    scale_factor_full = 1 / scaler_full.scale_[0]
    future_predictions = np.array(future_preds_scaled) * scale_factor_full

    last_date = close_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    recent_historical = close_series.tail(60)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=recent_historical.index, y=recent_historical.values, name='Historical Close', line=dict(color=primary_line, width=2.2)))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_predictions, name=f'{forecast_days}-Day Forecast', line=dict(color=secondary_line, width=3, dash='dash')))
    fig_forecast = apply_plotly_style(fig_forecast)
    fig_forecast.update_layout(height=520, xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("##### Forecasted Values")
    df_future = pd.DataFrame({
        "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
        "Predicted Price ($)": [f"${p:,.2f}" for p in future_predictions]
    })
    st.dataframe(df_future, use_container_width=True)

# ------------------------------------------
# TAB 5: RAW DATA & EXPORT
# ------------------------------------------
with tab5:
    st.subheader("📥 Export Historical & Technical Dataset")
    st.dataframe(df_indicators.tail(150), use_container_width=True)

    csv_data = df_indicators.to_csv().encode('utf-8')
    st.download_button(
        label=f"⬇ Download {stock} Dataset (CSV)",
        data=csv_data,
        file_name=f"{stock}_midnight_gold_intelligence.csv",
        mime="text/csv"
    )
