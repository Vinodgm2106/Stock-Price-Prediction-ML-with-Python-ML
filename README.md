# ⚜️ Stock Price Prediction ML with Python

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000.svg)](https://keras.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)

A modern, interactive Deep Learning & Technical Analysis platform for Stock Price Forecasting. Built with Python, Keras/TensorFlow LSTM Neural Networks, Streamlit, and Plotly, styled in an exclusive **Midnight Luxury Gold** design system.

🔗 **GitHub Repository**: [Vinodgm2106/Stock-Price-Prediction-ML-with-Python-ML](https://github.com/Vinodgm2106/Stock-Price-Prediction-ML-with-Python-ML)

---

## ✨ Key Features

- **⚜️ Midnight Luxury Gold Theme**: Styled with obsidian dark background, gold/amber gradient accents, and responsive glassmorphic metric cards.
- **📊 Interactive Candlestick Charts**: Powered by Plotly with volume subplots and dynamic Moving Average overlays (20, 50, 100, 200 SMA/EMA).
- **📈 Technical Indicators Suite**:
  - **RSI (Relative Strength Index - 14 Days)** with overbought (70) and oversold (30) threshold bands.
  - **MACD (Moving Average Convergence Divergence)** line, signal line, and histogram.
  - **Bollinger Bands** (20-day SMA with $\pm 2$ standard deviation upper/lower bands).
- **🤖 Deep Learning LSTM Predictions**:
  - Evaluation of pre-trained multi-layer LSTM model on test data.
  - Performance Metrics: **RMSE**, **MAE**, **MAPE (%)**, and **\(R^2\) Score**.
- **🔮 Multi-Step Future Price Forecasting**:
  - Iterative N-Day future projection loop predicting stock trends for 1 to 30 days into the future.
- **📥 Data Export & Search**:
  - Searchable technical indicator data table with a 1-click **CSV Download Button**.
- **🛡️ Bulletproof Data Engine**:
  - Multi-stage retry fallback logic (`yf.download` $\rightarrow$ `yf.Ticker.history` $\rightarrow$ period fallback) preventing missing data errors.

---

## 📁 Repository Structure

```text
├── app.py                         # Main Streamlit Dashboard Application
├── Stock Predictions Model.keras  # Pre-trained LSTM Neural Network Model
├── Stock.py                       # Standalone Production Script
├── stock prediction.ipynb         # Model Training & EDA Notebook
├── first new model.ipynb          # Model Architecture Notebook
├── requirements.txt               # Python Dependencies
├── .gitignore                     # Git Exclusions
└── README.md                      # Project Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vinodgm2106/Stock-Price-Prediction-ML-with-Python-ML.git
cd Stock-Price-Prediction-ML-with-Python-ML
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model Architecture & Methodology

The core deep learning model is a **Sequential LSTM (Long Short-Term Memory)** neural network designed for time-series forecasting:
- **Input Layer**: 100 time steps of normalized closing stock prices (`MinMaxScaler(0, 1)`).
- **Layer 1**: LSTM (50 units, ReLU activation, Return Sequences) + Dropout (0.2).
- **Layer 2**: LSTM (60 units, ReLU activation, Return Sequences) + Dropout (0.3).
- **Layer 3**: LSTM (80 units, ReLU activation, Return Sequences) + Dropout (0.4).
- **Layer 4**: LSTM (120 units, ReLU activation) + Dropout (0.5).
- **Output Layer**: Dense (1 unit) output predicting the next time-step scaled stock price.
- **Optimizer**: Adam | **Loss Function**: Mean Squared Error (MSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Vinodgm2106/Stock-Price-Prediction-ML-with-Python-ML/issues).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
