## 🥇 Gold Price Prediction using LSTM

A Deep Learning-based time series forecasting system that predicts the next-day gold closing price using historical market data.
Deployed with Streamlit for interactive real-time predictions.

---

### 📌 Problem Statement
Gold prices are highly volatile and influenced by complex economic factors.
Traditional statistical models struggle to capture long-term temporal dependencies in financial data.

This project uses an LSTM (Long Short-Term Memory) network to:
1. Predict next-day gold closing prices
2. Identify whether prices are likely to go up or down
3. Assist in better investment and risk management decisions

---

### 📊 Dataset

Source: Yahoo Finance (via yfinance API)

Total Records: 6272

Type: Time-Series Data

Features Used:
1. Open
2. High
3. Low

Target Variable:
Close Price

Minimal preprocessing was required as the dataset was well-structured.

---

### ⚙️ Methodology
1️⃣ Data Preprocessing

1. Normalization using MinMaxScaler
2. Sliding window sequence generation
3. Train-test split
4. Time-series formatting for LSTM

2️⃣ Model Architecture

LSTM layers for temporal learning

Adam Optimizer

Loss Function: Mean Squared Error (MSE)

Epochs: 100

Batch Size: 32

LSTM was chosen because of its ability to capture long-term dependencies in sequential financial data.

---

### 📈 Model Performance
Metric	Value:

1. MAE =	0.0071
2. RMSE =	0.0107
3. R² Score =	0.9943
4. MAPE =	0.0131
5. Accuracy =	98.69%

The model demonstrates strong predictive capability with high R² and low error values.

---

### 🌐 Deployment

The model is deployed using Streamlit, allowing users to:

1. Fetch live gold price data
2. Visualize historical trends
3. Predict next-day closing price
4. Analyze market movement interactively

---

### 🛠 Tech Stack

Language: Python

Libraries:
1. TensorFlow / Keras
2. NumPy
3. Pandas
4. Scikit-learn
5. Matplotlib
6. Seaborn
7. yfinance

Deployment: Streamlit

## 🌍 Live Demo

[Click Here to Try the App](https://lstm-gold-price-forecasting-knbab6chktqtvzwfxwfk5k.streamlit.app)

