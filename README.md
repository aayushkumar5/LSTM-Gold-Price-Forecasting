🥇 Gold Price Prediction using LSTM

A Deep Learning-based time series forecasting system that predicts the next-day gold closing price using historical market data.

Deployed with Streamlit for interactive real-time predictions.

📌 Problem Statement

Gold prices are highly volatile and influenced by complex economic factors.
Traditional statistical models struggle to capture long-term temporal dependencies in financial data.

This project uses an LSTM (Long Short-Term Memory) network to:

Predict next-day gold closing prices

Identify whether prices are likely to go up or down

Assist in better investment and risk management decisions

📊 Dataset

Source: Yahoo Finance (via yfinance API)

Total Records: 6272

Type: Time-Series Data

Features Used:

Open

High

Low

Target Variable:

Close Price

Minimal preprocessing was required as the dataset was well-structured.

⚙️ Methodology
1️⃣ Data Preprocessing

Normalization using MinMaxScaler

Sliding window sequence generation

Train-test split

Time-series formatting for LSTM

2️⃣ Model Architecture

LSTM layers for temporal learning

Adam Optimizer

Loss Function: Mean Squared Error (MSE)

Epochs: 100

Batch Size: 32

LSTM was chosen because of its ability to capture long-term dependencies in sequential financial data.

📈 Model Performance
Metric	Value
MAE	0.0071
RMSE	0.0107
R² Score	0.9943
MAPE	0.0131
Accuracy	98.69%

The model demonstrates strong predictive capability with high R² and low error values.

🌐 Deployment

The model is deployed using Streamlit, allowing users to:

Fetch live gold price data

Visualize historical trends

Predict next-day closing price

Analyze market movement interactively

🛠 Tech Stack

Language: Python

Libraries:

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

yfinance

Deployment:

Streamlit
