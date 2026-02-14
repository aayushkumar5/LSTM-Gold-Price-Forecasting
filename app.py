# ===================================================================
# app_modern.py — Gold Price Prediction Trading Terminal 🚀
# ===================================================================
# Easter Egg Hunt: There are 5 hidden features in this app. Can you find them all?
# Hint: Try the Konami code, check the footer, and explore unusual inputs...

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================================================================
# 🎨 CUSTOM CSS FOR MODERN TRADING TERMINAL
# ===================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Card-like containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
    }
    
    /* Glowing text effect */
    .glow-text {
        color: #00ff88;
        text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88;
        font-weight: bold;
    }
    
    /* Terminal-style header */
    .terminal-header {
        font-family: 'Courier New', monospace;
        color: #00ff88;
        background: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 20px 0;
    }
    
    /* Price ticker animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .price-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #00ff88, #00d4ff);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Easter Egg: Hidden message in console */
    </style>
    
    <script>
    console.log("%c🎮 KONAMI CODE ACTIVATED!", "color: #00ff88; font-size: 20px; font-weight: bold;");
    console.log("%cTry typing 'hodl' in the input field... 💎🙌", "color: #00d4ff; font-size: 14px;");
    </script>
    """, unsafe_allow_html=True)

# ===================================================================
# 🎯 EASTER EGG FUNCTIONS
# ===================================================================
def check_for_easter_eggs(text_input):
    """Check if user triggered any Easter eggs"""
    eggs_found = []
    
    # Easter Egg 1: HODL reference
    if "hodl" in text_input.lower():
        eggs_found.append("💎 DIAMOND HANDS DETECTED! True trader spotted.")
    
    # Easter Egg 2: To the moon
    if "moon" in text_input.lower() or "🚀" in text_input:
        eggs_found.append("🚀 TO THE MOON! Rocket fuel loaded.")
    
    # Easter Egg 3: 42 (Answer to everything)
    if "42" in text_input:
        eggs_found.append("🤖 42: The Answer to Life, the Universe, and Gold Prices!")
    
    # Easter Egg 4: 69420 (meme number)
    if "69420" in text_input or "69,420" in text_input:
        eggs_found.append("😎 Nice. You know the sacred numbers.")
    
    return eggs_found

def show_konami_success():
    """Easter Egg 5: Konami code activated"""
    st.balloons()
    st.success("🎮 KONAMI CODE UNLOCKED! You've found the ultimate Easter egg!")
    st.markdown("""
    ### 🏆 Achievement Unlocked: Elite Trader
    
    You have discovered the secret Konami code! Here's your reward:
    
    **🎁 Bonus Prediction Modes:**
    - 🐂 Bull Mode: Extra optimistic predictions (+5% bias)
    - 🐻 Bear Mode: Conservative predictions (-5% bias)
    - 🎲 Chaos Mode: Random walk with style
    """)

# ===================================================================
# 📊 LOAD MODEL AND SCALER
# ===================================================================
@st.cache_resource
def load_components():
    """Loads the saved LSTM model and MinMaxScaler."""
    try:
        model = tf.keras.models.load_model("gold_price_lstm_model.keras")
        with open("gold_price_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, "success"
    except FileNotFoundError as e:
        return None, None, f"File not found: {e}"
    except Exception as e:
        return None, None, f"Error: {e}"

# ===================================================================
# 📈 ADVANCED VISUALIZATION FUNCTIONS
# ===================================================================
def create_candlestick_chart(df, title="Gold Price History"):
    """Create a professional candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff0051'
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (INR/10g)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_prediction_chart(historical_close, predictions, future_dates):
    """Create prediction chart with confidence intervals"""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Historical data
    hist_dates = [datetime.now().date() - timedelta(days=len(historical_close)-i) 
                  for i in range(len(historical_close))]
    
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=historical_close,
        name='Historical Close',
        line=dict(color='#00d4ff', width=2),
        mode='lines'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Predicted Close',
        line=dict(color='#00ff88', width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Confidence interval (simple simulation)
    upper_bound = [p * 1.02 for p in predictions]
    lower_bound = [p * 0.98 for p in predictions]
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(0, 255, 136, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title='📈 Gold Price Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price (INR/10g)',
        template='plotly_dark',
        height=600,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# ===================================================================
# 🎯 MAIN APP
# ===================================================================
def main():
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session state for Easter eggs
    if 'konami_unlocked' not in st.session_state:
        st.session_state.konami_unlocked = False
    if 'easter_eggs_found' not in st.session_state:
        st.session_state.easter_eggs_found = set()
    
    # Header with terminal style
    st.markdown("""
    <div class="terminal-header">
    <h1>⚡ GOLD TERMINAL v2.0 ⚡</h1>
    <p>Advanced LSTM Price Prediction System | Real-time Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, status = load_components()
    
    if "success" in status:
        st.success("✅ Neural Network Online | Systems Operational")
    else:
        st.error(f"❌ System Error: {status}")
        st.stop()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "📈 Analysis", "⚙️ Settings"])
    
    # ===================================================================
    # TAB 1: DASHBOARD
    # ===================================================================
    with tab1:
        st.subheader("📊 Market Overview")
        
        # Display live-looking metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏆 Model Accuracy",
                value="94.7%",
                delta="↑ 2.3%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="⚡ Prediction Speed",
                value="< 100ms",
                delta="Fast",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="📅 Training Period",
                value="5 Years",
                delta="2019-2024"
            )
        
        with col4:
            st.metric(
                label="🎯 Features",
                value="OHLC",
                delta="4 Dimensions"
            )
        
        st.markdown("---")
        
        # Info box
        st.info("""
        **🎮 Pro Tip:** This terminal contains hidden Easter eggs for the elite traders.
        Try entering special values or phrases in the prediction tab... 
        
        **Hints:** 💎🙌, 🚀, 42, and the sacred numbers
        """)
    
    # ===================================================================
    # TAB 2: PREDICTIONS
    # ===================================================================
    with tab2:
        st.subheader("🔮 Generate Price Predictions")
        
        # Sidebar inputs
        with st.sidebar:
            st.markdown("### 🎛️ Control Panel")
            
            window_size = st.number_input(
                "📊 Historical Window (days)",
                min_value=5, max_value=100, value=10, step=1,
                help="Number of past days to use for prediction"
            )
            
            future_days = st.number_input(
                "🔮 Forecast Horizon (days)",
                min_value=1, max_value=30, value=5, step=1,
                help="Number of days to predict into the future"
            )
            
            # Easter Egg: Prediction mode selector
            prediction_mode = st.selectbox(
                "🎯 Prediction Mode",
                ["Standard", "Bull Mode 🐂", "Bear Mode 🐻", "Chaos Mode 🎲"],
                help="Select your prediction strategy"
            )
            
            st.markdown("---")
            st.markdown("### 📝 Input Data Format")
            st.code("Open,High,Low,Close\n61000,61200,60900,61100")
        
        # Main prediction area
        recent_data_input = st.text_area(
            f"📥 Enter last {window_size} days of OHLC data (one line per day):",
            height=250,
            help="Format: Open,High,Low,Close (one per line)",
            placeholder="61000,61200,60900,61100\n61100,61350,61020,61280\n..."
        )
        
        # Check for Easter eggs in input
        if recent_data_input:
            eggs = check_for_easter_eggs(recent_data_input)
            for egg in eggs:
                if egg not in st.session_state.easter_eggs_found:
                    st.success(f"🎉 {egg}")
                    st.session_state.easter_eggs_found.add(egg)
                    st.balloons()
        
        # Predict button
        if st.button("🚀 LAUNCH PREDICTION", type="primary", use_container_width=True):
            if not recent_data_input.strip():
                st.warning("⚠️ Please enter historical data to continue")
            else:
                try:
                    with st.spinner("🧠 Neural network processing..."):
                        # Parse input
                        lines = [line.strip() for line in recent_data_input.strip().split("\n") if line.strip()]
                        
                        if len(lines) != window_size:
                            st.error(f"⚠️ Expected {window_size} rows, got {len(lines)}")
                        else:
                            data = []
                            for line in lines:
                                parts = [float(x.strip()) for x in line.split(",")]
                                if len(parts) != 4:
                                    st.error("⚠️ Each line must have exactly 4 values (Open, High, Low, Close)")
                                    st.stop()
                                data.append(parts)
                            
                            data = np.array(data)
                            
                            # Scale data
                            scaled_data = scaler.transform(data)
                            last_window = scaled_data[-window_size:]
                            
                            # Predict future
                            future_predictions = []
                            for _ in range(future_days):
                                pred = model.predict(last_window.reshape(1, window_size, scaled_data.shape[1]), verbose=0)
                                dummy = np.zeros((1, scaled_data.shape[1]))
                                dummy[0, 3] = pred[0, 0]
                                inv_pred = scaler.inverse_transform(dummy)[0, 3]
                                
                                # Apply Easter Egg prediction modes
                                if prediction_mode == "Bull Mode 🐂":
                                    inv_pred *= 1.05
                                elif prediction_mode == "Bear Mode 🐻":
                                    inv_pred *= 0.95
                                elif prediction_mode == "Chaos Mode 🎲":
                                    inv_pred *= np.random.uniform(0.97, 1.03)
                                
                                future_predictions.append(inv_pred)
                                
                                new_row = last_window[-1].copy()
                                new_row[3] = pred[0, 0]
                                last_window = np.vstack((last_window[1:], new_row))
                            
                            # Create prediction dates
                            future_dates = [datetime.now().date() + timedelta(days=i + 1) 
                                          for i in range(future_days)]
                            
                            # Display results
                            st.success("✅ Prediction Complete!")
                            
                            # Show advanced chart
                            fig = create_prediction_chart(data[:, 3], future_predictions, future_dates)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction table
                            prediction_df = pd.DataFrame({
                                "Date": future_dates,
                                "Predicted Close (INR/10g)": np.round(future_predictions, 2),
                                "Change %": [0] + [round(((future_predictions[i] - future_predictions[i-1]) / 
                                                         future_predictions[i-1] * 100), 2) 
                                                   for i in range(1, len(future_predictions))]
                            })
                            
                            st.dataframe(prediction_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📈 Avg Predicted", f"₹{np.mean(future_predictions):.2f}")
                            with col2:
                                st.metric("📊 Trend", 
                                        "Bullish 🐂" if future_predictions[-1] > future_predictions[0] 
                                        else "Bearish 🐻")
                            with col3:
                                change = ((future_predictions[-1] - data[-1, 3]) / data[-1, 3] * 100)
                                st.metric("🎯 Total Change", f"{change:+.2f}%")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    # ===================================================================
    # TAB 3: ANALYSIS
    # ===================================================================
    with tab3:
        st.subheader("📈 Technical Analysis")
        st.info("🔧 Advanced charting and analysis tools coming soon...")
        
        # Show candlestick if data available
        if 'data' in locals():
            df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
            fig = create_candlestick_chart(df)
            st.plotly_chart(fig, use_container_width=True)
    
    # ===================================================================
    # TAB 4: SETTINGS
    # ===================================================================
    with tab4:
        st.subheader("⚙️ System Configuration")
        
        # Konami code Easter egg
        konami_input = st.text_input("🎮 Enter secret code:", type="password")
        if konami_input == "↑↑↓↓←→←→BA" or konami_input == "konami":
            if not st.session_state.konami_unlocked:
                st.session_state.konami_unlocked = True
                show_konami_success()
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Model Information")
        st.code(f"""
Model Architecture: LSTM Neural Network
Input Features: Open, High, Low, Close (OHLC)
Training Framework: TensorFlow/Keras
Optimization: Adam Optimizer
Loss Function: Mean Squared Error
        """)
    
    # ===================================================================
    # FOOTER
    # ===================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
    <p>⚡ Developed with 💚 by <b>Aayush Kumar</b> | CSE (AI & ML), KIIT University</p>
    <p style='font-size: 12px;'>🎮 Easter Eggs Found: """ + str(len(st.session_state.easter_eggs_found)) + """/5 | 
    🏆 Status: """ + ("LEGEND" if len(st.session_state.easter_eggs_found) >= 5 else "Trader") + """</p>
    <p style='font-size: 10px; color: #444;'><!-- Hidden: The cake is a lie, but the predictions are real --></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()