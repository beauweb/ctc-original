import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import time

# Initialize exchange
@st.cache_resource
def get_exchange():
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })

# Cache market data for 1 minute
@st.cache_data(ttl=60)
def get_market_data(symbol, timeframe, limit=100):
    try:
        exchange = get_exchange()
        # Add retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except ccxt.NetworkError as e:
                if attempt == max_retries - 1:  # Last attempt
                    st.error(f"Network error: {str(e)}. Please check your internet connection.")
                    return None
                time.sleep(retry_delay)
            except ccxt.ExchangeError as e:
                st.error(f"Exchange error: {str(e)}. The exchange may be experiencing issues.")
                return None
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None

@st.cache_data
def calculate_indicators(df):
    try:
        # Calculate technical indicators
        df['SMA20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['EMA20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['close']).rsi()
        
        bb = BollingerBands(close=df['close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return None

@st.cache_resource
def load_models(symbol):
    base_path = 'models'
    symbol_path = os.path.join(base_path, symbol.split('/')[0])
    
    try:
        lstm_model = tf.keras.models.load_model(os.path.join(symbol_path, 'lstm_model.h5'))
        rf_model = joblib.load(os.path.join(symbol_path, 'rf_model.joblib'))
        scaler = joblib.load(os.path.join(symbol_path, 'scaler.joblib'))
        return lstm_model, rf_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def generate_trade_setup(df, symbol):
    try:
        lstm_model, rf_model, scaler = load_models(symbol)
        
        if lstm_model is None or rf_model is None or scaler is None:
            return None
            
        # Prepare data for prediction
        features = df[['close', 'SMA20', 'EMA20', 'RSI']].values
        scaled_features = scaler.transform(features)
        
        # Make predictions
        lstm_pred = lstm_model.predict(np.array([scaled_features]))
        rf_pred = rf_model.predict(scaled_features)
        
        # Calculate trend and volatility
        current_price = df['close'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        bb_upper = df['BB_upper'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1]
        
        trend = "Bullish" if current_price > sma20 else "Bearish"
        volatility = "High" if (bb_upper - bb_lower) / current_price > 0.02 else "Low"
        
        # Calculate support and resistance
        support = df['low'].rolling(window=20).min().iloc[-1]
        resistance = df['high'].rolling(window=20).max().iloc[-1]
        
        # Generate trading signal
        signal = "Buy" if lstm_pred[0][0] > current_price and rf_pred[-1] > current_price else "Sell"
        confidence = int(abs(lstm_pred[0][0] - current_price) / current_price * 100)
        
        return {
            'current_price': current_price,
            'trend': trend,
            'volatility': volatility,
            'support': support,
            'resistance': resistance,
            'signal': signal,
            'confidence': confidence,
            'ml_prediction': (lstm_pred[0][0] + rf_pred[-1]) / 2,
            'risk_reward': abs(resistance - current_price) / abs(current_price - support)
        }
    except Exception as e:
        st.error(f"Error generating trade setup: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Crypto Trading Assistant",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("")
    st.markdown("---")
    
    # Add sidebar for settings
    with st.sidebar:
        st.header("")
        st.markdown("---")
        
        # Add refresh interval selector
        refresh_interval = st.slider(
            "",
            min_value=30,
            max_value=300,
            value=60,
            step=30
        )
        
        # Add auto-refresh checkbox
        auto_refresh = st.checkbox("", value=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )
    
    with col2:
        timeframe = st.selectbox(
            "",
            ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
    
    analyze_button = st.button("", use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)  # Prevent too frequent updates
        if int(time.time()) % refresh_interval == 0:
            analyze_button = True
    
    if analyze_button:
        with st.spinner(""):
            # Get market data
            df = get_market_data(symbol, timeframe)
            if df is not None:
                # Calculate indicators
                df = calculate_indicators(df)
                if df is not None:
                    # Generate trade setup
                    setup = generate_trade_setup(df, symbol)
                    
                    if setup:
                        # Display results in a modern layout
                        st.markdown("### ")
                        
                        # Metrics row 1
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("", f"${setup['current_price']:.2f}")
                        with col2:
                            st.metric("", setup['trend'])
                        with col3:
                            st.metric("", setup['volatility'])
                        
                        # Metrics row 2
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("", f"${setup['support']:.2f}")
                        with col2:
                            st.metric("", f"${setup['resistance']:.2f}")
                        with col3:
                            st.metric("", f"{setup['risk_reward']:.2f}")
                        
                        # Trading signal
                        st.markdown("### ")
                        signal_color = "green" if setup['signal'] == "Buy" else "red"
                        st.markdown(f"""
                            <div style='background-color: {signal_color}; padding: 20px; border-radius: 10px; color: white;'>
                                <h2 style='margin: 0;'>{setup['signal'].upper()} {symbol}</h2>
                                <p style='margin: 0;'>: {setup['confidence']}%</p>
                                <p style='margin: 0;'>ML : ${setup['ml_prediction']:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Price chart
                        st.markdown("### ")
                        chart_data = df[['close', 'SMA20', 'EMA20']].copy()
                        st.line_chart(chart_data)
                        
                        # Technical indicators
                        st.markdown("### ")
                        indicators_df = df[['RSI', 'BB_upper', 'BB_lower']].tail()
                        st.dataframe(indicators_df)
                        
                        # Last updated time
                        st.markdown("---")
                        st.markdown(f"*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
                    else:
                        st.error("")
                else:
                    st.error("")
            else:
                st.error("")

if __name__ == "__main__":
    main()
