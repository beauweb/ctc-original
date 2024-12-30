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

# Initialize exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

def get_market_data(symbol, timeframe, limit=100):
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None

def calculate_indicators(df):
    # Calculate technical indicators
    df['SMA20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['EMA20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close']).rsi()
    
    bb = BollingerBands(close=df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    
    return df

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

def main():
    st.set_page_config(page_title="Crypto Trading Assistant", layout="wide")
    
    st.title(" Crypto Trading Assistant")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Select Trading Pair",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )
    
    with col2:
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
    
    if st.button("Analyze Market", use_container_width=True):
        with st.spinner("Analyzing market data..."):
            # Get market data
            df = get_market_data(symbol, timeframe)
            if df is not None:
                # Calculate indicators
                df = calculate_indicators(df)
                
                # Generate trade setup
                setup = generate_trade_setup(df, symbol)
                
                if setup:
                    # Display results in a modern layout
                    st.markdown("### Market Analysis Results")
                    
                    # Metrics row 1
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${setup['current_price']:.2f}")
                    with col2:
                        st.metric("Trend", setup['trend'])
                    with col3:
                        st.metric("Volatility", setup['volatility'])
                    
                    # Metrics row 2
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Support", f"${setup['support']:.2f}")
                    with col2:
                        st.metric("Resistance", f"${setup['resistance']:.2f}")
                    with col3:
                        st.metric("Risk/Reward", f"{setup['risk_reward']:.2f}")
                    
                    # Trading signal
                    st.markdown("### Trading Signal")
                    signal_color = "green" if setup['signal'] == "Buy" else "red"
                    st.markdown(f"""
                        <div style='background-color: {signal_color}; padding: 20px; border-radius: 10px; color: white;'>
                            <h2 style='margin: 0;'>{setup['signal'].upper()} {symbol}</h2>
                            <p style='margin: 0;'>Confidence: {setup['confidence']}%</p>
                            <p style='margin: 0;'>ML Price Prediction: ${setup['ml_prediction']:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Price chart
                    st.markdown("### Price Chart")
                    chart_data = df[['close', 'SMA20', 'EMA20']].copy()
                    st.line_chart(chart_data)
                    
                    # Technical indicators
                    st.markdown("### Technical Indicators")
                    indicators_df = df[['RSI', 'BB_upper', 'BB_lower']].tail()
                    st.dataframe(indicators_df)
                else:
                    st.error("Failed to generate trade setup. Please try again.")
            else:
                st.error("Failed to fetch market data. Please try again.")

if __name__ == "__main__":
    main()
