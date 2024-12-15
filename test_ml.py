import pandas as pd
import numpy as np
from ml_predictions import MLPredictor
import ccxt
from ta import momentum, trend, volatility
import time

def get_market_data(symbol='BTC/USDT', timeframe='1m', limit=100):
    """Fetch market data from Delta Exchange"""
    try:
        exchange = ccxt.delta({
            'enableRateLimit': True
        })
        print(f"Fetching data for {symbol}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or df.empty:
        return None
    
    print("Calculating indicators...")
    # RSI
    df['RSI'] = momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = volatility.BollingerBands(df['close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    return df

def test_predictions():
    """Test ML predictions"""
    print("\n=== Starting ML Prediction Test ===")
    
    # Initialize predictor
    predictor = MLPredictor()
    
    # Get training data for each symbol
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    print("\nTraining models for each symbol...")
    
    for symbol in symbols:
        print(f"\nTraining for {symbol}")
        df = get_market_data(symbol, '1h', limit=500)
        if df is not None:
            df = calculate_indicators(df)
            if df is not None:
                print(f"Training models for {symbol}...")
                predictor.train_models(df, symbol)
                time.sleep(1)  # Avoid rate limits
    
    # Test predictions
    timeframes = ['1m', '5m', '15m']
    
    for symbol in symbols:
        print(f"\nTesting {symbol}")
        for timeframe in timeframes:
            print(f"\nTimeframe: {timeframe}")
            
            # Get market data
            df = get_market_data(symbol, timeframe, limit=100)
            if df is None:
                print(f"Skipping {symbol} {timeframe} due to data fetch error")
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            if df is None:
                print(f"Skipping {symbol} {timeframe} due to indicator calculation error")
                continue
            
            print(f"Data shape: {df.shape}")
            
            # Get predictions
            lstm_pred, rf_pred, confidence = predictor.predict(df, symbol)
            
            if lstm_pred is not None and rf_pred is not None:
                current_price = df['close'].iloc[-1]
                ensemble_pred = (lstm_pred + rf_pred) / 2
                
                # Get prediction signal
                signal = predictor.get_prediction_signal(current_price, ensemble_pred, confidence)
                
                print("\nPrediction Results:")
                print(f"Current Price: {current_price:.2f}")
                print(f"LSTM Prediction: {lstm_pred:.2f}")
                print(f"RF Prediction: {rf_pred:.2f}")
                print(f"Ensemble Prediction: {ensemble_pred:.2f}")
                print(f"Price Change: {signal['price_change_pct']:.2f}%")
                print(f"Confidence: {signal['confidence']:.2f}%")
                print(f"Signal: {signal['signal']} (Strength: {signal['strength']})")
            else:
                print("Failed to get predictions")
            
            # Wait a bit between requests to avoid rate limits
            time.sleep(1)

if __name__ == "__main__":
    test_predictions()
