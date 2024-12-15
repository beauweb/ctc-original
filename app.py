from flask import Flask, render_template, jsonify, request, redirect, url_for, session, send_file
from dotenv import load_dotenv
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import json
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility
from ml_predictions import MLPredictor

# Configure logging
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create log file with current date
current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_directory, f'trading_app_{current_date}.log')

# Time rotating handler (rotate daily and keep logs for 7 days)
file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',
    interval=1,
    backupCount=7
)

# Console handler
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
logger.handlers = []

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
SUPPORTED_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Updated to match Delta Exchange symbols
TIMEFRAMES = ['1m', '5m', '15m', '1h']
PASSWORD = 'Beaubhavik@2024$'

# Initialize Delta Exchange client
exchange = ccxt.delta({
    'apiKey': os.getenv('DELTA_API_KEY'),
    'secret': os.getenv('DELTA_SECRET_KEY'),
    'enableRateLimit': True
})

# Initialize ML predictor
ml_predictor = MLPredictor()

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

def get_market_data(symbol='BTC/USDT', timeframe='1m', limit=500):  # Increased limit from 100 to 500
    """Fetch market data from Delta Exchange"""
    try:
        # Just log the symbol we're trying to fetch
        logger.info(f"Fetching data for symbol: {symbol}")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        logger.info(f"Successfully fetched data for {symbol}")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or df.empty:
        return None
    
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
    
    # Calculate volatility (using ATR - Average True Range)
    atr = volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['ATR'] = atr.average_true_range()
    
    # Calculate Moving Averages
    df['SMA_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['SMA_50'] = trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['EMA_20'] = trend.EMAIndicator(df['close'], window=20).ema_indicator()
    
    # Stochastic Oscillator
    stoch = momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    return df

def get_volatility_level(atr_value, price):
    """Determine volatility level based on ATR percentage"""
    atr_percentage = (atr_value / price) * 100
    
    if atr_percentage < 1:
        return "Low"
    elif atr_percentage < 2:
        return "Moderate"
    elif atr_percentage < 3:
        return "High"
    else:
        return "Extreme"

def get_trend_analysis(df):
    """Analyze overall trend based on multiple indicators"""
    last_row = df.iloc[-1]
    price = last_row['close']
    sma_20 = last_row['SMA_20']
    sma_50 = last_row['SMA_50']
    ema_20 = last_row['EMA_20']
    
    trend_signals = []
    
    # Moving Average Analysis
    if price > sma_20 and price > sma_50 and sma_20 > sma_50:
        trend_signals.append("Strong Uptrend")
    elif price < sma_20 and price < sma_50 and sma_20 < sma_50:
        trend_signals.append("Strong Downtrend")
    elif price > sma_20 and sma_20 > sma_50:
        trend_signals.append("Moderate Uptrend")
    elif price < sma_20 and sma_20 < sma_50:
        trend_signals.append("Moderate Downtrend")
    else:
        trend_signals.append("Sideways")
    
    return trend_signals[0]

def get_key_levels(df):
    """Calculate key support and resistance levels"""
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    
    return {
        'resistance': recent_high,
        'support': recent_low,
        'pivot': (recent_high + recent_low + df.iloc[-1]['close']) / 3
    }

def generate_trade_setup(df, symbol):
    """Generate trade setup values based on technical analysis and ML predictions"""
    if df is None or df.empty:
        logger.error(f"Error: DataFrame is None or empty for {symbol}")
        return None
    
    last_row = df.iloc[-1]
    current_price = last_row['close']
    
    logger.info(f"\nGenerating trade setup for {symbol}")
    logger.info(f"Current price: {current_price}")
    
    # Get ML predictions
    try:
        lstm_pred, rf_pred, confidence = ml_predictor.predict(df, symbol)
        logger.info(f"LSTM prediction: {lstm_pred}")
        logger.info(f"RF prediction: {rf_pred}")
        logger.info(f"Raw confidence: {confidence}")
    except Exception as e:
        logger.error(f"Error getting ML predictions for {symbol}: {e}")
        return None
    
    # Initialize ML signal with default values
    ml_signal = {
        'predicted_price': current_price,  # Default to current price
        'price_change_pct': 0,
        'confidence': 0,
        'signal': 'NEUTRAL',
        'strength': 0
    }
    
    # Only update ML signal if predictions are valid
    if lstm_pred is not None and rf_pred is not None:
        ensemble_pred = (lstm_pred + rf_pred) / 2
        logger.info(f"Ensemble prediction: {ensemble_pred}")
        try:
            ml_signal = ml_predictor.get_prediction_signal(current_price, ensemble_pred, confidence)
            logger.info(f"ML Signal: {ml_signal}")
            
            # Update main signal based on ML prediction
            setup = {
                'current_price': current_price,
                'rsi': last_row['RSI'],
                'macd': last_row['MACD'],
                'macd_signal': last_row['MACD_signal'],
                'bb_upper': last_row['BB_upper'],
                'bb_lower': last_row['BB_lower'],
                'stoch_k': last_row['Stoch_k'],
                'stoch_d': last_row['Stoch_d'],
                'suggested_entry': ensemble_pred,
                'stop_loss': current_price * (0.99 if ml_signal['signal'] == 'BUY' else 1.01),  # 1% stop loss
                'take_profit': current_price * (1.02 if ml_signal['signal'] == 'BUY' else 0.98),  # 2% take profit
                'signal': ml_signal['signal'],  # Use ML signal directly
                'strength': ml_signal['strength'],  # Use ML signal strength
                'volatility': get_volatility_level(last_row['ATR'], current_price),
                'trend': get_trend_analysis(df),
                'key_levels': get_key_levels(df),
                'ml_prediction': ml_signal
            }
            return setup
            
        except Exception as e:
            logger.error(f"Error getting ML signal for {symbol}: {e}")
    
    # Get volatility level
    try:
        volatility_level = get_volatility_level(last_row['ATR'], current_price)
    except Exception as e:
        logger.error(f"Error getting volatility level for {symbol}: {e}")
        return None
    
    # Get trend analysis
    try:
        trend = get_trend_analysis(df)
    except Exception as e:
        logger.error(f"Error getting trend analysis for {symbol}: {e}")
        return None
    
    # Get key levels
    try:
        levels = get_key_levels(df)
    except Exception as e:
        logger.error(f"Error getting key levels for {symbol}: {e}")
        return None
    
    # Default setup if ML predictions fail
    setup = {
        'current_price': current_price,
        'rsi': last_row['RSI'],
        'macd': last_row['MACD'],
        'macd_signal': last_row['MACD_signal'],
        'bb_upper': last_row['BB_upper'],
        'bb_lower': last_row['BB_lower'],
        'stoch_k': last_row['Stoch_k'],
        'stoch_d': last_row['Stoch_d'],
        'suggested_entry': None,
        'stop_loss': None,
        'take_profit': None,
        'signal': 'NEUTRAL',
        'strength': 0,
        'volatility': volatility_level,
        'trend': trend,
        'key_levels': levels,
        'ml_prediction': ml_signal
    }
    
    return setup

@app.route('/')
@login_required
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/documentation')
@login_required
def documentation():
    """Render documentation page"""
    return send_file('documentation.html')

@app.route('/templates/ml_prediction_documentation.html')
@login_required
def ml_documentation():
    """Render ML prediction documentation"""
    return render_template('ml_prediction_documentation.html')

@app.route('/api/market-data')
@login_required
def market_data():
    """Get market data and analysis"""
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1m')
    
    if timeframe not in TIMEFRAMES:
        return jsonify({'error': 'Unsupported timeframe'}), 400
    
    df = get_market_data(symbol, timeframe)
    df = calculate_indicators(df)
    trade_setup = generate_trade_setup(df, symbol)
    
    if df is not None:
        data = {
            'trade_setup': trade_setup
        }
        return jsonify(data)
    else:
        return jsonify({'error': 'Failed to fetch market data'}), 500

if __name__ == '__main__':
    app.run(debug=True)
