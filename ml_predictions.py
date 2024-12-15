import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import tensorflow as tf
import ta

class MLPredictor:
    def __init__(self, model_path='models'):
        """Initialize ML predictor with model path"""
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.lstm_models = {}
        self.rf_models = {}
        self.scalers = {}
        
        # Initialize TensorFlow to use CPU
        tf.config.set_visible_devices([], 'GPU')
        
        # Try to load existing models
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        for symbol in self.symbols:
            self.load_or_train_models(pd.DataFrame(), symbol)

    def load_models(self, symbol):
        """Load models for a specific symbol"""
        print(f"\nAttempting to load models for {symbol}")
        try:
            # Get the base symbol (e.g., 'BTC' from 'BTC/USDT')
            base_symbol = symbol.split('/')[0]
            
            # Define model paths using the correct directory structure
            symbol_path = os.path.join(self.model_path, base_symbol)
            lstm_path = os.path.join(symbol_path, 'lstm_model.h5')
            rf_path = os.path.join(symbol_path, 'rf_model.joblib')
            scaler_path = os.path.join(symbol_path, 'scaler.joblib')
            
            print(f"Looking for models at:")
            print(f"LSTM: {lstm_path}")
            print(f"RF: {rf_path}")
            print(f"Scaler: {scaler_path}")
            
            # Check if model files exist
            if not all(os.path.exists(path) for path in [lstm_path, rf_path, scaler_path]):
                print(f"Error: Missing model files for {symbol}")
                print(f"LSTM exists: {os.path.exists(lstm_path)}")
                print(f"RF exists: {os.path.exists(rf_path)}")
                print(f"Scaler exists: {os.path.exists(scaler_path)}")
                return False
            
            # Load LSTM model
            self.lstm_models[symbol] = load_model(lstm_path)
            print(f"Successfully loaded LSTM model for {symbol}")
            
            # Load Random Forest model
            self.rf_models[symbol] = joblib.load(rf_path)
            print(f"Successfully loaded RF model for {symbol}")
            
            # Load scaler
            self.scalers[symbol] = joblib.load(scaler_path)
            print(f"Successfully loaded scaler for {symbol}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def prepare_data(self, df, symbol, sequence_length=60):
        """Prepare data for ML models"""
        print(f"\nPreparing data for {symbol}")
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {df.columns.tolist()}")
        
        # Create scaler if it doesn't exist
        if symbol not in self.scalers:
            print(f"Creating new scaler for {symbol}")
            self.scalers[symbol] = MinMaxScaler()
        
        try:
            # Calculate technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['price_ma'] = df['close'].rolling(window=20).mean()
            
            print("Calculated technical indicators")
            print(f"Data shape after indicators: {df.shape}")
            
            # Drop NaN values
            df = df.dropna()
            print(f"Data shape after dropping NaN: {df.shape}")
            
            if df.empty:
                print(f"Error: All data was dropped for {symbol}")
                return np.array([]), np.array([]), pd.DataFrame()
            
            # Select features
            features = ['close', 'volume', 'returns', 'volatility', 'rsi', 
                       'macd', 'bb_upper', 'bb_lower', 'volume_ma', 'price_ma']
            
            # Scale features
            feature_data = df[features].values
            print(f"Feature data shape: {feature_data.shape}")
            
            if len(feature_data) == 0:
                print(f"Error: No feature data available for {symbol}")
                return np.array([]), np.array([]), pd.DataFrame()
            
            # During training, fit the scaler. During prediction, use existing transformation
            if not hasattr(self.scalers[symbol], 'n_features_in_'):
                print("Fitting scaler...")
                scaled_data = self.scalers[symbol].fit_transform(feature_data)
            else:
                print("Using existing scaler...")
                scaled_data = self.scalers[symbol].transform(feature_data)
            
            print(f"Scaled data shape: {scaled_data.shape}")
            
            # Prepare sequences for LSTM
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length, 0])  # Predict next close price
            
            X = np.array(X)
            y = np.array(y)
            print(f"Final X shape: {X.shape}")
            print(f"Final y shape: {y.shape}")
            
            return X, y, df[features].iloc[sequence_length:]
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return np.array([]), np.array([]), pd.DataFrame()

    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss=MeanSquaredError())
        return model

    def train_models(self, df, symbol, force=False):
        """Train models for a specific symbol"""
        print(f"\nTraining models for {symbol}")
        try:
            # Get the base symbol (e.g., 'BTC' from 'BTC/USDT')
            base_symbol = symbol.split('/')[0]
            symbol_path = os.path.join(self.model_path, base_symbol)
            
            if not os.path.exists(symbol_path):
                os.makedirs(symbol_path)
            
            # Prepare paths
            lstm_path = os.path.join(symbol_path, 'lstm_model.h5')
            rf_path = os.path.join(symbol_path, 'rf_model.joblib')
            scaler_path = os.path.join(symbol_path, 'scaler.joblib')
            
            # Check if we need to train
            if not force and all(os.path.exists(p) for p in [lstm_path, rf_path, scaler_path]):
                # Check if models are less than 24 hours old
                if all(time.time() - os.path.getmtime(p) < 24*60*60 for p in [lstm_path, rf_path, scaler_path]):
                    print(f"Recent models exist for {symbol}, skipping training")
                    return True
            
            print(f"Training new models for {symbol}")
            
            # Prepare data
            X, y, features_df = self.prepare_data(df, symbol)
            if len(X) == 0 or len(y) == 0:
                print(f"Error: No data available for training for {symbol}")
                return False
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            features_train = features_df.iloc[:split_idx]
            features_test = features_df.iloc[split_idx:]
            
            # Train LSTM
            print("Training LSTM model...")
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1,
                         validation_data=(X_test, y_test))
            
            # Train Random Forest
            print("Training Random Forest model...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(features_train, y_train)
            
            # Save models
            print("Saving models...")
            lstm_model.save(lstm_path)
            joblib.dump(rf_model, rf_path)
            joblib.dump(self.scalers[symbol], scaler_path)
            
            # Store models in memory
            self.lstm_models[symbol] = lstm_model
            self.rf_models[symbol] = rf_model
            
            print(f"Successfully trained and saved models for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error training models for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def load_or_train_models(self, df, symbol):
        """Load existing models or train new ones if needed"""
        if not self.load_models(symbol):
            print(f"Could not load models for {symbol}, training new ones...")
            return self.train_models(df, symbol)
        return True

    def predict(self, df, symbol, sequence_length=60):
        """Make predictions using both models for a specific symbol"""
        print(f"\nMaking prediction for {symbol}")
        
        if symbol not in self.lstm_models or symbol not in self.rf_models:
            print(f"Models not found in memory for {symbol}")
            if not self.load_or_train_models(df, symbol):
                print(f"Error: Models not loaded for {symbol}")
                return None, None, 0
        
        try:
            print(f"Data shape before preparation: {df.shape}")
            print(f"Columns available: {df.columns.tolist()}")
            
            # Prepare data
            X, _, features_df = self.prepare_data(df, symbol, sequence_length)
            print(f"Prepared data shape: {X.shape if len(X) > 0 else 'Empty'}")
            print(f"Features shape: {features_df.shape if not features_df.empty else 'Empty'}")
            
            if len(X) == 0:
                print(f"Error: No data available for prediction for {symbol}")
                return None, None, 0
            
            # Get the last sequence and features
            last_sequence = X[-1:]
            last_features = features_df.iloc[-1:].values
            current_price = df['close'].iloc[-1]
            print(f"Current price: {current_price}")
            print(f"Last sequence shape: {last_sequence.shape}")
            print(f"Last features shape: {last_features.shape}")
            
            # LSTM prediction (scaled)
            print("Making LSTM prediction...")
            lstm_scaled_pred = self.lstm_models[symbol].predict(last_sequence, verbose=0)
            print(f"LSTM scaled prediction shape: {lstm_scaled_pred.shape}")
            print(f"LSTM scaled prediction value: {lstm_scaled_pred}")
            
            # Create a full feature vector for inverse transform
            dummy_features = np.zeros((lstm_scaled_pred.shape[0], 9))  # 9 other features
            full_scaled_pred = np.concatenate([lstm_scaled_pred, dummy_features], axis=1)
            print(f"Full scaled prediction shape: {full_scaled_pred.shape}")
            
            # Inverse transform to get actual price
            lstm_pred = self.scalers[symbol].inverse_transform(full_scaled_pred)[:, 0]
            print(f"LSTM prediction: {lstm_pred[0]}")
            
            # Random Forest prediction (directly predicts price change)
            print("Making RF prediction...")
            rf_change_pred = self.rf_models[symbol].predict(last_features)[0]
            rf_pred = current_price * (1 + rf_change_pred)
            print(f"RF change prediction: {rf_change_pred}")
            print(f"RF final prediction: {rf_pred}")
            
            # Calculate confidence based on prediction agreement
            price_diff = abs(lstm_pred[0] - rf_pred)
            avg_price = (lstm_pred[0] + rf_pred) / 2
            confidence = float(np.clip(1 - (price_diff / avg_price), 0, 1))
            
            print(f"Predictions summary:")
            print(f"  LSTM: {lstm_pred[0]:.2f}")
            print(f"  RF: {rf_pred:.2f}")
            print(f"  Price difference: {price_diff:.2f}")
            print(f"  Average price: {avg_price:.2f}")
            print(f"  Confidence: {confidence:.2%}")
            
            return float(lstm_pred[0]), float(rf_pred), confidence
            
        except Exception as e:
            print(f"Error during prediction for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None, 0

    def get_prediction_signal(self, current_price, predicted_price, confidence):
        """Generate trading signal based on predicted price and confidence"""
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # More sensitive thresholds
        signal = 'NEUTRAL'
        if abs(price_change_pct) < 0.1:  # Reduced from 0.2%
            signal = 'NEUTRAL'
        elif price_change_pct > 0:
            signal = 'BUY'
        else:
            signal = 'SELL'
        
        # Calculate signal strength (1-5)
        raw_strength = min(abs(price_change_pct) / 0.1, 5)  # Scale change % to 1-5, max at 0.5%
        strength = round(raw_strength * confidence)  # Weight by confidence
        
        return {
            'predicted_price': float(predicted_price),
            'price_change_pct': float(price_change_pct),
            'confidence': float(confidence),
            'signal': signal,
            'strength': int(strength)
        }
