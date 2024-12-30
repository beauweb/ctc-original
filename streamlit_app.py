import streamlit as st
from app import get_market_data, calculate_indicators, generate_trade_setup

def main():
    st.title("Crypto Trading Assistant")
    
    # Add symbol selector
    symbol = st.selectbox(
        "Select Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    
    # Add timeframe selector
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    
    if st.button("Analyze"):
        with st.spinner("Fetching market data..."):
            # Get market data
            df = get_market_data(symbol, timeframe)
            if df is not None:
                # Calculate indicators
                df = calculate_indicators(df)
                
                # Generate trade setup
                setup = generate_trade_setup(df, symbol)
                
                # Display results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Price", f"${setup['current_price']:.2f}")
                    st.metric("Trend", setup['trend'])
                
                with col2:
                    st.metric("Volatility", setup['volatility'])
                    st.metric("ML Prediction", f"${setup['ml_prediction']:.2f}")
                
                # Display support and resistance levels
                st.subheader("Key Levels")
                st.write(f"Support: ${setup['support']:.2f}")
                st.write(f"Resistance: ${setup['resistance']:.2f}")
                
                # Display trading signals
                st.subheader("Trading Signals")
                st.write(f"Signal: {setup['signal']}")
                st.write(f"Confidence: {setup['confidence']}%")
                
                if 'risk_reward' in setup:
                    st.write(f"Risk/Reward Ratio: {setup['risk_reward']:.2f}")
            else:
                st.error("Failed to fetch market data. Please try again.")

if __name__ == "__main__":
    main()
