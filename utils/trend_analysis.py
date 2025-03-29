import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
from utils.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd

def analyze_trend(data):
    """
    Analyze price trend of a stock.
    
    Args:
        data (pandas.DataFrame): Stock data with OHLC prices
        
    Returns:
        dict: Trend analysis results
    """
    if data is None or data.empty:
        return None
    
    try:
        # Calculate key indicators
        data['SMA20'] = calculate_sma(data, 20)
        data['SMA50'] = calculate_sma(data, 50)
        data['SMA200'] = calculate_sma(data, 200)
        data['EMA20'] = calculate_ema(data, 20)
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
        
        # Get the last row of data for analysis
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2] if len(data) > 1 else last_row
        
        # Identify price momentum
        price_change = ((last_row['Close'] / data['Close'].iloc[-20]) - 1) * 100
        
        # Trend signals based on moving averages
        ma_signals = []
        
        if last_row['Close'] > last_row['SMA20']:
            ma_signals.append("Price > SMA20 (Bullish)")
        else:
            ma_signals.append("Price < SMA20 (Bearish)")
            
        if last_row['SMA20'] > last_row['SMA50']:
            ma_signals.append("SMA20 > SMA50 (Bullish)")
        else:
            ma_signals.append("SMA20 < SMA50 (Bearish)")
            
        if last_row['SMA50'] > last_row['SMA200']:
            ma_signals.append("SMA50 > SMA200 (Bullish - Golden Cross)")
        else:
            ma_signals.append("SMA50 < SMA200 (Bearish - Death Cross)")
        
        # RSI signals
        if last_row['RSI'] > 70:
            rsi_signal = "Overbought (RSI > 70)"
        elif last_row['RSI'] < 30:
            rsi_signal = "Oversold (RSI < 30)"
        else:
            rsi_signal = "Neutral"
        
        # MACD signals
        if last_row['MACD'] > last_row['Signal'] and prev_row['MACD'] <= prev_row['Signal']:
            macd_signal = "Bullish Crossover"
        elif last_row['MACD'] < last_row['Signal'] and prev_row['MACD'] >= prev_row['Signal']:
            macd_signal = "Bearish Crossover"
        elif last_row['MACD'] > last_row['Signal']:
            macd_signal = "Bullish"
        else:
            macd_signal = "Bearish"
        
        # Volume analysis
        avg_volume = data['Volume'].iloc[-20:].mean()
        last_volume = last_row['Volume']
        volume_signal = "High" if last_volume > avg_volume * 1.5 else "Normal" if last_volume > avg_volume * 0.5 else "Low"
        
        # Overall trend determination
        bullish_signals = sum(1 for signal in ma_signals if "Bullish" in signal)
        bearish_signals = sum(1 for signal in ma_signals if "Bearish" in signal)
        
        if bullish_signals >= 2 and last_row['RSI'] > 50 and macd_signal in ["Bullish", "Bullish Crossover"]:
            overall_trend = "Strong Uptrend"
        elif bullish_signals >= 2:
            overall_trend = "Uptrend"
        elif bearish_signals >= 2 and last_row['RSI'] < 50 and macd_signal in ["Bearish", "Bearish Crossover"]:
            overall_trend = "Strong Downtrend"
        elif bearish_signals >= 2:
            overall_trend = "Downtrend"
        else:
            overall_trend = "Sideways/Neutral"
        
        # Package results
        results = {
            'Last Close': last_row['Close'],
            'Price Change (20 days)': f"{price_change:.2f}%",
            'MA Signals': ma_signals,
            'RSI': last_row['RSI'],
            'RSI Signal': rsi_signal,
            'MACD Signal': macd_signal,
            'Volume': f"{int(last_volume):,}",
            'Volume Signal': volume_signal,
            'Overall Trend': overall_trend
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error analyzing trend: {str(e)}")
        return None

def predict_short_term_trend(data, days=5):
    """
    Predict short-term price trend using linear regression.
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        days (int): Number of days to predict
        
    Returns:
        pandas.DataFrame: DataFrame with actual and predicted prices
    """
    if data is None or data.empty or len(data) < 30:
        return None
    
    try:
        # Use last 30 days for prediction
        close_prices = data['Close'].iloc[-30:].values.reshape(-1, 1)
        
        # Create features (day index)
        days_index = np.array(range(len(close_prices))).reshape(-1, 1)
        
        # Create and fit linear regression model
        model = LinearRegression()
        model.fit(days_index, close_prices)
        
        # Predict next few days
        last_day = len(close_prices) - 1
        next_days = np.array(range(last_day + 1, last_day + days + 1)).reshape(-1, 1)
        predicted_prices = model.predict(next_days)
        
        # Create prediction dataframe
        last_date = data.index[-1]
        date_range = pd.date_range(start=last_date, periods=days+1)[1:]
        
        pred_df = pd.DataFrame({
            'Predicted_Price': [price[0] for price in predicted_prices]
        }, index=date_range)
        
        return pred_df
    
    except Exception as e:
        st.error(f"Error predicting trend: {str(e)}")
        return None

def get_support_resistance_levels(data):
    """
    Calculate support and resistance levels.
    
    Args:
        data (pandas.DataFrame): Stock data with OHLC prices
        
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    if data is None or data.empty or len(data) < 20:
        return None, None
    
    try:
        # Use closing prices for last 100 days or all available data
        prices = data['Close'].iloc[-min(100, len(data)):].values
        
        # Create histogram to find price clusters
        hist, bin_edges = np.histogram(prices, bins=20)
        
        # Find local maxima in histogram (price levels with high frequency)
        level_indices = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                level_indices.append(i)
        
        # Convert indices to price levels
        levels = [(bin_edges[i] + bin_edges[i+1])/2 for i in level_indices]
        
        # Determine support and resistance based on current price
        current_price = data['Close'].iloc[-1]
        support_levels = sorted([level for level in levels if level < current_price])
        resistance_levels = sorted([level for level in levels if level > current_price])
        
        # Limit number of levels
        support_levels = support_levels[-3:] if len(support_levels) > 3 else support_levels
        resistance_levels = resistance_levels[:3] if len(resistance_levels) > 3 else resistance_levels
        
        return support_levels, resistance_levels
    
    except Exception as e:
        st.error(f"Error calculating support/resistance: {str(e)}")
        return None, None
