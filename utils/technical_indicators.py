import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """
    Calculate Simple Moving Average
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        window (int): Window size for moving average
    
    Returns:
        pandas.Series: Simple Moving Average
    """
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window):
    """
    Calculate Exponential Moving Average
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        window (int): Window size for moving average
    
    Returns:
        pandas.Series: Exponential Moving Average
    """
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        window (int): RSI window, typically 14
    
    Returns:
        pandas.Series: RSI values
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Handle subsequent calculations
    for i in range(window, len(delta)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window-1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window-1) + loss.iloc[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        fast (int): Fast EMA window
        slow (int): Slow EMA window
        signal (int): Signal line window
    
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate fast and slow EMAs
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        window (int): Window for moving average
        num_std (int): Number of standard deviations
    
    Returns:
        tuple: (Middle Band, Upper Band, Lower Band)
    """
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return middle_band, upper_band, lower_band

def calculate_atr(data, window=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        data (pandas.DataFrame): Stock data with 'High', 'Low', 'Close' prices
        window (int): ATR window
    
    Returns:
        pandas.Series: ATR values
    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=window).mean()
    
    return atr

def identify_trend(data, sma_short=20, sma_long=50):
    """
    Identify trend direction based on moving averages
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' prices
        sma_short (int): Short-term SMA window
        sma_long (int): Long-term SMA window
    
    Returns:
        str: 'Uptrend', 'Downtrend', or 'Sideways'
    """
    # Calculate short and long SMAs
    short_sma = calculate_sma(data, sma_short)
    long_sma = calculate_sma(data, sma_long)
    
    # Get the most recent values
    current_short_sma = short_sma.iloc[-1]
    current_long_sma = long_sma.iloc[-1]
    
    # Trend determination
    if current_short_sma > current_long_sma * 1.02:  # 2% buffer
        return "Uptrend"
    elif current_short_sma < current_long_sma * 0.98:  # 2% buffer
        return "Downtrend"
    else:
        return "Sideways"
