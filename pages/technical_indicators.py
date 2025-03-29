import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from utils.data_fetcher import fetch_stock_data
from utils.technical_indicators import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, identify_trend
)

def display_technical_indicators():
    st.header("📊 Technical Indicators")
    
    # Stock ticker input
    ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
    
    if ticker_input:
        ticker = ticker_input.upper().strip()
        
        # Data period selection
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "Select Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                index=2  # Default to 6 months
            )
        with col2:
            interval = st.selectbox(
                "Select Data Interval",
                ["1d", "5d", "1wk", "1mo"],
                index=0
            )
        
        # Fetch stock data
        data = fetch_stock_data(ticker, period=period, interval=interval)
        
        if data is not None and not data.empty:
            # Create tabs for different indicators
            tabs = st.tabs([
                "Moving Averages", 
                "Oscillators", 
                "Volatility Indicators", 
                "Combined View",
                "Trading Signals"
            ])
            
            with tabs[0]:
                display_moving_averages(data, ticker)
            
            with tabs[1]:
                display_oscillators(data, ticker)
            
            with tabs[2]:
                display_volatility_indicators(data, ticker)
            
            with tabs[3]:
                display_combined_view(data, ticker)
            
            with tabs[4]:
                display_trading_signals(data, ticker)
        else:
            st.error(f"Error fetching data for ticker: {ticker}. Please check the symbol and try again.")

def display_moving_averages(data, ticker):
    st.subheader("Moving Averages")
    
    # Parameters selection for moving averages
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sma_short = st.number_input("Short SMA Period", min_value=5, max_value=50, value=20, step=1)
    with col2:
        sma_medium = st.number_input("Medium SMA Period", min_value=20, max_value=100, value=50, step=5)
    with col3:
        sma_long = st.number_input("Long SMA Period", min_value=50, max_value=200, value=200, step=10)
    
    # Calculate moving averages
    data['SMA_short'] = calculate_sma(data, sma_short)
    data['SMA_medium'] = calculate_sma(data, sma_medium)
    data['SMA_long'] = calculate_sma(data, sma_long)
    data['EMA_short'] = calculate_ema(data, sma_short)
    data['EMA_medium'] = calculate_ema(data, sma_medium)
    
    # Create the figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Price",
            line=dict(color='#000000', width=1)
        )
    )
    
    # Add SMA lines
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_short'],
            name=f"SMA {sma_short}",
            line=dict(color='#1f77b4', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_medium'],
            name=f"SMA {sma_medium}",
            line=dict(color='#ff7f0e', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_long'],
            name=f"SMA {sma_long}",
            line=dict(color='#2ca02c', width=1)
        )
    )
    
    # Add EMA lines
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_short'],
            name=f"EMA {sma_short}",
            line=dict(color='#d62728', width=1, dash='dash')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_medium'],
            name=f"EMA {sma_medium}",
            line=dict(color='#9467bd', width=1, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving Average Analysis
    st.subheader("Moving Average Analysis")
    
    # Get the most recent values
    last_price = data['Close'].iloc[-1]
    last_sma_short = data['SMA_short'].iloc[-1]
    last_sma_medium = data['SMA_medium'].iloc[-1]
    last_sma_long = data['SMA_long'].iloc[-1]
    last_ema_short = data['EMA_short'].iloc[-1]
    last_ema_medium = data['EMA_medium'].iloc[-1]
    
    # Create columns for analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Price vs. Moving Averages")
        
        # Check price position relative to moving averages
        ma_signals = []
        
        if last_price > last_sma_short:
            ma_signals.append(f"✅ Price > SMA {sma_short} (Bullish)")
        else:
            ma_signals.append(f"❌ Price < SMA {sma_short} (Bearish)")
            
        if last_price > last_sma_medium:
            ma_signals.append(f"✅ Price > SMA {sma_medium} (Bullish)")
        else:
            ma_signals.append(f"❌ Price < SMA {sma_medium} (Bearish)")
            
        if last_price > last_sma_long:
            ma_signals.append(f"✅ Price > SMA {sma_long} (Bullish)")
        else:
            ma_signals.append(f"❌ Price < SMA {sma_long} (Bearish)")
        
        for signal in ma_signals:
            st.markdown(signal)
    
    with col2:
        st.markdown("##### Moving Average Crossovers")
        
        # Check for crossovers
        crossover_signals = []
        
        if last_sma_short > last_sma_medium:
            crossover_signals.append(f"✅ SMA {sma_short} > SMA {sma_medium} (Bullish)")
        else:
            crossover_signals.append(f"❌ SMA {sma_short} < SMA {sma_medium} (Bearish)")
            
        if last_sma_medium > last_sma_long:
            crossover_signals.append(f"✅ SMA {sma_medium} > SMA {sma_long} (Bullish - Golden Cross)")
        else:
            crossover_signals.append(f"❌ SMA {sma_medium} < SMA {sma_long} (Bearish - Death Cross)")
            
        if last_ema_short > last_ema_medium:
            crossover_signals.append(f"✅ EMA {sma_short} > EMA {sma_medium} (Bullish)")
        else:
            crossover_signals.append(f"❌ EMA {sma_short} < EMA {sma_medium} (Bearish)")
        
        for signal in crossover_signals:
            st.markdown(signal)
    
    # Moving Average Values
    st.subheader("Moving Average Values")
    
    value_cols = st.columns(3)
    
    with value_cols[0]:
        st.metric("Current Price", f"${last_price:.2f}")
        st.metric(f"SMA {sma_short}", f"${last_sma_short:.2f}")
    
    with value_cols[1]:
        st.metric(f"SMA {sma_medium}", f"${last_sma_medium:.2f}")
        st.metric(f"EMA {sma_short}", f"${last_ema_short:.2f}")
    
    with value_cols[2]:
        st.metric(f"SMA {sma_long}", f"${last_sma_long:.2f}")
        st.metric(f"EMA {sma_medium}", f"${last_ema_medium:.2f}")
    
    # Moving Average Theory
    with st.expander("Moving Average Theory"):
        st.markdown("""
        ### Understanding Moving Averages
        
        Moving averages smooth out price data to create a single flowing line, making it easier to identify the direction of the trend.
        
        #### Simple Moving Average (SMA)
        - **Short-term SMA (e.g., 20-day)**: Reacts quickly to price changes; useful for identifying short-term trends
        - **Medium-term SMA (e.g., 50-day)**: Balances responsiveness and smoothing; commonly used as a trend separator
        - **Long-term SMA (e.g., 200-day)**: Identifies the primary long-term trend; crossing above/below often signals major trend shifts
        
        #### Exponential Moving Average (EMA)
        - Places greater weight on more recent price data
        - Reacts more quickly to price changes than SMA
        - Usually more sensitive to new information
        
        #### Moving Average Crossovers
        - **Golden Cross**: When a shorter-term MA crosses above a longer-term MA (bullish)
        - **Death Cross**: When a shorter-term MA crosses below a longer-term MA (bearish)
        
        #### Using Moving Averages
        - Moving averages work best in trending markets
        - Multiple timeframe analysis can provide confirmation of trends
        - Price crossing above/below a moving average can signal potential trend changes
        - Moving averages can act as dynamic support and resistance levels
        """)

def display_oscillators(data, ticker):
    st.subheader("Oscillators")
    
    # Parameters selection
    col1, col2 = st.columns(2)
    
    with col1:
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1)
    
    with col2:
        macd_options = st.multiselect(
            "MACD Parameters",
            ["12,26,9", "8,17,9", "5,35,5"],
            default=["12,26,9"]
        )
        
        # Default MACD values if none selected
        if not macd_options:
            macd_options = ["12,26,9"]
    
    # Create tabs for different oscillators
    osc_tabs = st.tabs(["RSI", "MACD", "Stochastic"])
    
    with osc_tabs[0]:
        display_rsi(data, ticker, rsi_period)
    
    with osc_tabs[1]:
        display_macd(data, ticker, macd_options)
    
    with osc_tabs[2]:
        display_stochastic(data, ticker)

def display_rsi(data, ticker, period=14):
    st.markdown(f"### Relative Strength Index (RSI-{period})")
    
    # Calculate RSI
    data['RSI'] = calculate_rsi(data, window=period)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add price to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add RSI to subplot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1, annotation_text="Neutral")
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - RSI ({period})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Analysis
    st.markdown("#### RSI Analysis")
    
    # Get the most recent RSI value
    last_rsi = data['RSI'].iloc[-1]
    rsi_trend = last_rsi - data['RSI'].iloc[-6] if len(data) >= 6 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI value and condition
        if last_rsi > 70:
            rsi_condition = "Overbought"
            rsi_color = "red"
        elif last_rsi < 30:
            rsi_condition = "Oversold"
            rsi_color = "green"
        else:
            rsi_condition = "Neutral"
            rsi_color = "gray"
        
        st.markdown(f"**Current RSI:** <span style='color:{rsi_color};font-weight:bold'>{last_rsi:.2f} ({rsi_condition})</span>", unsafe_allow_html=True)
        
        # RSI trend
        trend_color = "green" if rsi_trend > 0 else "red" if rsi_trend < 0 else "gray"
        trend_text = "Rising" if rsi_trend > 0 else "Falling" if rsi_trend < 0 else "Flat"
        st.markdown(f"**5-Period Trend:** <span style='color:{trend_color}'>{trend_text} ({rsi_trend:.2f})</span>", unsafe_allow_html=True)
    
    with col2:
        # RSI interpretation
        st.markdown("**Signal Interpretation:**")
        
        if last_rsi > 70:
            st.markdown("❗ **Potential Selling Opportunity**: RSI indicates the stock may be overvalued")
        elif last_rsi < 30:
            st.markdown("💡 **Potential Buying Opportunity**: RSI indicates the stock may be undervalued")
        elif last_rsi > 50 and rsi_trend > 0:
            st.markdown("📈 **Bullish Momentum**: RSI is above 50 and rising")
        elif last_rsi < 50 and rsi_trend < 0:
            st.markdown("📉 **Bearish Momentum**: RSI is below 50 and falling")
        else:
            st.markdown("↔️ **No Clear Signal**: RSI is in neutral territory")
    
    # RSI Theory
    with st.expander("RSI Theory"):
        st.markdown("""
        ### Understanding RSI (Relative Strength Index)
        
        RSI is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.
        
        #### Key Levels:
        - **Above 70**: Traditionally considered overbought (potential reversal or correction)
        - **Below 30**: Traditionally considered oversold (potential reversal or bounce)
        - **50 Level**: Acts as a centerline; above 50 suggests bullish momentum, below 50 suggests bearish momentum
        
        #### RSI Signals:
        1. **Overbought/Oversold Conditions**: When RSI reaches extreme levels (>70 or <30)
        2. **Divergence**: When price makes a new high/low but RSI fails to confirm
        3. **Failure Swings**: When RSI fails to make a new high during an uptrend or new low during a downtrend
        4. **Centerline Crossover**: When RSI crosses above or below the 50 level
        
        #### Limitations:
        - During strong trends, RSI can remain in overbought/oversold territory for extended periods
        - False signals can occur, especially in choppy or sideways markets
        - Best used in conjunction with other indicators for confirmation
        """)

def display_macd(data, ticker, macd_options):
    st.markdown("### Moving Average Convergence Divergence (MACD)")
    
    # Process MACD options
    if not macd_options or len(macd_options) == 0:
        macd_options = ["12,26,9"]
    
    # Parse the first option to get default values
    default_option = macd_options[0].split(",")
    fast_period = int(default_option[0])
    slow_period = int(default_option[1])
    signal_period = int(default_option[2])
    
    # Calculate MACD
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(
        data, fast=fast_period, slow=slow_period, signal=signal_period
    )
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add price to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add MACD to subplot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name="MACD",
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Signal'],
            name="Signal",
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Add histogram
    colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Histogram'],
            name="Histogram",
            marker=dict(color=colors),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - MACD ({fast_period},{slow_period},{signal_period})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # MACD Analysis
    st.markdown("#### MACD Analysis")
    
    # Get the most recent values
    last_macd = data['MACD'].iloc[-1]
    last_signal = data['Signal'].iloc[-1]
    last_hist = data['Histogram'].iloc[-1]
    
    # Check for crossovers
    prev_macd = data['MACD'].iloc[-2] if len(data) > 1 else last_macd
    prev_signal = data['Signal'].iloc[-2] if len(data) > 1 else last_signal
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**MACD Values:**")
        st.markdown(f"- MACD Line: {last_macd:.4f}")
        st.markdown(f"- Signal Line: {last_signal:.4f}")
        st.markdown(f"- Histogram: {last_hist:.4f}")
        
        # MACD position relative to signal
        if last_macd > last_signal:
            st.markdown("✅ **MACD is above Signal Line (Bullish)**")
        else:
            st.markdown("❌ **MACD is below Signal Line (Bearish)**")
    
    with col2:
        st.markdown("**MACD Signals:**")
        
        # Check for crossovers
        if last_macd > last_signal and prev_macd <= prev_signal:
            st.markdown("📈 **Bullish Crossover**: MACD crossed above Signal Line (Buy Signal)")
        elif last_macd < last_signal and prev_macd >= prev_signal:
            st.markdown("📉 **Bearish Crossover**: MACD crossed below Signal Line (Sell Signal)")
        
        # Check histogram
        if last_hist > 0 and last_hist > data['Histogram'].iloc[-2] if len(data) > 1 else 0:
            st.markdown("📈 **Increasing Positive Histogram (Strong Bullish)**")
        elif last_hist > 0 and last_hist < data['Histogram'].iloc[-2] if len(data) > 1 else 0:
            st.markdown("⚠️ **Decreasing Positive Histogram (Weakening Bullish)**")
        elif last_hist < 0 and last_hist < data['Histogram'].iloc[-2] if len(data) > 1 else 0:
            st.markdown("📉 **Decreasing Negative Histogram (Strong Bearish)**")
        elif last_hist < 0 and last_hist > data['Histogram'].iloc[-2] if len(data) > 1 else 0:
            st.markdown("⚠️ **Increasing Negative Histogram (Weakening Bearish)**")
    
    # MACD Theory
    with st.expander("MACD Theory"):
        st.markdown("""
        ### Understanding MACD (Moving Average Convergence Divergence)
        
        MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
        
        #### Components:
        - **MACD Line**: Difference between a fast EMA and a slow EMA (typically 12 and 26 periods)
        - **Signal Line**: EMA of the MACD Line (typically 9 periods)
        - **Histogram**: Difference between MACD Line and Signal Line
        
        #### Key Signals:
        1. **Crossovers**: When MACD crosses above/below Signal Line (buy/sell signals)
        2. **Zero Line Crossovers**: When MACD crosses above/below zero (bullish/bearish shift)
        3. **Divergence**: When MACD diverges from price (potential reversal)
        4. **Histogram Changes**: Direction changes in histogram can signal momentum shifts
        
        #### Interpretation:
        - **MACD above zero**: Bullish momentum
        - **MACD below zero**: Bearish momentum
        - **Rising MACD**: Increasing bullish momentum
        - **Falling MACD**: Increasing bearish momentum
        
        #### Common Settings:
        - Standard: (12,26,9) - Default parameters
        - Fast: (8,17,9) - More responsive, shorter term
        - Slow: (5,35,5) - Less responsive, longer term
        """)

def display_stochastic(data, ticker):
    st.markdown("### Stochastic Oscillator")
    
    # Calculate Stochastic Oscillator
    k_period = 14
    d_period = 3
    
    # Calculate %K (Fast Stochastic)
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    data['%K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    
    # Calculate %D (Slow Stochastic)
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add price to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add Stochastic to subplot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['%K'],
            name="%K",
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['%D'],
            name="%D",
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Add overbought/oversold lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Stochastic Oscillator ({k_period},{d_period})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Stochastic", row=2, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stochastic Analysis
    st.markdown("#### Stochastic Oscillator Analysis")
    
    # Get the most recent values
    last_k = data['%K'].iloc[-1]
    last_d = data['%D'].iloc[-1]
    
    # Check for crossovers
    prev_k = data['%K'].iloc[-2] if len(data) > 1 else last_k
    prev_d = data['%D'].iloc[-2] if len(data) > 1 else last_d
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Stochastic Values:**")
        st.markdown(f"- %K (Fast): {last_k:.2f}")
        st.markdown(f"- %D (Slow): {last_d:.2f}")
        
        # Overbought/Oversold conditions
        if last_k > 80 and last_d > 80:
            st.markdown("🔴 **Overbought**: Both %K and %D above 80")
        elif last_k < 20 and last_d < 20:
            st.markdown("🟢 **Oversold**: Both %K and %D below 20")
        elif last_k > 80:
            st.markdown("⚠️ **%K is in Overbought territory**")
        elif last_k < 20:
            st.markdown("⚠️ **%K is in Oversold territory**")
    
    with col2:
        st.markdown("**Stochastic Signals:**")
        
        # Check for crossovers
        if last_k > last_d and prev_k <= prev_d:
            if last_k < 20 and last_d < 20:
                st.markdown("💡 **Strong Buy Signal**: %K crossed above %D in oversold territory")
            else:
                st.markdown("📈 **Bullish Crossover**: %K crossed above %D")
        elif last_k < last_d and prev_k >= prev_d:
            if last_k > 80 and last_d > 80:
                st.markdown("💡 **Strong Sell Signal**: %K crossed below %D in overbought territory")
            else:
                st.markdown("📉 **Bearish Crossover**: %K crossed below %D")
        
        # Current position
        if last_k > last_d:
            st.markdown("✅ **%K is above %D (Bullish)**")
        else:
            st.markdown("❌ **%K is below %D (Bearish)**")
    
    # Stochastic Theory
    with st.expander("Stochastic Oscillator Theory"):
        st.markdown("""
        ### Understanding Stochastic Oscillator
        
        The Stochastic Oscillator is a momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period of time.
        
        #### Components:
        - **%K (Fast Stochastic)**: Main line representing the current price relative to the high-low range
        - **%D (Slow Stochastic)**: Moving average of %K (typically 3-period)
        
        #### Key Levels:
        - **Above 80**: Overbought territory (potential selling opportunity)
        - **Below 20**: Oversold territory (potential buying opportunity)
        
        #### Key Signals:
        1. **Crossovers**: When %K crosses above/below %D (buy/sell signals)
        2. **Overbought/Oversold Reversal**: When Stochastic moves from overbought/oversold back toward the middle
        3. **Divergence**: When Stochastic diverges from price (potential reversal)
        4. **Bull/Bear Set-ups**: Particular pattern formations within the Stochastic
        
        #### Interpretation:
        - **Rising Stochastic**: Increasing upward momentum
        - **Falling Stochastic**: Increasing downward momentum
        - **Strongest signals** occur when crossovers happen in overbought or oversold territory
        
        #### Types:
        - **Fast Stochastic**: More sensitive, generates more signals (but potentially more false ones)
        - **Slow Stochastic**: Less sensitive, generates fewer but potentially more reliable signals
        """)

def display_volatility_indicators(data, ticker):
    st.subheader("Volatility Indicators")
    
    # Parameters selection
    bb_period = st.slider("Bollinger Bands Period", min_value=5, max_value=50, value=20, step=1)
    bb_std = st.slider("Bollinger Bands Standard Deviation", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
    # Calculate Bollinger Bands
    middle, upper, lower = calculate_bollinger_bands(data, window=bb_period, num_std=bb_std)
    data['BB_Middle'] = middle
    data['BB_Upper'] = upper
    data['BB_Lower'] = lower
    
    # Calculate other volatility indicators
    data['ATR'] = calculate_atr(data)
    
    # Calculate Bollinger Band Width
    data['BB_Width'] = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']) * 100
    
    # Create tabs for different volatility indicators
    vol_tabs = st.tabs(["Bollinger Bands", "Average True Range", "Bollinger Band Width"])
    
    with vol_tabs[0]:
        display_bollinger_bands(data, ticker, bb_period, bb_std)
    
    with vol_tabs[1]:
        display_atr(data, ticker)
    
    with vol_tabs[2]:
        display_bbw(data, ticker, bb_period, bb_std)

def display_bollinger_bands(data, ticker, period, std):
    st.markdown(f"### Bollinger Bands ({period}, {std})")
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        )
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            name="Upper Band",
            line=dict(color='rgba(250, 128, 114, 0.7)', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            name="Middle Band (SMA)",
            line=dict(color='rgba(73, 134, 230, 0.8)', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            name="Lower Band",
            line=dict(color='rgba(144, 238, 144, 0.7)', width=1),
            fill='tonexty', 
            fillcolor='rgba(73, 134, 230, 0.1)'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bollinger Bands Analysis
    st.markdown("#### Bollinger Bands Analysis")
    
    # Get the most recent values
    last_price = data['Close'].iloc[-1]
    last_upper = data['BB_Upper'].iloc[-1]
    last_middle = data['BB_Middle'].iloc[-1]
    last_lower = data['BB_Lower'].iloc[-1]
    
    # Calculate Bollinger %B
    bb_percent_b = (last_price - last_lower) / (last_upper - last_lower)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Band Values:**")
        st.markdown(f"- Upper Band: ${last_upper:.2f}")
        st.markdown(f"- Middle Band: ${last_middle:.2f}")
        st.markdown(f"- Lower Band: ${last_lower:.2f}")
        st.markdown(f"- Current Price: ${last_price:.2f}")
        
        # Calculate %B
        st.markdown(f"- %B: {bb_percent_b:.2f}")
    
    with col2:
        st.markdown("**Bollinger Band Signals:**")
        
        # Price position relative to bands
        if last_price > last_upper:
            st.markdown("⚠️ **Price Above Upper Band**: Potential overbought condition")
        elif last_price < last_lower:
            st.markdown("⚠️ **Price Below Lower Band**: Potential oversold condition")
        elif last_price > last_middle:
            st.markdown("📈 **Price Above Middle Band**: Bullish bias")
        else:
            st.markdown("📉 **Price Below Middle Band**: Bearish bias")
        
        # Squeeze condition
        recent_width = data['BB_Width'].iloc[-20:].mean() if len(data) >= 20 else data['BB_Width'].mean()
        current_width = data['BB_Width'].iloc[-1]
        
        if current_width < recent_width * 0.9:
            st.markdown("🔍 **Bollinger Squeeze**: Bands are narrowing, potential breakout ahead")
        elif current_width > recent_width * 1.5:
            st.markdown("↔️ **Wide Bands**: High volatility, potential trend continuation")
    
    # Bollinger Bands Theory
    with st.expander("Bollinger Bands Theory"):
        st.markdown("""
        ### Understanding Bollinger Bands
        
        Bollinger Bands consist of a middle band (simple moving average) with an upper and lower band (standard deviations away from the middle band), creating a price channel that varies based on volatility.
        
        #### Components:
        - **Middle Band**: Usually a 20-period simple moving average
        - **Upper Band**: Middle band + (usually 2) standard deviations
        - **Lower Band**: Middle band - (usually 2) standard deviations
        
        #### Key Concepts:
        1. **Volatility Measure**: The width of the bands indicates market volatility
        2. **Mean Reversion**: Prices tend to return to the middle band over time
        3. **Trend Identification**: Direction of the middle band indicates the overall trend
        4. **Support/Resistance**: The bands can act as dynamic support and resistance
        
        #### Key Signals:
        - **Bollinger Bounce**: Price bouncing off the bands back toward the middle (common in ranging markets)
        - **Bollinger Squeeze**: Bands narrowing significantly (low volatility), often precedes a strong move
        - **Walking the Bands**: Price consistently touching/following along a band indicates a strong trend
        - **W-Bottoms and M-Tops**: Specific patterns that form along the bands
        
        #### %B Indicator:
        - **%B = 1.0**: Price is at the upper band
        - **%B = 0.5**: Price is at the middle band
        - **%B = 0.0**: Price is at the lower band
        - **%B > 1.0**: Price is above the upper band
        - **%B < 0.0**: Price is below the lower band
        """)

def display_atr(data, ticker):
    st.markdown("### Average True Range (ATR)")
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add price to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add ATR to subplot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['ATR'],
            name="ATR (14)",
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Average True Range (ATR)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ATR Analysis
    st.markdown("#### ATR Analysis")
    
    # Get the most recent ATR value
    last_atr = data['ATR'].iloc[-1]
    last_price = data['Close'].iloc[-1]
    atr_percent = (last_atr / last_price) * 100
    
    atr_trend = last_atr - data['ATR'].iloc[-6] if len(data) >= 6 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ATR Values:**")
        st.markdown(f"- Current ATR (14): ${last_atr:.2f}")
        st.markdown(f"- ATR as % of Price: {atr_percent:.2f}%")
        
        if atr_trend > 0:
            st.markdown("📈 **Increasing Volatility**: ATR is rising")
        elif atr_trend < 0:
            st.markdown("📉 **Decreasing Volatility**: ATR is falling")
        else:
            st.markdown("↔️ **Stable Volatility**: ATR is flat")
    
    with col2:
        st.markdown("**ATR Application:**")
        
        # Potential stop loss levels
        st.markdown("**Potential Stop Loss Levels:**")
        st.markdown(f"- Tight (1 ATR): ${last_price - last_atr:.2f}")
        st.markdown(f"- Medium (2 ATR): ${last_price - (2 * last_atr):.2f}")
        st.markdown(f"- Wide (3 ATR): ${last_price - (3 * last_atr):.2f}")
    
    # ATR Theory
    with st.expander("ATR Theory"):
        st.markdown("""
        ### Understanding Average True Range (ATR)
        
        ATR is a volatility indicator that shows how much a stock price typically moves over time.
        
        #### What ATR Measures:
        ATR measures volatility by calculating the average range of price movement over a specified period (typically 14 periods). It accounts for gaps by using the True Range, which is the greatest of:
        - Current high minus current low
        - Current high minus previous close (absolute value)
        - Current low minus previous close (absolute value)
        
        #### Interpreting ATR:
        - **Higher ATR**: Indicates higher volatility
        - **Lower ATR**: Indicates lower volatility
        - **Increasing ATR**: Volatility is rising (often occurs during market reversals, especially downturns)
        - **Decreasing ATR**: Volatility is falling (often occurs as a trend matures)
        
        #### Common Applications:
        1. **Position Sizing**: Adjusting position size based on volatility
        2. **Stop Loss Placement**: Setting stops at a multiple of ATR from entry
        3. **Breakout Confirmation**: Validating breakouts with ATR expansion
        4. **Volatility Comparison**: Comparing volatility across different stocks or time periods
        
        #### Key Points:
        - ATR is not directional; it only measures volatility
        - ATR should be considered relative to the stock's price (as a percentage)
        - ATR tends to be mean-reverting; periods of high volatility are typically followed by lower volatility
        - ATR is often used with other indicators to confirm signals
        """)

def display_bbw(data, ticker, period, std):
    st.markdown(f"### Bollinger Band Width (BBW)")
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add price to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands to main plot
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Upper'],
            name="Upper Band",
            line=dict(color='rgba(250, 128, 114, 0.7)', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Lower'],
            name="Lower Band",
            line=dict(color='rgba(144, 238, 144, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(73, 134, 230, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add BB Width to subplot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Width'],
            name="BB Width",
            line=dict(color='orange', width=1)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Bollinger Band Width",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="BB Width %", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # BB Width Analysis
    st.markdown("#### Bollinger Band Width Analysis")
    
    # Get the most recent BBW value
    last_bbw = data['BB_Width'].iloc[-1]
    
    # Get historical percentiles for context
    low_percentile = np.percentile(data['BB_Width'].dropna(), 20)
    high_percentile = np.percentile(data['BB_Width'].dropna(), 80)
    
    bbw_trend = last_bbw - data['BB_Width'].iloc[-6] if len(data) >= 6 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**BB Width Values:**")
        st.markdown(f"- Current BB Width: {last_bbw:.2f}%")
        st.markdown(f"- 20th Percentile: {low_percentile:.2f}%")
        st.markdown(f"- 80th Percentile: {high_percentile:.2f}%")
        
        if last_bbw < low_percentile:
            st.markdown("🔍 **Low Volatility**: BB Width is in the bottom 20% (potential squeeze)")
        elif last_bbw > high_percentile:
            st.markdown("📊 **High Volatility**: BB Width is in the top 20%")
        else:
            st.markdown("↔️ **Normal Volatility**: BB Width is in the normal range")
    
    with col2:
        st.markdown("**BB Width Signals:**")
        
        if bbw_trend > 0:
            st.markdown("📈 **Expanding Bands**: Volatility is increasing")
            
            if last_bbw > high_percentile:
                st.markdown("⚠️ **Strong Trend Continuation Likely**")
            else:
                st.markdown("🔄 **Potential Start of New Trend**")
                
        elif bbw_trend < 0:
            st.markdown("📉 **Contracting Bands**: Volatility is decreasing")
            
            if last_bbw < low_percentile:
                st.markdown("🔍 **Bollinger Squeeze**: Potential breakout ahead")
            else:
                st.markdown("🔄 **Trend May Be Weakening**")
        else:
            st.markdown("↔️ **Stable Bands**: Volatility is unchanged")
    
    # BB Width Theory
    with st.expander("Bollinger Band Width Theory"):
        st.markdown("""
        ### Understanding Bollinger Band Width (BBW)
        
        Bollinger Band Width (BBW) measures the spread between the upper and lower Bollinger Bands relative to the middle band, expressed as a percentage.
        
        #### Formula:
        BBW = ((Upper Band - Lower Band) / Middle Band) × 100
        
        #### Key Concepts:
        1. **Volatility Cycle**: Markets alternate between periods of high and low volatility
        2. **Bollinger Squeeze**: When BBW reaches very low levels, indicating contracted bands and low volatility
        3. **Volatility Expansion**: When BBW rises rapidly, indicating expanding bands and increasing volatility
        
        #### Trading Applications:
        - **Breakout Trading**: Enter trades when BBW expands after a squeeze
        - **Trend Trading**: Use BBW expansion to confirm strength of trends
        - **Mean Reversion**: Look for extremely high BBW values which may indicate unsustainable price moves
        - **Volatility Analysis**: Compare current BBW to historical levels to gauge relative volatility
        
        #### Key Points:
        - BBW is non-directional (doesn't indicate direction of movement)
        - Low BBW often precedes significant price movements
        - High BBW often indicates continuation of existing trends
        - BBW tends to be cyclical, alternating between periods of expansion and contraction
        - BBW is most useful when viewed in context of historical percentiles for the specific security
        """)

def display_combined_view(data, ticker):
    st.subheader("Combined Technical View")
    
    # Calculate indicators
    data['SMA20'] = calculate_sma(data, 20)
    data['SMA50'] = calculate_sma(data, 50)
    data['SMA200'] = calculate_sma(data, 200)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    data['BB_Middle'], data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=("Price with Moving Averages and Bollinger Bands", "Volume", "MACD", "RSI")
    )
    
    # Add price and bands to main plot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA20'], name="SMA 20", line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA50'], name="SMA 50", line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA200'], name="SMA 200", line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Upper'], 
            name="Upper BB",
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Lower'], 
            name="Lower BB",
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 0, 0.05)'
        ),
        row=1, col=1
    )
    
    # Add volume
    colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker=dict(color=colors),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name="MACD",
            line=dict(color='blue', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Signal'],
            name="Signal",
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Add MACD histogram
    colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Histogram'],
            name="Histogram",
            marker=dict(color=colors),
            opacity=0.7
        ),
        row=3, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        row=4, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Combined Technical View",
        xaxis_title="Date",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Summary
    st.subheader("Technical Summary")
    
    # Identify overall trend
    trend = identify_trend(data)
    
    # Get the most recent values
    last_price = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    last_macd = data['MACD'].iloc[-1]
    last_signal = data['Signal'].iloc[-1]
    
    # Create columns for summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Trend Analysis")
        st.markdown(f"**Overall Trend:** {trend}")
        
        # MA trend
        ma_bullish = sum([
            1 if last_price > data['SMA20'].iloc[-1] else 0,
            1 if last_price > data['SMA50'].iloc[-1] else 0,
            1 if last_price > data['SMA200'].iloc[-1] else 0,
            1 if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1] else 0,
            1 if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1] else 0
        ])
        
        if ma_bullish >= 4:
            st.markdown("✅ **Moving Averages: Strong Bullish**")
        elif ma_bullish >= 3:
            st.markdown("✅ **Moving Averages: Bullish**")
        elif ma_bullish <= 1:
            st.markdown("❌ **Moving Averages: Strong Bearish**")
        elif ma_bullish <= 2:
            st.markdown("❌ **Moving Averages: Bearish**")
        else:
            st.markdown("➖ **Moving Averages: Neutral**")
    
    with col2:
        st.markdown("#### Momentum")
        
        # RSI
        if last_rsi > 70:
            st.markdown("🔴 **RSI: Overbought** (%.2f)" % last_rsi)
        elif last_rsi < 30:
            st.markdown("🟢 **RSI: Oversold** (%.2f)" % last_rsi)
        elif last_rsi > 50:
            st.markdown("✅ **RSI: Bullish** (%.2f)" % last_rsi)
        else:
            st.markdown("❌ **RSI: Bearish** (%.2f)" % last_rsi)
        
        # MACD
        if last_macd > last_signal:
            st.markdown("✅ **MACD: Bullish**")
        else:
            st.markdown("❌ **MACD: Bearish**")
    
    with col3:
        st.markdown("#### Volatility")
        
        # Bollinger Bands
        if last_price > data['BB_Upper'].iloc[-1]:
            st.markdown("⚠️ **Bollinger Bands: Price above upper band (Overbought)**")
        elif last_price < data['BB_Lower'].iloc[-1]:
            st.markdown("⚠️ **Bollinger Bands: Price below lower band (Oversold)**")
        else:
            band_position = (last_price - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])
            if band_position > 0.8:
                st.markdown("📈 **Bollinger Bands: Price near upper band**")
            elif band_position < 0.2:
                st.markdown("📉 **Bollinger Bands: Price near lower band**")
            else:
                st.markdown("↔️ **Bollinger Bands: Price within bands**")

def display_trading_signals(data, ticker):
    st.subheader("Trading Signals")
    
    # Calculate all indicators
    data['SMA20'] = calculate_sma(data, 20)
    data['SMA50'] = calculate_sma(data, 50)
    data['SMA200'] = calculate_sma(data, 200)
    data['EMA20'] = calculate_ema(data, 20)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    data['BB_Middle'], data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data)
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    
    # MA Crossover Signals
    signals['SMA20_cross_SMA50'] = 0
    signals['SMA50_cross_SMA200'] = 0
    
    # Calculate SMA20/SMA50 crossover
    for i in range(1, len(data)):
        if data['SMA20'].iloc[i] > data['SMA50'].iloc[i] and data['SMA20'].iloc[i-1] <= data['SMA50'].iloc[i-1]:
            signals['SMA20_cross_SMA50'].iloc[i] = 1  # Bullish crossover
        elif data['SMA20'].iloc[i] < data['SMA50'].iloc[i] and data['SMA20'].iloc[i-1] >= data['SMA50'].iloc[i-1]:
            signals['SMA20_cross_SMA50'].iloc[i] = -1  # Bearish crossover
    
    # Calculate SMA50/SMA200 crossover (Golden/Death Cross)
    for i in range(1, len(data)):
        if data['SMA50'].iloc[i] > data['SMA200'].iloc[i] and data['SMA50'].iloc[i-1] <= data['SMA200'].iloc[i-1]:
            signals['SMA50_cross_SMA200'].iloc[i] = 1  # Golden Cross
        elif data['SMA50'].iloc[i] < data['SMA200'].iloc[i] and data['SMA50'].iloc[i-1] >= data['SMA200'].iloc[i-1]:
            signals['SMA50_cross_SMA200'].iloc[i] = -1  # Death Cross
    
    # RSI Signals
    signals['RSI_Signal'] = 0
    for i in range(1, len(data)):
        if data['RSI'].iloc[i] < 30 and data['RSI'].iloc[i-1] >= 30:
            signals['RSI_Signal'].iloc[i] = 1  # Oversold
        elif data['RSI'].iloc[i] > 70 and data['RSI'].iloc[i-1] <= 70:
            signals['RSI_Signal'].iloc[i] = -1  # Overbought
    
    # MACD Crossover Signals
    signals['MACD_Signal'] = 0
    for i in range(1, len(data)):
        if data['MACD'].iloc[i] > data['Signal'].iloc[i] and data['MACD'].iloc[i-1] <= data['Signal'].iloc[i-1]:
            signals['MACD_Signal'].iloc[i] = 1  # Bullish crossover
        elif data['MACD'].iloc[i] < data['Signal'].iloc[i] and data['MACD'].iloc[i-1] >= data['Signal'].iloc[i-1]:
            signals['MACD_Signal'].iloc[i] = -1  # Bearish crossover
    
    # Bollinger Band Signals
    signals['BB_Signal'] = 0
    for i in range(1, len(data)):
        if data['Close'].iloc[i] < data['BB_Lower'].iloc[i] and data['Close'].iloc[i-1] >= data['BB_Lower'].iloc[i-1]:
            signals['BB_Signal'].iloc[i] = 1  # Price crosses below lower band
        elif data['Close'].iloc[i] > data['BB_Upper'].iloc[i] and data['Close'].iloc[i-1] <= data['BB_Upper'].iloc[i-1]:
            signals['BB_Signal'].iloc[i] = -1  # Price crosses above upper band
    
    # Filter out rows with no signals
    trading_signals = signals[(signals['SMA20_cross_SMA50'] != 0) | 
                            (signals['SMA50_cross_SMA200'] != 0) | 
                            (signals['RSI_Signal'] != 0) | 
                            (signals['MACD_Signal'] != 0) | 
                            (signals['BB_Signal'] != 0)]
    
    # Sort in reverse chronological order
    trading_signals = trading_signals.sort_index(ascending=False)
    
    # Format the signals for display
    display_signals = trading_signals.copy()
    display_signals.index = display_signals.index.strftime('%Y-%m-%d')
    display_signals = display_signals.rename(columns={'Price': 'Close Price'})
    
    # Create a new column for signal description
    signal_descriptions = []
    
    for idx, row in trading_signals.iterrows():
        signals_on_day = []
        
        if row['SMA20_cross_SMA50'] == 1:
            signals_on_day.append("🟢 SMA20 crossed above SMA50 (Bullish)")
        elif row['SMA20_cross_SMA50'] == -1:
            signals_on_day.append("🔴 SMA20 crossed below SMA50 (Bearish)")
        
        if row['SMA50_cross_SMA200'] == 1:
            signals_on_day.append("🟢 Golden Cross: SMA50 crossed above SMA200 (Strong Bullish)")
        elif row['SMA50_cross_SMA200'] == -1:
            signals_on_day.append("🔴 Death Cross: SMA50 crossed below SMA200 (Strong Bearish)")
        
        if row['RSI_Signal'] == 1:
            signals_on_day.append("🟢 RSI entered oversold territory (Potential Buy)")
        elif row['RSI_Signal'] == -1:
            signals_on_day.append("🔴 RSI entered overbought territory (Potential Sell)")
        
        if row['MACD_Signal'] == 1:
            signals_on_day.append("🟢 MACD crossed above Signal line (Bullish)")
        elif row['MACD_Signal'] == -1:
            signals_on_day.append("🔴 MACD crossed below Signal line (Bearish)")
        
        if row['BB_Signal'] == 1:
            signals_on_day.append("🟢 Price crossed below lower Bollinger Band (Potential Buy)")
        elif row['BB_Signal'] == -1:
            signals_on_day.append("🔴 Price crossed above upper Bollinger Band (Potential Sell)")
        
        signal_descriptions.append("<br>".join(signals_on_day))
    
    display_signals['Signal Description'] = signal_descriptions
    
    # Display recent trading signals
    st.markdown("### Recent Trading Signals")
    
    if not trading_signals.empty:
        # Display only needed columns
        st.dataframe({
            'Date': display_signals.index,
            'Price': display_signals['Close Price'].apply(lambda x: f"${x:.2f}"),
            'Signal': display_signals['Signal Description']
        }, hide_index=True)
    else:
        st.info("No trading signals detected in the selected time period.")
    
    # Current Technical Signals Summary
    st.markdown("### Current Technical Signals")
    
    # Get the most recent data
    last_data = data.iloc[-1]
    prev_data = data.iloc[-2] if len(data) > 1 else last_data
    
    # Create a list of current signals
    current_signals = []
    
    # Moving Averages
    if last_data['Close'] > last_data['SMA20']:
        current_signals.append(("MA Position", "🟢 Price > SMA20 (Bullish)"))
    else:
        current_signals.append(("MA Position", "🔴 Price < SMA20 (Bearish)"))
    
    if last_data['Close'] > last_data['SMA50']:
        current_signals.append(("MA Position", "🟢 Price > SMA50 (Bullish)"))
    else:
        current_signals.append(("MA Position", "🔴 Price < SMA50 (Bearish)"))
    
    if last_data['SMA20'] > last_data['SMA50']:
        current_signals.append(("MA Cross", "🟢 SMA20 > SMA50 (Bullish)"))
    else:
        current_signals.append(("MA Cross", "🔴 SMA20 < SMA50 (Bearish)"))
    
    if last_data['SMA50'] > last_data['SMA200']:
        current_signals.append(("Golden/Death Cross", "🟢 SMA50 > SMA200 (Golden Cross - Bullish)"))
    else:
        current_signals.append(("Golden/Death Cross", "🔴 SMA50 < SMA200 (Death Cross - Bearish)"))
    
    # RSI
    if last_data['RSI'] > 70:
        current_signals.append(("RSI", f"🔴 RSI: {last_data['RSI']:.2f} (Overbought)"))
    elif last_data['RSI'] < 30:
        current_signals.append(("RSI", f"🟢 RSI: {last_data['RSI']:.2f} (Oversold)"))
    elif last_data['RSI'] > 50:
        current_signals.append(("RSI", f"🟢 RSI: {last_data['RSI']:.2f} (Bullish)"))
    else:
        current_signals.append(("RSI", f"🔴 RSI: {last_data['RSI']:.2f} (Bearish)"))
    
    # MACD
    if last_data['MACD'] > last_data['Signal']:
        current_signals.append(("MACD", "🟢 MACD > Signal (Bullish)"))
    else:
        current_signals.append(("MACD", "🔴 MACD < Signal (Bearish)"))
    
    if last_data['MACD'] > last_data['Signal'] and prev_data['MACD'] <= prev_data['Signal']:
        current_signals.append(("MACD Cross", "🟢 MACD crossed above Signal (Bullish Signal)"))
    elif last_data['MACD'] < last_data['Signal'] and prev_data['MACD'] >= prev_data['Signal']:
        current_signals.append(("MACD Cross", "🔴 MACD crossed below Signal (Bearish Signal)"))
    
    # Bollinger Bands
    if last_data['Close'] > last_data['BB_Upper']:
        current_signals.append(("Bollinger Bands", "🔴 Price > Upper BB (Overbought)"))
    elif last_data['Close'] < last_data['BB_Lower']:
        current_signals.append(("Bollinger Bands", "🟢 Price < Lower BB (Oversold)"))
    else:
        bb_position = (last_data['Close'] - last_data['BB_Lower']) / (last_data['BB_Upper'] - last_data['BB_Lower'])
        if bb_position > 0.8:
            current_signals.append(("Bollinger Bands", "🟡 Price near Upper BB"))
        elif bb_position < 0.2:
            current_signals.append(("Bollinger Bands", "🟡 Price near Lower BB"))
        else:
            current_signals.append(("Bollinger Bands", "⚪ Price within BB (Neutral)"))
    
    # Overall Signal
    bullish_count = sum(1 for _, signal in current_signals if "🟢" in signal)
    bearish_count = sum(1 for _, signal in current_signals if "🔴" in signal)
    
    if bullish_count >= 6:
        overall_signal = "🟢 Strong Buy"
    elif bullish_count >= 4:
        overall_signal = "🟢 Buy"
    elif bearish_count >= 6:
        overall_signal = "🔴 Strong Sell"
    elif bearish_count >= 4:
        overall_signal = "🔴 Sell"
    else:
        overall_signal = "🟡 Neutral/Hold"
    
    # Display current signals in a table
    signal_df = pd.DataFrame(current_signals, columns=['Indicator', 'Signal'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(signal_df, hide_index=True)
    
    with col2:
        st.markdown("### Overall Signal")
        st.markdown(f"## {overall_signal}")
        st.markdown(f"Bullish Signals: {bullish_count}")
        st.markdown(f"Bearish Signals: {bearish_count}")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** The trading signals shown here are based on technical analysis only and should not be used as the sole basis for investment decisions.
    Always conduct thorough research and consider fundamental factors, market conditions, and your personal investment goals before making any trading decisions.
    Past performance is not indicative of future results.
    """)
