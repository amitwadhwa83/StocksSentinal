import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

from utils.data_fetcher import fetch_stock_data, get_stock_info
from utils.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands
from utils.trend_analysis import analyze_trend, predict_short_term_trend, get_support_resistance_levels

def display_stock_analysis():
    st.header("📈 Stock Analysis")
    
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
                index=3
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
            # Create tabs for different analyses
            tabs = st.tabs(["Overview", "Price Chart", "Technical Indicators", "Trend Analysis"])
            
            with tabs[0]:
                display_stock_overview(ticker, data)
            
            with tabs[1]:
                display_price_chart(data, ticker)
            
            with tabs[2]:
                display_technical_indicators(data)
            
            with tabs[3]:
                display_trend_analysis(data)
        else:
            st.error(f"Error fetching data for ticker: {ticker}. Please check the symbol and try again.")

def display_stock_overview(ticker, data):
    st.subheader("Stock Overview")
    
    # Fetch company info
    info = get_stock_info(ticker)
    
    if info is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display basic company info
            if 'shortName' in info:
                st.markdown(f"### {info.get('shortName', ticker)}")
            if 'sector' in info and 'industry' in info:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            
            # Display current price and change
            current_price = data['Close'].iloc[-1]
            prev_close = info.get('previousClose', data['Close'].iloc[-2] if len(data) > 1 else current_price)
            price_change = current_price - prev_close
            pct_change = (price_change / prev_close) * 100
            
            delta_color = "normal" if price_change == 0 else "up" if price_change > 0 else "down"
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({pct_change:.2f}%)",
                delta_color=delta_color
            )
            
            # Display trading range
            day_low = info.get('dayLow', data['Low'].iloc[-1])
            day_high = info.get('dayHigh', data['High'].iloc[-1])
            st.write(f"**Day Range:** ${day_low:.2f} - ${day_high:.2f}")
            
            # Display 52-week range
            yearly_low = info.get('fiftyTwoWeekLow', 'N/A')
            yearly_high = info.get('fiftyTwoWeekHigh', 'N/A')
            if yearly_low != 'N/A' and yearly_high != 'N/A':
                st.write(f"**52-Week Range:** ${yearly_low:.2f} - ${yearly_high:.2f}")
        
        with col2:
            # Display key statistics
            st.subheader("Key Statistics")
            
            # Create two columns for statistics
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                market_cap = info.get('marketCap', None)
                if market_cap is not None:
                    market_cap_str = f"${market_cap / 1e9:.2f} B" if market_cap >= 1e9 else f"${market_cap / 1e6:.2f} M"
                    st.write(f"**Market Cap:** {market_cap_str}")
                
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**EPS (TTM):** ${info.get('trailingEPS', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%")
                st.write(f"**Beta:** {info.get('beta', 'N/A')}")
                
            with stat_col2:
                st.write(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
                st.write(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
                st.write(f"**Price/Book:** {info.get('priceToBook', 'N/A')}")
                st.write(f"**Price/Sales:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
                st.write(f"**Volume:** {int(data['Volume'].iloc[-1]):,}")
        
        # Company description
        if 'longBusinessSummary' in info:
            with st.expander("Company Description"):
                st.write(info.get('longBusinessSummary', 'No description available.'))
    else:
        st.warning(f"Could not fetch company information for {ticker}")

def display_price_chart(data, ticker):
    st.subheader("Price Chart")
    
    # Technical overlay selection
    overlay_options = st.multiselect(
        "Select Technical Overlays",
        ["SMA 20", "SMA 50", "SMA 200", "EMA 20", "Bollinger Bands"],
        default=["SMA 50"]
    )
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            increasing=dict(line=dict(color='#26A69A')),
            decreasing=dict(line=dict(color='#EF5350')),
        ),
        row=1, col=1
    )
    
    # Add technical overlays
    if "SMA 20" in overlay_options:
        sma20 = calculate_sma(data, 20)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma20,
                name="SMA 20",
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1)
            ),
            row=1, col=1
        )
    
    if "SMA 50" in overlay_options:
        sma50 = calculate_sma(data, 50)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma50,
                name="SMA 50",
                line=dict(color='rgba(46, 139, 87, 0.7)', width=1)
            ),
            row=1, col=1
        )
    
    if "SMA 200" in overlay_options:
        sma200 = calculate_sma(data, 200)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma200,
                name="SMA 200",
                line=dict(color='rgba(70, 130, 180, 0.7)', width=1)
            ),
            row=1, col=1
        )
    
    if "EMA 20" in overlay_options:
        ema20 = calculate_ema(data, 20)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ema20,
                name="EMA 20",
                line=dict(color='rgba(148, 0, 211, 0.7)', width=1)
            ),
            row=1, col=1
        )
    
    if "Bollinger Bands" in overlay_options:
        middle, upper, lower = calculate_bollinger_bands(data, window=20, num_std=2)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=upper,
                name="Upper BB",
                line=dict(color='rgba(169, 169, 169, 0.5)', width=1)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=middle,
                name="Middle BB",
                line=dict(color='rgba(169, 169, 169, 0.7)', width=1)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=lower,
                name="Lower BB",
                line=dict(color='rgba(169, 169, 169, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(169, 169, 169, 0.1)'
            ),
            row=1, col=1
        )
    
    # Add volume bars
    colors = ['#26A69A' if data['Close'].iloc[i] > data['Open'].iloc[i] else '#EF5350' for i in range(len(data))]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker=dict(color=colors, opacity=0.7)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_indicators(data):
    st.subheader("Technical Indicators")
    
    # Calculate indicators
    data['SMA20'] = calculate_sma(data, 20)
    data['SMA50'] = calculate_sma(data, 50)
    data['SMA200'] = calculate_sma(data, 200)
    data['EMA20'] = calculate_ema(data, 20)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    
    # Create tabs for different indicators
    indicator_tabs = st.tabs(["Moving Averages", "Oscillators", "MACD"])
    
    with indicator_tabs[0]:
        st.markdown("### Moving Averages")
        
        # Moving average crossovers
        st.markdown("#### Moving Average Crossovers")
        
        last_values = data.iloc[-1]
        prev_values = data.iloc[-2] if len(data) > 1 else last_values
        
        # SMA20 vs SMA50 crossover
        sma20_cross_sma50 = None
        if last_values['SMA20'] > last_values['SMA50'] and prev_values['SMA20'] <= prev_values['SMA50']:
            sma20_cross_sma50 = "Golden Cross (Bullish): SMA20 crossed above SMA50"
        elif last_values['SMA20'] < last_values['SMA50'] and prev_values['SMA20'] >= prev_values['SMA50']:
            sma20_cross_sma50 = "Death Cross (Bearish): SMA20 crossed below SMA50"
        
        # SMA50 vs SMA200 crossover
        sma50_cross_sma200 = None
        if last_values['SMA50'] > last_values['SMA200'] and prev_values['SMA50'] <= prev_values['SMA200']:
            sma50_cross_sma200 = "Golden Cross (Bullish): SMA50 crossed above SMA200"
        elif last_values['SMA50'] < last_values['SMA200'] and prev_values['SMA50'] >= prev_values['SMA200']:
            sma50_cross_sma200 = "Death Cross (Bearish): SMA50 crossed below SMA200"
        
        # Display crossover information
        if sma20_cross_sma50:
            st.markdown(f"**SMA20/SMA50:** {sma20_cross_sma50}")
        else:
            if last_values['SMA20'] > last_values['SMA50']:
                st.markdown("**SMA20/SMA50:** SMA20 is above SMA50 (Bullish)")
            else:
                st.markdown("**SMA20/SMA50:** SMA20 is below SMA50 (Bearish)")
        
        if sma50_cross_sma200:
            st.markdown(f"**SMA50/SMA200:** {sma50_cross_sma200}")
        else:
            if last_values['SMA50'] > last_values['SMA200']:
                st.markdown("**SMA50/SMA200:** SMA50 is above SMA200 (Bullish)")
            else:
                st.markdown("**SMA50/SMA200:** SMA50 is below SMA200 (Bearish)")
        
        # Price vs MA relationship
        st.markdown("#### Price vs. Moving Averages")
        
        price = last_values['Close']
        
        if price > last_values['SMA20']:
            st.markdown("**Price vs SMA20:** Price is above SMA20 (Bullish)")
        else:
            st.markdown("**Price vs SMA20:** Price is below SMA20 (Bearish)")
        
        if price > last_values['SMA50']:
            st.markdown("**Price vs SMA50:** Price is above SMA50 (Bullish)")
        else:
            st.markdown("**Price vs SMA50:** Price is below SMA50 (Bearish)")
        
        if price > last_values['SMA200']:
            st.markdown("**Price vs SMA200:** Price is above SMA200 (Bullish)")
        else:
            st.markdown("**Price vs SMA200:** Price is below SMA200 (Bearish)")
        
        # MA values
        st.markdown("#### Current Moving Average Values")
        ma_col1, ma_col2 = st.columns(2)
        
        with ma_col1:
            st.metric("SMA20", f"${last_values['SMA20']:.2f}")
            st.metric("SMA50", f"${last_values['SMA50']:.2f}")
            st.metric("SMA200", f"${last_values['SMA200']:.2f}")
        
        with ma_col2:
            st.metric("EMA20", f"${last_values['EMA20']:.2f}")
            st.metric("Current Price", f"${price:.2f}")
    
    with indicator_tabs[1]:
        st.markdown("### Oscillators")
        
        # RSI Chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                name="RSI",
                line=dict(color='purple', width=1)
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral")
        
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI interpretation
        st.markdown("#### RSI Interpretation")
        rsi_value = data['RSI'].iloc[-1]
        
        if rsi_value > 70:
            st.markdown(f"**Current RSI: {rsi_value:.2f}** - Overbought (potential selling opportunity)")
        elif rsi_value < 30:
            st.markdown(f"**Current RSI: {rsi_value:.2f}** - Oversold (potential buying opportunity)")
        else:
            st.markdown(f"**Current RSI: {rsi_value:.2f}** - Neutral")
        
        # RSI trends
        rsi_5d_ago = data['RSI'].iloc[-5] if len(data) >= 5 else data['RSI'].iloc[0]
        rsi_trend = rsi_value - rsi_5d_ago
        
        if rsi_trend > 0:
            st.markdown(f"**RSI Trend:** Increasing (+{rsi_trend:.2f} over 5 periods)")
        else:
            st.markdown(f"**RSI Trend:** Decreasing ({rsi_trend:.2f} over 5 periods)")
    
    with indicator_tabs[2]:
        st.markdown("### MACD (Moving Average Convergence Divergence)")
        
        # MACD Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line on secondary y-axis for reference
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name="Price",
                line=dict(color='black', width=1, dash='dot'),
                opacity=0.3
            ),
            secondary_y=True
        )
        
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                name="MACD",
                line=dict(color='#1E88E5', width=1)
            ),
            secondary_y=False
        )
        
        # Add Signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Signal'],
                name="Signal",
                line=dict(color='#FF8F00', width=1)
            ),
            secondary_y=False
        )
        
        # Add Histogram
        colors = ['#26A69A' if val >= 0 else '#EF5350' for val in data['Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Histogram'],
                name="Histogram",
                marker=dict(color=colors),
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title="MACD Indicator with Price",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500
        )
        
        fig.update_yaxes(title_text="MACD", secondary_y=False)
        fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MACD interpretation
        st.markdown("#### MACD Interpretation")
        
        last_macd = data['MACD'].iloc[-1]
        last_signal = data['Signal'].iloc[-1]
        last_hist = data['Histogram'].iloc[-1]
        
        prev_macd = data['MACD'].iloc[-2] if len(data) > 1 else last_macd
        prev_signal = data['Signal'].iloc[-2] if len(data) > 1 else last_signal
        prev_hist = data['Histogram'].iloc[-2] if len(data) > 1 else last_hist
        
        # Check for crossovers
        if last_macd > last_signal and prev_macd <= prev_signal:
            st.markdown("**MACD Crossover:** MACD crossed above Signal line (Bullish)")
        elif last_macd < last_signal and prev_macd >= prev_signal:
            st.markdown("**MACD Crossover:** MACD crossed below Signal line (Bearish)")
        else:
            if last_macd > last_signal:
                st.markdown("**MACD Position:** MACD is above Signal line (Bullish)")
            else:
                st.markdown("**MACD Position:** MACD is below Signal line (Bearish)")
        
        # Check histogram
        if last_hist > 0 and prev_hist <= 0:
            st.markdown("**Histogram:** Changed from negative to positive (Bullish)")
        elif last_hist < 0 and prev_hist >= 0:
            st.markdown("**Histogram:** Changed from positive to negative (Bearish)")
        elif last_hist > 0:
            if last_hist > prev_hist:
                st.markdown("**Histogram:** Positive and increasing (Strong Bullish)")
            else:
                st.markdown("**Histogram:** Positive but decreasing (Weakening Bullish)")
        else:
            if last_hist < prev_hist:
                st.markdown("**Histogram:** Negative and decreasing (Strong Bearish)")
            else:
                st.markdown("**Histogram:** Negative but increasing (Weakening Bearish)")

def display_trend_analysis(data):
    st.subheader("Trend Analysis")
    
    # Get trend analysis
    trend_results = analyze_trend(data)
    
    if trend_results is not None:
        # Display trend summary
        st.markdown(f"### Overall Trend: {trend_results['Overall Trend']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Action")
            st.markdown(f"**Last Close:** ${trend_results['Last Close']:.2f}")
            st.markdown(f"**Price Change (20d):** {trend_results['Price Change (20 days)']}")
            
            st.markdown("#### Moving Average Signals")
            for signal in trend_results['MA Signals']:
                if "Bullish" in signal:
                    st.markdown(f"- ✅ {signal}")
                else:
                    st.markdown(f"- ❌ {signal}")
        
        with col2:
            st.markdown("#### Oscillator Signals")
            st.markdown(f"**RSI:** {trend_results['RSI']:.2f} - {trend_results['RSI Signal']}")
            st.markdown(f"**MACD:** {trend_results['MACD Signal']}")
            
            st.markdown("#### Volume Analysis")
            st.markdown(f"**Volume:** {trend_results['Volume']}")
            st.markdown(f"**Volume Signal:** {trend_results['Volume Signal']}")
        
        # Support and resistance levels
        support_levels, resistance_levels = get_support_resistance_levels(data)
        
        if support_levels is not None and resistance_levels is not None:
            st.markdown("### Support & Resistance Levels")
            
            res_col, sup_col = st.columns(2)
            
            with res_col:
                st.markdown("#### Resistance Levels")
                for level in reversed(resistance_levels):
                    st.markdown(f"- ${level:.2f}")
            
            with sup_col:
                st.markdown("#### Support Levels")
                for level in reversed(support_levels):
                    st.markdown(f"- ${level:.2f}")
        
        # Short-term prediction
        st.markdown("### Short-term Trend Prediction (Linear)")
        
        prediction_df = predict_short_term_trend(data, days=5)
        
        if prediction_df is not None:
            # Create prediction chart
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(
                go.Scatter(
                    x=data.index[-10:],
                    y=data['Close'].iloc[-10:],
                    name="Historical Price",
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add predicted prices
            fig.add_trace(
                go.Scatter(
                    x=prediction_df.index,
                    y=prediction_df['Predicted_Price'],
                    name="Predicted Price",
                    line=dict(color='orange', width=2, dash='dot')
                )
            )
            
            # Add current price marker
            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[data['Close'].iloc[-1]],
                    name="Current Price",
                    mode="markers",
                    marker=dict(color='red', size=10)
                )
            )
            
            fig.update_layout(
                title="5-Day Price Prediction (Linear Model)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction values
            st.markdown("#### Predicted Prices")
            
            pred_table = pd.DataFrame({
                'Date': prediction_df.index.strftime('%Y-%m-%d'),
                'Predicted Price': [f"${price:.2f}" for price in prediction_df['Predicted_Price']]
            })
            
            st.table(pred_table)
            
            # Prediction disclaimer
            st.info("""
                **Disclaimer:** This is a simple linear prediction based on recent price action and should not be used as the sole basis for investment decisions. 
                The stock market is influenced by numerous factors that this model does not account for.
            """)
        else:
            st.warning("Insufficient data for short-term prediction.")
    else:
        st.error("Could not analyze trend. Insufficient data.")
