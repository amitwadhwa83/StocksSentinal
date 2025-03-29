import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
import streamlit as st

def fetch_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch stock data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        pandas.DataFrame: Historical stock data or None if error occurs
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return None
        
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_stock_info(ticker):
    """
    Get detailed information about a stock.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information or None if error occurs
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching information for {ticker}: {str(e)}")
        return None

def get_sector_performance():
    """
    Get performance data for different market sectors.
    
    Returns:
        pandas.DataFrame: Sector performance data or None if error occurs
    """
    sector_etfs = {
        'Technology': 'XLK',
        'Financial': 'XLF',
        'Health Care': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Industrial': 'XLI',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    
    try:
        sector_data = {}
        
        for sector, etf in sector_etfs.items():
            data = fetch_stock_data(etf, period="1mo")
            if data is not None and not data.empty:
                # Calculate monthly performance
                start_price = data['Close'].iloc[0]
                current_price = data['Close'].iloc[-1]
                performance = ((current_price - start_price) / start_price) * 100
                sector_data[sector] = performance
        
        return pd.DataFrame(list(sector_data.items()), columns=['Sector', 'Monthly Performance %'])
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        return None

def parse_portfolio_csv(uploaded_file):
    """
    Parse an uploaded portfolio CSV file.
    
    Expected CSV format:
    Symbol,Shares,Purchase Price
    AAPL,10,150.00
    MSFT,5,280.00
    
    Args:
        uploaded_file: StreamlitUploadedFile object
    
    Returns:
        pandas.DataFrame: Parsed portfolio data or None if error occurs
    """
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        required_columns = ['Symbol', 'Shares', 'Purchase Price']
        
        # Check if all required columns are present
        if not all(col in portfolio_df.columns for col in required_columns):
            st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
            return None
        
        # Validate data types and values
        if not pd.to_numeric(portfolio_df['Shares'], errors='coerce').notna().all():
            st.error("All values in 'Shares' column must be numeric")
            return None
            
        if not pd.to_numeric(portfolio_df['Purchase Price'], errors='coerce').notna().all():
            st.error("All values in 'Purchase Price' column must be numeric")
            return None
        
        # Convert to appropriate types
        portfolio_df['Shares'] = pd.to_numeric(portfolio_df['Shares'])
        portfolio_df['Purchase Price'] = pd.to_numeric(portfolio_df['Purchase Price'])
        
        return portfolio_df
    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return None
