import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

def analyze_portfolio(portfolio_df):
    """
    Analyze a portfolio of stocks.
    
    Args:
        portfolio_df (pandas.DataFrame): Portfolio with 'Symbol', 'Shares', and 'Purchase Price' columns
    
    Returns:
        tuple: (Enhanced portfolio DataFrame, performance metrics dict)
    """
    if portfolio_df is None or portfolio_df.empty:
        return None, None
    
    try:
        # Create a copy to avoid modifying the original
        result_df = portfolio_df.copy()
        
        # Initialize new columns
        result_df['Current Price'] = 0.0
        result_df['Market Value'] = 0.0
        result_df['Cost Basis'] = 0.0
        result_df['Gain/Loss $'] = 0.0
        result_df['Gain/Loss %'] = 0.0
        
        total_market_value = 0
        total_cost_basis = 0
        
        # Process each stock in the portfolio
        for idx, row in result_df.iterrows():
            try:
                # Fetch current price
                ticker = yf.Ticker(row['Symbol'])
                info = ticker.info
                
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    current_price = info['regularMarketPrice']
                else:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        st.warning(f"Could not get current price for {row['Symbol']}. Using purchase price.")
                        current_price = row['Purchase Price']
                
                # Calculate values
                shares = row['Shares']
                purchase_price = row['Purchase Price']
                market_value = shares * current_price
                cost_basis = shares * purchase_price
                gain_loss_dollar = market_value - cost_basis
                gain_loss_percent = (gain_loss_dollar / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Update the dataframe
                result_df.loc[idx, 'Current Price'] = current_price
                result_df.loc[idx, 'Market Value'] = market_value
                result_df.loc[idx, 'Cost Basis'] = cost_basis
                result_df.loc[idx, 'Gain/Loss $'] = gain_loss_dollar
                result_df.loc[idx, 'Gain/Loss %'] = gain_loss_percent
                
                # Add to totals
                total_market_value += market_value
                total_cost_basis += cost_basis
                
            except Exception as e:
                st.warning(f"Error processing {row['Symbol']}: {str(e)}")
        
        # Calculate portfolio metrics
        total_gain_loss = total_market_value - total_cost_basis
        total_gain_loss_percent = (total_gain_loss / total_cost_basis) * 100 if total_cost_basis > 0 else 0
        
        # Calculate weights
        result_df['Weight %'] = (result_df['Market Value'] / total_market_value) * 100 if total_market_value > 0 else 0
        
        # Format the dataframe
        result_df['Current Price'] = result_df['Current Price'].round(2)
        result_df['Market Value'] = result_df['Market Value'].round(2)
        result_df['Cost Basis'] = result_df['Cost Basis'].round(2)
        result_df['Gain/Loss $'] = result_df['Gain/Loss $'].round(2)
        result_df['Gain/Loss %'] = result_df['Gain/Loss %'].round(2)
        result_df['Weight %'] = result_df['Weight %'].round(2)
        
        # Portfolio metrics dictionary
        metrics = {
            'Total Market Value': total_market_value,
            'Total Cost Basis': total_cost_basis,
            'Total Gain/Loss $': total_gain_loss,
            'Total Gain/Loss %': total_gain_loss_percent,
            'Number of Holdings': len(result_df),
        }
        
        return result_df, metrics
    
    except Exception as e:
        st.error(f"Error analyzing portfolio: {str(e)}")
        return None, None

def calculate_portfolio_risk(portfolio_df, lookback_period='1y'):
    """
    Calculate portfolio risk metrics.
    
    Args:
        portfolio_df (pandas.DataFrame): Portfolio with 'Symbol' and 'Weight %' columns
        lookback_period (str): Period for historical data
    
    Returns:
        dict: Risk metrics
    """
    if portfolio_df is None or portfolio_df.empty:
        return None
    
    try:
        symbols = portfolio_df['Symbol'].tolist()
        weights = portfolio_df['Weight %'].tolist() if 'Weight %' in portfolio_df.columns else None
        
        if weights is None:
            # Equal weights if not provided
            weights = [1.0 / len(symbols)] * len(symbols)
        else:
            # Convert from percentage to decimal
            weights = [w / 100 for w in weights]
        
        # Fetch historical data
        stock_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, period=lookback_period, progress=False)
                if not data.empty:
                    stock_data[symbol] = data['Adj Close']
            except Exception as e:
                st.warning(f"Error fetching historical data for {symbol}: {str(e)}")
        
        if not stock_data:
            return None
        
        # Create a dataframe of close prices
        close_df = pd.DataFrame(stock_data)
        
        # Fill NaN values with forward fill followed by backward fill
        close_df = close_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate daily returns
        returns_df = close_df.pct_change().dropna()
        
        # Calculate portfolio statistics
        portfolio_return = np.sum(returns_df.mean() * 252 * weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns_df.cov() * 252, weights)))
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02 or 2%)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns_df).cumprod()
        weighted_cumulative_returns = pd.DataFrame({
            'Portfolio': np.sum(cumulative_returns.mul(weights, axis=1), axis=1)
        })
        rolling_max = weighted_cumulative_returns['Portfolio'].cummax()
        drawdown = (weighted_cumulative_returns['Portfolio'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Risk metrics dictionary
        risk_metrics = {
            'Expected Annual Return': portfolio_return * 100,  # Convert to percentage
            'Annual Volatility': portfolio_volatility * 100,   # Convert to percentage
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown * 100,           # Convert to percentage
        }
        
        return risk_metrics
    
    except Exception as e:
        st.error(f"Error calculating portfolio risk: {str(e)}")
        return None

def create_sample_portfolio():
    """
    Create a sample portfolio for demonstration.
    
    Returns:
        pandas.DataFrame: Sample portfolio
    """
    data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Shares': [10, 5, 2, 3, 8],
        'Purchase Price': [150.00, 280.00, 2800.00, 3500.00, 700.00]
    }
    return pd.DataFrame(data)

def get_sector_allocation(portfolio_df):
    """
    Calculate sector allocation for the portfolio.
    
    Args:
        portfolio_df (pandas.DataFrame): Portfolio with 'Symbol' and 'Market Value' columns
    
    Returns:
        pandas.DataFrame: Sector allocation data
    """
    if portfolio_df is None or portfolio_df.empty:
        return None
    
    if 'Symbol' not in portfolio_df.columns or 'Market Value' not in portfolio_df.columns:
        return None
    
    try:
        sectors = {}
        total_value = portfolio_df['Market Value'].sum()
        
        for _, row in portfolio_df.iterrows():
            try:
                ticker = yf.Ticker(row['Symbol'])
                info = ticker.info
                sector = info.get('sector', 'Unknown')
                
                if sector in sectors:
                    sectors[sector] += row['Market Value']
                else:
                    sectors[sector] = row['Market Value']
                    
            except Exception as e:
                st.warning(f"Error getting sector for {row['Symbol']}: {str(e)}")
        
        # Convert to dataframe
        sector_df = pd.DataFrame({
            'Sector': list(sectors.keys()),
            'Value': list(sectors.values()),
            'Allocation %': [val / total_value * 100 for val in sectors.values()]
        })
        
        # Round values
        sector_df['Value'] = sector_df['Value'].round(2)
        sector_df['Allocation %'] = sector_df['Allocation %'].round(2)
        
        # Sort by allocation
        sector_df = sector_df.sort_values('Allocation %', ascending=False).reset_index(drop=True)
        
        return sector_df
        
    except Exception as e:
        st.error(f"Error calculating sector allocation: {str(e)}")
        return None
