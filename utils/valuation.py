import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

def get_valuation_metrics(ticker):
    """
    Get key valuation metrics for a stock.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary of valuation metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic valuation metrics
        metrics = {}
        
        # Price multiples
        metrics['Market Cap'] = info.get('marketCap', None)
        metrics['P/E Ratio'] = info.get('trailingPE', None)
        metrics['Forward P/E'] = info.get('forwardPE', None)
        metrics['PEG Ratio'] = info.get('pegRatio', None)
        metrics['Price/Sales'] = info.get('priceToSalesTrailing12Months', None)
        metrics['Price/Book'] = info.get('priceToBook', None)
        metrics['Enterprise Value'] = info.get('enterpriseValue', None)
        metrics['EV/Revenue'] = info.get('enterpriseToRevenue', None)
        metrics['EV/EBITDA'] = info.get('enterpriseToEbitda', None)
        
        # Dividend metrics
        metrics['Dividend Yield'] = info.get('dividendYield', None)
        if metrics['Dividend Yield'] is not None:
            metrics['Dividend Yield'] *= 100  # Convert to percentage
        metrics['Dividend Rate'] = info.get('dividendRate', None)
        metrics['Payout Ratio'] = info.get('payoutRatio', None)
        if metrics['Payout Ratio'] is not None:
            metrics['Payout Ratio'] *= 100  # Convert to percentage
        
        # Profitability metrics
        metrics['Profit Margin'] = info.get('profitMargins', None)
        if metrics['Profit Margin'] is not None:
            metrics['Profit Margin'] *= 100  # Convert to percentage
        metrics['Operating Margin'] = info.get('operatingMargins', None)
        if metrics['Operating Margin'] is not None:
            metrics['Operating Margin'] *= 100  # Convert to percentage
        metrics['ROE'] = info.get('returnOnEquity', None)
        if metrics['ROE'] is not None:
            metrics['ROE'] *= 100  # Convert to percentage
        metrics['ROA'] = info.get('returnOnAssets', None)
        if metrics['ROA'] is not None:
            metrics['ROA'] *= 100  # Convert to percentage
        
        # Growth metrics
        metrics['Revenue Growth'] = info.get('revenueGrowth', None)
        if metrics['Revenue Growth'] is not None:
            metrics['Revenue Growth'] *= 100  # Convert to percentage
        metrics['Earnings Growth'] = info.get('earningsGrowth', None)
        if metrics['Earnings Growth'] is not None:
            metrics['Earnings Growth'] *= 100  # Convert to percentage
        
        # Clean up None values for display
        for key in list(metrics.keys()):
            if metrics[key] is None:
                metrics[key] = 'N/A'
        
        return metrics
    
    except Exception as e:
        st.error(f"Error fetching valuation metrics for {ticker}: {str(e)}")
        return None

def get_peer_comparison(ticker):
    """
    Compare valuation metrics with industry peers.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        pandas.DataFrame: Peer comparison data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get industry/sector for peer comparison
        industry = info.get('industry', None)
        sector = info.get('sector', None)
        
        if not industry and not sector:
            return None
        
        # Get peer companies
        # In a real implementation, you would use a more comprehensive database
        # For this demo, we'll just use a small set of known companies in major sectors
        tech_peers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
        financial_peers = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        healthcare_peers = ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT']
        consumer_peers = ['PG', 'KO', 'PEP', 'WMT', 'COST']
        energy_peers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        
        # Select appropriate peer set based on sector
        if sector == 'Technology':
            peers = tech_peers
        elif sector == 'Financial Services':
            peers = financial_peers
        elif sector == 'Healthcare':
            peers = healthcare_peers
        elif sector == 'Consumer Defensive' or sector == 'Consumer Cyclical':
            peers = consumer_peers
        elif sector == 'Energy':
            peers = energy_peers
        else:
            # Default to companies from mixed sectors
            peers = ['AAPL', 'JPM', 'JNJ', 'PG', 'XOM']
        
        # Make sure we're not duplicating the ticker
        if ticker in peers:
            peers.remove(ticker)
        
        # Add the target company at the beginning
        peers = [ticker] + peers[:4]  # Limit to 5 companies total for readability
        
        # Metrics to compare
        comparison_metrics = {
            'P/E Ratio': 'trailingPE',
            'Forward P/E': 'forwardPE',
            'P/S Ratio': 'priceToSalesTrailing12Months',
            'P/B Ratio': 'priceToBook',
            'EV/EBITDA': 'enterpriseToEbitda',
            'Profit Margin (%)': 'profitMargins',
            'ROE (%)': 'returnOnEquity'
        }
        
        # Collect data for each peer
        comparison_data = []
        
        for peer in peers:
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                peer_row = {'Symbol': peer}
                
                # Get company name
                peer_row['Company'] = peer_info.get('shortName', peer)
                
                # Get metrics
                for display_name, info_key in comparison_metrics.items():
                    value = peer_info.get(info_key, None)
                    
                    # Convert ratios to percentages where applicable
                    if display_name in ['Profit Margin (%)', 'ROE (%)'] and value is not None:
                        value = value * 100
                    
                    peer_row[display_name] = value
                
                comparison_data.append(peer_row)
            except Exception as e:
                st.warning(f"Error fetching data for peer {peer}: {str(e)}")
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format values for display
        for column in comparison_df.columns:
            if column not in ['Symbol', 'Company']:
                comparison_df[column] = comparison_df[column].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"
                )
        
        return comparison_df
    
    except Exception as e:
        st.error(f"Error creating peer comparison for {ticker}: {str(e)}")
        return None

def estimate_fair_value_pe(ticker):
    """
    Estimate fair value based on P/E ratio and historical averages.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary with fair value estimate and related metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current_price = info.get('regularMarketPrice', None)
        if current_price is None:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return None
        
        # Get earnings per share
        eps = info.get('trailingEPS', None)
        if eps is None or eps <= 0:
            return None
        
        # Get current P/E
        current_pe = info.get('trailingPE', None)
        if current_pe is None:
            current_pe = current_price / eps
        
        # Get industry average P/E (normally would come from a database)
        # Using fixed values for demo purposes
        industry_pe_map = {
            'Technology': 25,
            'Financial Services': 15,
            'Healthcare': 22,
            'Consumer Defensive': 20,
            'Consumer Cyclical': 18,
            'Energy': 12,
            'Utilities': 17,
            'Communication Services': 20,
            'Basic Materials': 15,
            'Industrials': 18,
            'Real Estate': 16
        }
        
        sector = info.get('sector', 'Technology')
        industry_pe = industry_pe_map.get(sector, 20)  # Default to 20 if sector not found
        
        # Get 5-year average P/E (would normally calculate from historical data)
        # Using a rough approximation for demo
        hist_pe = info.get('fiveYearAvgDividendYield', None)
        if hist_pe is None or hist_pe <= 0:
            hist_pe = industry_pe * 0.9  # Approximate for demo
        
        # Calculate fair values based on different P/E multiples
        fair_value_industry_pe = eps * industry_pe
        fair_value_hist_pe = eps * hist_pe
        
        # Discount and premium scenarios
        fair_value_discount = eps * max(current_pe * 0.8, industry_pe * 0.8)
        fair_value_premium = eps * min(current_pe * 1.2, industry_pe * 1.2)
        
        # Average fair value
        avg_fair_value = (fair_value_industry_pe + fair_value_hist_pe) / 2
        
        # Upside/downside potential
        upside_pct = ((avg_fair_value / current_price) - 1) * 100
        
        # Results
        results = {
            'Current Price': current_price,
            'EPS (TTM)': eps,
            'Current P/E': current_pe,
            'Industry Avg P/E': industry_pe,
            'Historical Avg P/E': hist_pe,
            'Fair Value (Industry P/E)': fair_value_industry_pe,
            'Fair Value (Historical P/E)': fair_value_hist_pe,
            'Fair Value (Discounted)': fair_value_discount,
            'Fair Value (Premium)': fair_value_premium,
            'Average Fair Value': avg_fair_value,
            'Potential Upside/Downside': upside_pct
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error estimating fair value for {ticker}: {str(e)}")
        return None
