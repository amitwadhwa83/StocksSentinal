import streamlit as st
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app structure
def main():
    st.title("📊 Stock Analysis Tool")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Dashboard", "Stock Analysis", "Portfolio Management", "Technical Indicators", "Valuation Metrics"]
    )
    
    # Dashboard page
    if page == "Dashboard":
        display_dashboard()
    elif page == "Stock Analysis":
        from pages.stock_analysis import display_stock_analysis
        display_stock_analysis()
    elif page == "Portfolio Management":
        from pages.portfolio_management import display_portfolio_management
        display_portfolio_management()
    elif page == "Technical Indicators":
        from pages.technical_indicators import display_technical_indicators
        display_technical_indicators()
    elif page == "Valuation Metrics":
        from pages.valuation_metrics import display_valuation_metrics
        display_valuation_metrics()

def display_dashboard():
    st.header("📈 Market Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Major Indices")
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        index_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
        
        # Create a placeholder for the indices data
        indices_placeholder = st.empty()
        
        try:
            # Fetch data for major indices
            indices_data = {}
            for idx, symbol in enumerate(indices):
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_close = ticker.info.get('previousClose', current)
                    change = ((current - prev_close) / prev_close) * 100
                    indices_data[index_names[idx]] = {
                        'price': current,
                        'change': change
                    }
            
            # Display indices data
            if indices_data:
                for name, data in indices_data.items():
                    color = "green" if data['change'] >= 0 else "red"
                    indices_placeholder.markdown(
                        f"**{name}**: {data['price']:.2f} "
                        f"<span style='color:{color}'>({data['change']:.2f}%)</span>", 
                        unsafe_allow_html=True
                    )
            else:
                indices_placeholder.error("Failed to load indices data.")
        except Exception as e:
            indices_placeholder.error(f"Error fetching indices data: {str(e)}")
    
    with col2:
        st.subheader("Quick Stock Lookup")
        ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)")
        
        if ticker_symbol:
            try:
                ticker_data = yf.Ticker(ticker_symbol)
                info = ticker_data.info
                
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    price = info['regularMarketPrice']
                    prev_close = info.get('previousClose', price)
                    change = ((price - prev_close) / prev_close) * 100
                    
                    col_price, col_change = st.columns(2)
                    with col_price:
                        st.metric(label="Current Price", value=f"${price:.2f}")
                    with col_change:
                        st.metric(label="Change", value=f"{change:.2f}%", delta=f"{change:.2f}%")
                    
                    st.subheader("Company Info")
                    if 'longName' in info:
                        st.write(f"**Name:** {info.get('longName', 'N/A')}")
                    if 'sector' in info:
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    if 'industry' in info:
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                else:
                    st.error(f"Could not find data for ticker: {ticker_symbol}")
            except Exception as e:
                st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
    
    # Recent market news
    st.subheader("📰 Recent Market News")
    st.info("This section will display recent market news when integrated with a news API.")
    
    # Getting started guide
    st.header("🚀 Getting Started")
    st.write("""
    Welcome to the Stock Analysis Tool! This application helps you analyze stocks, manage your portfolio, and make informed investment decisions.
    
    ### Key Features:
    - **Stock Analysis**: Analyze individual stocks with real-time data
    - **Portfolio Management**: Upload your portfolio and get insights
    - **Technical Indicators**: View SMA, EMA, RSI, MACD and other indicators
    - **Valuation Metrics**: Evaluate stocks using various valuation models
    
    Choose a page from the sidebar to get started!
    """)

if __name__ == "__main__":
    main()
