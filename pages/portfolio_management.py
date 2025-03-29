import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

from utils.data_fetcher import parse_portfolio_csv
from utils.portfolio_analysis import analyze_portfolio, calculate_portfolio_risk, create_sample_portfolio, get_sector_allocation

def display_portfolio_management():
    st.header("📊 Portfolio Management")
    
    # Portfolio input options
    input_method = st.radio(
        "Choose input method",
        ["Upload CSV", "Use Sample Portfolio"]
    )
    
    if input_method == "Upload CSV":
        # CSV upload
        st.markdown("""
        ### Upload Portfolio CSV
        
        Upload a CSV file with your portfolio data. The CSV should have the following columns:
        - **Symbol**: Stock ticker symbol (e.g., AAPL, MSFT)
        - **Shares**: Number of shares owned
        - **Purchase Price**: Price per share when purchased
        
        Example:
        
        | Symbol | Shares | Purchase Price |
        |--------|--------|---------------|
        | AAPL   | 10     | 150.00        |
        | MSFT   | 5      | 280.00        |
        """)
        
        uploaded_file = st.file_uploader("Upload your portfolio CSV", type="csv")
        
        if uploaded_file is not None:
            portfolio_df = parse_portfolio_csv(uploaded_file)
            
            if portfolio_df is not None:
                process_portfolio(portfolio_df)
            else:
                st.error("Failed to parse the uploaded CSV file. Please ensure it has the correct format.")
    else:
        # Sample portfolio
        st.info("Using a sample portfolio for demonstration purposes.")
        portfolio_df = create_sample_portfolio()
        process_portfolio(portfolio_df)

def process_portfolio(portfolio_df):
    # Analyze portfolio
    analyzed_df, metrics = analyze_portfolio(portfolio_df)
    
    if analyzed_df is not None and metrics is not None:
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Market Value",
                value=f"${metrics['Total Market Value']:,.2f}"
            )
        
        with col2:
            delta = metrics['Total Gain/Loss $']
            delta_pct = metrics['Total Gain/Loss %']
            delta_color = "normal" if delta == 0 else "up" if delta > 0 else "down"
            
            st.metric(
                label="Total Gain/Loss",
                value=f"${delta:,.2f}",
                delta=f"{delta_pct:.2f}%",
                delta_color=delta_color
            )
        
        with col3:
            st.metric(
                label="Holdings",
                value=metrics['Number of Holdings']
            )
        
        # Display portfolio holdings
        st.subheader("Portfolio Holdings")
        
        # Format the dataframe for display
        display_df = analyzed_df.copy()
        
        # Apply formatting
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
        display_df['Purchase Price'] = display_df['Purchase Price'].apply(lambda x: f"${x:.2f}")
        display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:,.2f}")
        display_df['Gain/Loss $'] = display_df['Gain/Loss $'].apply(lambda x: f"${x:,.2f}")
        display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(lambda x: f"{x:.2f}%")
        display_df['Weight %'] = display_df['Weight %'].apply(lambda x: f"{x:.2f}%")
        
        # Display the holdings table
        st.dataframe(display_df)
        
        # Portfolio visualizations
        st.subheader("Portfolio Visualizations")
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Allocation", "Performance", "Risk Analysis"])
        
        with viz_tabs[0]:
            display_allocation(analyzed_df)
        
        with viz_tabs[1]:
            display_performance(analyzed_df)
        
        with viz_tabs[2]:
            display_risk_analysis(analyzed_df)
        
        # Export portfolio
        st.subheader("Export Portfolio")
        
        # Create CSV for download
        csv = convert_portfolio_to_csv(display_df)
        
        st.download_button(
            label="Download Portfolio as CSV",
            data=csv,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def display_allocation(portfolio_df):
    st.markdown("### Portfolio Allocation")
    
    # Calculate weights for pie chart
    weights = portfolio_df[['Symbol', 'Market Value']].copy()
    weights['Weight'] = weights['Market Value'] / weights['Market Value'].sum() * 100
    
    # Create pie chart for holdings allocation
    fig = px.pie(
        weights,
        values='Weight',
        names='Symbol',
        title='Portfolio Allocation by Holdings',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector allocation
    sector_allocation = get_sector_allocation(portfolio_df)
    
    if sector_allocation is not None:
        st.markdown("### Sector Allocation")
        
        # Create pie chart for sector allocation
        fig = px.pie(
            sector_allocation,
            values='Allocation %',
            names='Sector',
            title='Portfolio Allocation by Sector',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sector allocation table
        sector_table = sector_allocation.copy()
        sector_table['Value'] = sector_table['Value'].apply(lambda x: f"${x:,.2f}")
        sector_table['Allocation %'] = sector_table['Allocation %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(sector_table)
    else:
        st.warning("Could not retrieve sector allocation data.")

def display_performance(portfolio_df):
    st.markdown("### Portfolio Performance")
    
    # Sort by gain/loss percentage
    performance_df = portfolio_df.sort_values(by='Gain/Loss %', ascending=False).copy()
    
    # Create a waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Portfolio",
        orientation="v",
        measure=["relative"] * len(performance_df),
        x=performance_df['Symbol'],
        y=performance_df['Gain/Loss $'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        text=performance_df['Gain/Loss %'].apply(lambda x: f"{x:.2f}%"),
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Gain/Loss by Holdings",
        xaxis_title="Holdings",
        yaxis_title="Gain/Loss ($)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.markdown("### Performance Breakdown")
    
    performance_table = portfolio_df[['Symbol', 'Shares', 'Purchase Price', 'Current Price', 'Gain/Loss $', 'Gain/Loss %']].copy()
    
    # Format the table
    performance_table['Purchase Price'] = performance_table['Purchase Price'].apply(lambda x: f"${x:.2f}")
    performance_table['Current Price'] = performance_table['Current Price'].apply(lambda x: f"${x:.2f}")
    
    # Create a conditional formatter for gain/loss
    def format_gain_loss(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'<span style="color:{color}">${val:.2f}</span>'
    
    def format_gain_loss_pct(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'<span style="color:{color}">{val:.2f}%</span>'
    
    performance_table['Gain/Loss $'] = performance_table['Gain/Loss $'].apply(format_gain_loss)
    performance_table['Gain/Loss %'] = performance_table['Gain/Loss %'].apply(format_gain_loss_pct)
    
    st.dataframe(performance_table)
    
    # Winners and losers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Performers")
        top_performers = portfolio_df.nlargest(3, 'Gain/Loss %')[['Symbol', 'Gain/Loss %']].copy()
        top_performers['Gain/Loss %'] = top_performers['Gain/Loss %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(top_performers)
    
    with col2:
        st.markdown("#### Bottom Performers")
        bottom_performers = portfolio_df.nsmallest(3, 'Gain/Loss %')[['Symbol', 'Gain/Loss %']].copy()
        bottom_performers['Gain/Loss %'] = bottom_performers['Gain/Loss %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(bottom_performers)

def display_risk_analysis(portfolio_df):
    st.markdown("### Risk Analysis")
    
    # Calculate risk metrics
    risk_metrics = calculate_portfolio_risk(portfolio_df)
    
    if risk_metrics is not None:
        # Display risk metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Expected Annual Return",
                value=f"{risk_metrics['Expected Annual Return']:.2f}%"
            )
            
            st.metric(
                label="Sharpe Ratio",
                value=f"{risk_metrics['Sharpe Ratio']:.2f}"
            )
        
        with col2:
            st.metric(
                label="Annual Volatility",
                value=f"{risk_metrics['Annual Volatility']:.2f}%"
            )
            
            st.metric(
                label="Maximum Drawdown",
                value=f"{risk_metrics['Maximum Drawdown']:.2f}%"
            )
        
        # Risk interpretation
        st.markdown("### Risk Interpretation")
        
        # Volatility interpretation
        volatility = risk_metrics['Annual Volatility']
        if volatility < 10:
            vol_message = "Your portfolio has **low volatility**, suggesting lower risk but potentially lower returns."
        elif volatility < 20:
            vol_message = "Your portfolio has **moderate volatility**, balancing risk and potential returns."
        else:
            vol_message = "Your portfolio has **high volatility**, suggesting higher risk but potentially higher returns."
        
        st.markdown(vol_message)
        
        # Sharpe ratio interpretation
        sharpe = risk_metrics['Sharpe Ratio']
        if sharpe < 0:
            sharpe_message = "Your portfolio has a **negative Sharpe ratio**, suggesting it's underperforming the risk-free rate."
        elif sharpe < 1:
            sharpe_message = "Your portfolio has a **low Sharpe ratio**, suggesting poor risk-adjusted returns."
        elif sharpe < 2:
            sharpe_message = "Your portfolio has a **decent Sharpe ratio**, suggesting reasonable risk-adjusted returns."
        else:
            sharpe_message = "Your portfolio has an **excellent Sharpe ratio**, suggesting strong risk-adjusted returns."
        
        st.markdown(sharpe_message)
        
        # Diversification analysis
        weights = portfolio_df[['Symbol', 'Weight %']].copy()
        weights['Weight'] = weights['Weight %']
        
        max_weight = weights['Weight'].max()
        if max_weight > 20:
            div_message = "Your portfolio appears to be **concentrated** in certain holdings, which may increase risk."
        else:
            div_message = "Your portfolio appears to be **well-diversified** across holdings, which may reduce risk."
        
        st.markdown(div_message)
        
    else:
        st.warning("Could not calculate risk metrics. Insufficient historical data.")

def convert_portfolio_to_csv(portfolio_df):
    """Convert portfolio dataframe to CSV for download"""
    # Create a copy to avoid modifying the original
    export_df = portfolio_df.copy()
    
    # Create a string buffer
    buffer = io.StringIO()
    
    # Write the dataframe to the buffer
    export_df.to_csv(buffer, index=False)
    
    # Get the CSV string
    csv_str = buffer.getvalue()
    
    return csv_str
