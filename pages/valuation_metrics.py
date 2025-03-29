import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from utils.data_fetcher import fetch_stock_data, get_stock_info
from utils.valuation import get_valuation_metrics, get_peer_comparison, estimate_fair_value_pe

def display_valuation_metrics():
    st.header("📊 Valuation Metrics")
    
    # Stock ticker input
    ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
    
    if ticker_input:
        ticker = ticker_input.upper().strip()
        
        # Fetch stock data
        data = fetch_stock_data(ticker, period="1y")
        
        if data is not None and not data.empty:
            # Create tabs for different valuation views
            tabs = st.tabs([
                "Key Metrics",
                "Peer Comparison",
                "Valuation Models",
                "Price Ratios"
            ])
            
            with tabs[0]:
                display_key_metrics(ticker)
            
            with tabs[1]:
                display_peer_comparison(ticker)
            
            with tabs[2]:
                display_valuation_models(ticker, data)
            
            with tabs[3]:
                display_price_ratios(ticker, data)
        else:
            st.error(f"Error fetching data for ticker: {ticker}. Please check the symbol and try again.")

def display_key_metrics(ticker):
    st.subheader("Key Valuation Metrics")
    
    # Get valuation metrics
    metrics = get_valuation_metrics(ticker)
    
    if metrics is not None:
        # Display company info
        info = get_stock_info(ticker)
        if info is not None and 'longName' in info:
            st.markdown(f"### {info.get('longName', ticker)}")
        
        # Create columns for metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Price Multiples")
            
            if metrics['Market Cap'] != 'N/A':
                market_cap = metrics['Market Cap']
                if market_cap > 1e9:
                    market_cap_str = f"${market_cap / 1e9:.2f}B"
                else:
                    market_cap_str = f"${market_cap / 1e6:.2f}M"
                st.metric("Market Cap", market_cap_str)
            else:
                st.metric("Market Cap", "N/A")
            
            st.metric("P/E Ratio (TTM)", metrics['P/E Ratio'])
            st.metric("Forward P/E", metrics['Forward P/E'])
            st.metric("PEG Ratio", metrics['PEG Ratio'])
            st.metric("Price/Sales", metrics['Price/Sales'])
            st.metric("Price/Book", metrics['Price/Book'])
        
        with col2:
            st.markdown("#### Enterprise Value Metrics")
            
            if metrics['Enterprise Value'] != 'N/A':
                enterprise_value = metrics['Enterprise Value']
                if enterprise_value > 1e9:
                    ev_str = f"${enterprise_value / 1e9:.2f}B"
                else:
                    ev_str = f"${enterprise_value / 1e6:.2f}M"
                st.metric("Enterprise Value", ev_str)
            else:
                st.metric("Enterprise Value", "N/A")
            
            st.metric("EV/Revenue", metrics['EV/Revenue'])
            st.metric("EV/EBITDA", metrics['EV/EBITDA'])
            
            st.markdown("#### Dividend Metrics")
            st.metric("Dividend Yield", metrics['Dividend Yield'])
            st.metric("Dividend Rate", metrics['Dividend Rate'])
            st.metric("Payout Ratio", metrics['Payout Ratio'])
        
        with col3:
            st.markdown("#### Profitability Metrics")
            st.metric("Profit Margin", metrics['Profit Margin'])
            st.metric("Operating Margin", metrics['Operating Margin'])
            st.metric("ROE", metrics['ROE'])
            st.metric("ROA", metrics['ROA'])
            
            st.markdown("#### Growth Metrics")
            st.metric("Revenue Growth", metrics['Revenue Growth'])
            st.metric("Earnings Growth", metrics['Earnings Growth'])
        
        # Valuation interpretation
        st.subheader("Valuation Interpretation")
        
        # P/E ratio interpretation
        pe_ratio = metrics['P/E Ratio']
        if pe_ratio != 'N/A':
            try:
                pe_ratio = float(pe_ratio)
                if pe_ratio < 10:
                    st.markdown("📉 **P/E Ratio:** The stock appears to be potentially undervalued compared to typical market valuations, or the market expects earnings to decline.")
                elif pe_ratio < 20:
                    st.markdown("🟢 **P/E Ratio:** The stock is trading at a reasonable valuation relative to its earnings.")
                elif pe_ratio < 30:
                    st.markdown("🟡 **P/E Ratio:** The stock is trading at a premium valuation, suggesting high growth expectations.")
                else:
                    st.markdown("🔴 **P/E Ratio:** The stock is trading at a significant premium, indicating very high growth expectations or possibly overvaluation.")
            except:
                st.markdown("**P/E Ratio:** Unable to interpret value.")
        else:
            st.markdown("**P/E Ratio:** No data available for interpretation.")
        
        # PEG ratio interpretation
        peg_ratio = metrics['PEG Ratio']
        if peg_ratio != 'N/A':
            try:
                peg_ratio = float(peg_ratio)
                if peg_ratio < 1:
                    st.markdown("🟢 **PEG Ratio:** The stock may be undervalued relative to its growth rate.")
                elif peg_ratio < 2:
                    st.markdown("🟡 **PEG Ratio:** The stock is reasonably valued relative to its growth rate.")
                else:
                    st.markdown("🔴 **PEG Ratio:** The stock may be overvalued relative to its growth rate.")
            except:
                st.markdown("**PEG Ratio:** Unable to interpret value.")
        else:
            st.markdown("**PEG Ratio:** No data available for interpretation.")
        
        # Dividend yield interpretation
        div_yield = metrics['Dividend Yield']
        if div_yield != 'N/A':
            try:
                div_yield = float(div_yield)
                if div_yield == 0:
                    st.markdown("ℹ️ **Dividend Yield:** The company does not pay dividends.")
                elif div_yield < 2:
                    st.markdown("ℹ️ **Dividend Yield:** The stock offers a low dividend yield.")
                elif div_yield < 4:
                    st.markdown("🟢 **Dividend Yield:** The stock offers a moderate dividend yield.")
                else:
                    st.markdown("🟢 **Dividend Yield:** The stock offers a high dividend yield.")
            except:
                st.markdown("**Dividend Yield:** Unable to interpret value.")
        else:
            st.markdown("**Dividend Yield:** No data available for interpretation.")
    else:
        st.error(f"Unable to retrieve valuation metrics for {ticker}")

def display_peer_comparison(ticker):
    st.subheader("Peer Comparison")
    
    # Get peer comparison data
    peer_data = get_peer_comparison(ticker)
    
    if peer_data is not None:
        # Create a heatmap for the comparison
        st.dataframe(peer_data, hide_index=True)
        
        # Select metrics for visualization
        metrics_to_plot = ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio', 'EV/EBITDA']
        selected_metrics = st.multiselect(
            "Select metrics to visualize",
            metrics_to_plot,
            default=['P/E Ratio', 'Forward P/E']
        )
        
        if selected_metrics:
            # Convert data for plotting
            plot_data = {}
            companies = peer_data['Symbol'].tolist()
            
            for metric in selected_metrics:
                if metric in peer_data.columns:
                    values = []
                    for val in peer_data[metric]:
                        try:
                            values.append(float(val))
                        except:
                            values.append(None)
                    plot_data[metric] = values
            
            if plot_data:
                # Create bar charts for each selected metric
                for metric, values in plot_data.items():
                    if None not in values:
                        fig = px.bar(
                            x=companies,
                            y=values,
                            title=f"{metric} Comparison",
                            labels={'x': 'Company', 'y': metric},
                            color=values,
                            color_continuous_scale='Viridis'
                        )
                        
                        # Add average line
                        avg_value = sum(values) / len(values)
                        fig.add_hline(
                            y=avg_value,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Average: {avg_value:.2f}"
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("Peer Comparison Interpretation")
        
        try:
            company_name = peer_data.loc[peer_data['Symbol'] == ticker, 'Company'].iloc[0]
            
            # Check P/E ratio compared to peers
            if 'P/E Ratio' in peer_data.columns:
                pe_values = []
                for val in peer_data['P/E Ratio']:
                    try:
                        pe_values.append(float(val))
                    except:
                        pe_values.append(None)
                
                # Remove None values
                pe_values = [val for val in pe_values if val is not None]
                
                if pe_values:
                    avg_pe = sum(pe_values) / len(pe_values)
                    company_pe = None
                    
                    for i, symbol in enumerate(peer_data['Symbol']):
                        if symbol == ticker:
                            try:
                                company_pe = float(peer_data['P/E Ratio'].iloc[i])
                            except:
                                pass
                    
                    if company_pe is not None:
                        if company_pe < avg_pe * 0.8:
                            st.markdown(f"🟢 **P/E Ratio Comparison:** {company_name} is trading at a significant discount compared to peers (P/E of {company_pe:.2f} vs. peer average of {avg_pe:.2f}).")
                        elif company_pe < avg_pe:
                            st.markdown(f"🟢 **P/E Ratio Comparison:** {company_name} is trading at a slight discount compared to peers (P/E of {company_pe:.2f} vs. peer average of {avg_pe:.2f}).")
                        elif company_pe < avg_pe * 1.2:
                            st.markdown(f"🟡 **P/E Ratio Comparison:** {company_name} is trading at a slight premium compared to peers (P/E of {company_pe:.2f} vs. peer average of {avg_pe:.2f}).")
                        else:
                            st.markdown(f"🔴 **P/E Ratio Comparison:** {company_name} is trading at a significant premium compared to peers (P/E of {company_pe:.2f} vs. peer average of {avg_pe:.2f}).")
            
            # Check profit margin compared to peers
            if 'Profit Margin (%)' in peer_data.columns:
                margin_values = []
                for val in peer_data['Profit Margin (%)']:
                    try:
                        margin_values.append(float(val))
                    except:
                        margin_values.append(None)
                
                # Remove None values
                margin_values = [val for val in margin_values if val is not None]
                
                if margin_values:
                    avg_margin = sum(margin_values) / len(margin_values)
                    company_margin = None
                    
                    for i, symbol in enumerate(peer_data['Symbol']):
                        if symbol == ticker:
                            try:
                                company_margin = float(peer_data['Profit Margin (%)'].iloc[i])
                            except:
                                pass
                    
                    if company_margin is not None:
                        if company_margin > avg_margin * 1.2:
                            st.markdown(f"🟢 **Profit Margin Comparison:** {company_name} has a significantly higher profit margin compared to peers ({company_margin:.2f}% vs. peer average of {avg_margin:.2f}%).")
                        elif company_margin > avg_margin:
                            st.markdown(f"🟢 **Profit Margin Comparison:** {company_name} has a slightly higher profit margin compared to peers ({company_margin:.2f}% vs. peer average of {avg_margin:.2f}%).")
                        elif company_margin > avg_margin * 0.8:
                            st.markdown(f"🟡 **Profit Margin Comparison:** {company_name} has a slightly lower profit margin compared to peers ({company_margin:.2f}% vs. peer average of {avg_margin:.2f}%).")
                        else:
                            st.markdown(f"🔴 **Profit Margin Comparison:** {company_name} has a significantly lower profit margin compared to peers ({company_margin:.2f}% vs. peer average of {avg_margin:.2f}%).")
        
        except Exception as e:
            st.warning(f"Could not generate peer comparison interpretation: {str(e)}")
    else:
        st.warning(f"Unable to retrieve peer comparison data for {ticker}")

def display_valuation_models(ticker, data):
    st.subheader("Valuation Models")
    
    # PE-based fair value estimation
    st.markdown("### P/E-Based Valuation Model")
    
    # Get PE-based fair value
    fair_value = estimate_fair_value_pe(ticker)
    
    if fair_value is not None:
        # Display fair value estimates
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Inputs")
            st.markdown(f"- Current Price: ${fair_value['Current Price']:.2f}")
            st.markdown(f"- EPS (TTM): ${fair_value['EPS (TTM)']:.2f}")
            st.markdown(f"- Current P/E: {fair_value['Current P/E']:.2f}")
            st.markdown(f"- Industry Avg P/E: {fair_value['Industry Avg P/E']:.2f}")
            st.markdown(f"- Historical Avg P/E: {fair_value['Historical Avg P/E']:.2f}")
        
        with col2:
            st.markdown("#### Fair Value Estimates")
            
            current_price = fair_value['Current Price']
            industry_fair_value = fair_value['Fair Value (Industry P/E)']
            historical_fair_value = fair_value['Fair Value (Historical P/E)']
            avg_fair_value = fair_value['Average Fair Value']
            
            # Industry PE based estimate
            industry_pct = ((industry_fair_value / current_price) - 1) * 100
            industry_color = "green" if industry_pct > 0 else "red"
            st.markdown(f"- Industry P/E-based: ${industry_fair_value:.2f} (<span style='color:{industry_color}'>{industry_pct:.1f}%</span>)", unsafe_allow_html=True)
            
            # Historical PE based estimate
            hist_pct = ((historical_fair_value / current_price) - 1) * 100
            hist_color = "green" if hist_pct > 0 else "red"
            st.markdown(f"- Historical P/E-based: ${historical_fair_value:.2f} (<span style='color:{hist_color}'>{hist_pct:.1f}%</span>)", unsafe_allow_html=True)
            
            # Average fair value
            avg_pct = ((avg_fair_value / current_price) - 1) * 100
            avg_color = "green" if avg_pct > 0 else "red"
            st.markdown(f"- Average Fair Value: ${avg_fair_value:.2f} (<span style='color:{avg_color}'>{avg_pct:.1f}%</span>)", unsafe_allow_html=True)
        
        # Create a visualization of fair value range
        fair_values = [
            fair_value['Fair Value (Discounted)'],
            fair_value['Fair Value (Industry P/E)'],
            fair_value['Fair Value (Historical P/E)'],
            fair_value['Fair Value (Premium)']
        ]
        
        value_labels = [
            "Discounted",
            "Industry P/E",
            "Historical P/E",
            "Premium"
        ]
        
        # Create a fair value range chart
        fig = go.Figure()
        
        # Add current price line
        fig.add_vline(
            x=current_price,
            line_dash="solid",
            line_color="red",
            annotation_text="Current Price",
            annotation_position="top"
        )
        
        # Add average fair value line
        fig.add_vline(
            x=avg_fair_value,
            line_dash="dash",
            line_color="green",
            annotation_text="Avg Fair Value",
            annotation_position="bottom"
        )
        
        # Add fair value range
        min_fair_value = min(fair_values)
        max_fair_value = max(fair_values)
        
        # Add some margin for visualization
        value_range = max_fair_value - min_fair_value
        chart_min = min(min_fair_value, current_price) - (value_range * 0.1)
        chart_max = max(max_fair_value, current_price) + (value_range * 0.1)
        
        # Add value points
        for i, value in enumerate(fair_values):
            fig.add_trace(
                go.Scatter(
                    x=[value],
                    y=[0.5],
                    mode="markers+text",
                    marker=dict(size=15, color="blue"),
                    text=[value_labels[i]],
                    textposition="top center",
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Fair Value Range Based on P/E",
            xaxis_title="Price ($)",
            yaxis=dict(visible=False),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(range=[chart_min, chart_max])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Valuation summary
        st.markdown("#### Valuation Summary")
        
        upside_potential = fair_value['Potential Upside/Downside']
        if upside_potential > 20:
            st.markdown(f"🟢 **Strong Buy:** The stock appears significantly undervalued with {upside_potential:.1f}% potential upside based on P/E valuation models.")
        elif upside_potential > 10:
            st.markdown(f"🟢 **Buy:** The stock appears moderately undervalued with {upside_potential:.1f}% potential upside based on P/E valuation models.")
        elif upside_potential > -10:
            st.markdown(f"🟡 **Hold:** The stock appears fairly valued with {upside_potential:.1f}% potential upside/downside based on P/E valuation models.")
        elif upside_potential > -20:
            st.markdown(f"🔴 **Sell:** The stock appears moderately overvalued with {upside_potential:.1f}% potential downside based on P/E valuation models.")
        else:
            st.markdown(f"🔴 **Strong Sell:** The stock appears significantly overvalued with {upside_potential:.1f}% potential downside based on P/E valuation models.")
    else:
        st.warning("Unable to calculate fair value based on P/E model. This could be due to negative earnings or missing data.")
    
    # DCF Model explanation
    with st.expander("Discounted Cash Flow (DCF) Model"):
        st.markdown("""
        ### Discounted Cash Flow (DCF) Model Explanation
        
        The Discounted Cash Flow (DCF) model estimates the value of an investment based on its expected future cash flows, discounted to present value.
        
        #### Key Components of a DCF Model:
        1. **Forecast Period**: Typically 5-10 years of projected cash flows
        2. **Cash Flow Projections**: Estimated free cash flows for each year in the forecast period
        3. **Terminal Value**: The value of all cash flows beyond the forecast period
        4. **Discount Rate**: Usually the Weighted Average Cost of Capital (WACC)
        5. **Present Value Calculation**: Discounting all projected cash flows to present value
        
        #### Limitations of DCF:
        - Highly sensitive to input assumptions
        - Difficult to accurately forecast cash flows
        - Discount rate selection can significantly impact valuation
        - Terminal value often represents a large portion of the total value
        
        In a full implementation, this section would include a complete DCF model with customizable growth rates, discount rates, and other parameters.
        """)

def display_price_ratios(ticker, data):
    st.subheader("Price Ratio Analysis")
    
    # Get stock info
    info = get_stock_info(ticker)
    
    if info is not None:
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Create columns for different ratio charts
        col1, col2 = st.columns(2)
        
        with col1:
            # PE Ratio
            pe_ratio = info.get('trailingPE', None)
            forward_pe = info.get('forwardPE', None)
            
            if pe_ratio is not None:
                st.metric("P/E Ratio (TTM)", f"{pe_ratio:.2f}")
                
                if forward_pe is not None:
                    pe_change = ((forward_pe / pe_ratio) - 1) * 100
                    st.metric("Forward P/E", f"{forward_pe:.2f}", f"{pe_change:.1f}%")
                
                # Get sector info for comparison
                sector = info.get('sector', 'Technology')
                
                # Approximated sector average PEs
                sector_pe_map = {
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
                
                sector_pe = sector_pe_map.get(sector, 20)
                
                # Create PE comparison chart
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=["Current P/E", "Forward P/E", f"{sector} Sector Avg"],
                        y=[pe_ratio, forward_pe if forward_pe is not None else 0, sector_pe],
                        text=[f"{pe_ratio:.2f}", f"{forward_pe:.2f}" if forward_pe is not None else "N/A", f"{sector_pe:.2f}"],
                        textposition="auto",
                        marker_color=["blue", "green", "gray"]
                    )
                )
                
                fig.update_layout(
                    title=f"P/E Ratio Comparison",
                    height=400,
                    yaxis_title="P/E Ratio"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("P/E ratio data not available for this stock.")
        
        with col2:
            # Price to Sales
            ps_ratio = info.get('priceToSalesTrailing12Months', None)
            
            if ps_ratio is not None:
                st.metric("Price/Sales Ratio", f"{ps_ratio:.2f}")
                
                # Get price to book
                pb_ratio = info.get('priceToBook', None)
                if pb_ratio is not None:
                    st.metric("Price/Book Ratio", f"{pb_ratio:.2f}")
                
                # Create PS & PB comparison chart
                fig = go.Figure()
                
                # Approximate industry averages (would be better with real data)
                sector = info.get('sector', 'Technology')
                
                # Approximated sector average ratios
                sector_ps_map = {
                    'Technology': 7,
                    'Financial Services': 3,
                    'Healthcare': 5,
                    'Consumer Defensive': 2,
                    'Consumer Cyclical': 2,
                    'Energy': 1.5,
                    'Utilities': 2,
                    'Communication Services': 3,
                    'Basic Materials': 2,
                    'Industrials': 2,
                    'Real Estate': 6
                }
                
                sector_pb_map = {
                    'Technology': 6,
                    'Financial Services': 1.5,
                    'Healthcare': 4,
                    'Consumer Defensive': 3,
                    'Consumer Cyclical': 3,
                    'Energy': 1.5,
                    'Utilities': 1.5,
                    'Communication Services': 2.5,
                    'Basic Materials': 2,
                    'Industrials': 3,
                    'Real Estate': 2
                }
                
                sector_ps = sector_ps_map.get(sector, 3)
                sector_pb = sector_pb_map.get(sector, 2.5)
                
                labels = []
                values = []
                colors = []
                
                # Add PS ratio data
                labels.extend(["P/S Ratio", f"{sector} Sector P/S Avg"])
                values.extend([ps_ratio, sector_ps])
                colors.extend(["blue", "lightblue"])
                
                # Add PB ratio data if available
                if pb_ratio is not None:
                    labels.extend(["P/B Ratio", f"{sector} Sector P/B Avg"])
                    values.extend([pb_ratio, sector_pb])
                    colors.extend(["green", "lightgreen"])
                
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        text=[f"{val:.2f}" for val in values],
                        textposition="auto",
                        marker_color=colors
                    )
                )
                
                fig.update_layout(
                    title=f"Price/Sales & Price/Book Comparison",
                    height=400,
                    yaxis_title="Ratio Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Price/Sales ratio data not available for this stock.")
        
        # Historic PE Ratio Trend (if available)
        st.subheader("Historical Valuation Metrics")
        st.warning("Historical valuation metrics tracking requires additional data sources and would be implemented in a production version. This would show trends in P/E, P/S, P/B, and other valuation metrics over time.")
        
        # Valuation Comparison
        st.subheader("Relative Valuation Summary")
        
        # Get various metrics
        pe_ratio = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        pb_ratio = info.get('priceToBook', None)
        peg_ratio = info.get('pegRatio', None)
        
        # Create a scoring system based on relative valuation
        valuation_scores = []
        
        # PE Ratio scoring
        if pe_ratio is not None:
            sector = info.get('sector', 'Technology')
            sector_pe = sector_pe_map.get(sector, 20)
            
            if pe_ratio < sector_pe * 0.7:
                valuation_scores.append(("P/E Ratio", "Significantly undervalued", 2))
            elif pe_ratio < sector_pe * 0.9:
                valuation_scores.append(("P/E Ratio", "Moderately undervalued", 1))
            elif pe_ratio < sector_pe * 1.1:
                valuation_scores.append(("P/E Ratio", "Fairly valued", 0))
            elif pe_ratio < sector_pe * 1.3:
                valuation_scores.append(("P/E Ratio", "Moderately overvalued", -1))
            else:
                valuation_scores.append(("P/E Ratio", "Significantly overvalued", -2))
        
        # PEG Ratio scoring
        if peg_ratio is not None:
            if peg_ratio < 0.8:
                valuation_scores.append(("PEG Ratio", "Significantly undervalued", 2))
            elif peg_ratio < 1.0:
                valuation_scores.append(("PEG Ratio", "Moderately undervalued", 1))
            elif peg_ratio < 1.5:
                valuation_scores.append(("PEG Ratio", "Fairly valued", 0))
            elif peg_ratio < 2.0:
                valuation_scores.append(("PEG Ratio", "Moderately overvalued", -1))
            else:
                valuation_scores.append(("PEG Ratio", "Significantly overvalued", -2))
        
        # P/S Ratio scoring
        if ps_ratio is not None:
            sector = info.get('sector', 'Technology')
            sector_ps = sector_ps_map.get(sector, 3)
            
            if ps_ratio < sector_ps * 0.7:
                valuation_scores.append(("P/S Ratio", "Significantly undervalued", 2))
            elif ps_ratio < sector_ps * 0.9:
                valuation_scores.append(("P/S Ratio", "Moderately undervalued", 1))
            elif ps_ratio < sector_ps * 1.1:
                valuation_scores.append(("P/S Ratio", "Fairly valued", 0))
            elif ps_ratio < sector_ps * 1.3:
                valuation_scores.append(("P/S Ratio", "Moderately overvalued", -1))
            else:
                valuation_scores.append(("P/S Ratio", "Significantly overvalued", -2))
        
        # P/B Ratio scoring
        if pb_ratio is not None:
            sector = info.get('sector', 'Technology')
            sector_pb = sector_pb_map.get(sector, 2.5)
            
            if pb_ratio < sector_pb * 0.7:
                valuation_scores.append(("P/B Ratio", "Significantly undervalued", 2))
            elif pb_ratio < sector_pb * 0.9:
                valuation_scores.append(("P/B Ratio", "Moderately undervalued", 1))
            elif pb_ratio < sector_pb * 1.1:
                valuation_scores.append(("P/B Ratio", "Fairly valued", 0))
            elif pb_ratio < sector_pb * 1.3:
                valuation_scores.append(("P/B Ratio", "Moderately overvalued", -1))
            else:
                valuation_scores.append(("P/B Ratio", "Significantly overvalued", -2))
        
        # Display valuation summary
        if valuation_scores:
            # Create a dataframe for display
            valuation_df = pd.DataFrame(valuation_scores, columns=["Metric", "Assessment", "Score"])
            
            # Calculate total score
            total_score = sum(valuation_df["Score"])
            
            # Display the table
            st.dataframe(valuation_df[["Metric", "Assessment"]], hide_index=True)
            
            # Display overall assessment
            st.subheader("Overall Valuation Assessment")
            
            if total_score >= 4:
                st.markdown("🟢 **Significantly Undervalued:** Based on multiple valuation metrics, this stock appears to be trading at a substantial discount to its fair value.")
            elif total_score >= 2:
                st.markdown("🟢 **Moderately Undervalued:** Based on multiple valuation metrics, this stock appears to be trading below its fair value.")
            elif total_score >= -1:
                st.markdown("🟡 **Fairly Valued:** Based on multiple valuation metrics, this stock appears to be trading close to its fair value.")
            elif total_score >= -3:
                st.markdown("🔴 **Moderately Overvalued:** Based on multiple valuation metrics, this stock appears to be trading above its fair value.")
            else:
                st.markdown("🔴 **Significantly Overvalued:** Based on multiple valuation metrics, this stock appears to be trading at a substantial premium to its fair value.")
            
            # Add valuation score
            st.metric("Valuation Score", total_score, help="Scores: >4: Significantly Undervalued, 2-3: Moderately Undervalued, -1-1: Fairly Valued, -3--2: Moderately Overvalued, <-3: Significantly Overvalued")
        else:
            st.warning("Insufficient data to generate a valuation assessment.")
    else:
        st.error(f"Unable to retrieve company information for {ticker}")
