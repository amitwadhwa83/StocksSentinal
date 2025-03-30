# Stock Analysis Tool

This is a Stock Analysis Tool built using Streamlit and yfinance. It provides various features for analyzing stocks, managing portfolios, and viewing technical indicators and valuation metrics.

## Features

- **Dashboard**: View major indices and perform quick stock lookups.
- **Stock Analysis**: Analyze individual stocks with real-time data.
- **Portfolio Management**: Upload your portfolio and get insights.
- **Technical Indicators**: View SMA, EMA, RSI, MACD, and other indicators.
- **Valuation Metrics**: Evaluate stocks using various valuation models.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

## Configuration

The Streamlit configuration is set in the `.streamlit/config.toml` file:
```toml
[server]
headless = true
address = "localhost"
port = 8501
