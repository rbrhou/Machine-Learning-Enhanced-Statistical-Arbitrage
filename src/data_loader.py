import pandas as pd
import yfinance as yf

def fetch_equity_returns(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads adjusted closing prices and calculates daily percentage returns.
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()
    return returns

def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-centers returns for covariance mapping.
    """
    return returns - returns.mean()
