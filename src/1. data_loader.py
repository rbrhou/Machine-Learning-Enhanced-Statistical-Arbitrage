from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_equity_returns(
    tickers: list[str],
    start_date: str,
    end_date: str,
    filename: str = "equity_returns.parquet",
) -> pd.DataFrame:
  
    """Loads equity returns from the local data directory if cached;
    otherwise downloads via yfinance and saves locally.
    """
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / filename

    if file_path.exists():
        print(f"Loading cached returns from {file_path}")
        return pd.read_parquet(file_path)

    print(f"Downloading data for {len(tickers)} assets via yfinance...")
    downloaded_data = yf.download(tickers, start=start_date, end=end_date)
    if downloaded_data is None:
        raise RuntimeError("yfinance returned no data")
    raw_data = downloaded_data["Close"]
    returns = pd.DataFrame(raw_data.pct_change().dropna())

    # Cache locally
    returns.to_parquet(file_path)
    return returns
