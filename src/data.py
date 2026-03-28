import pandas as pd
import yfinance as yf

def to_yahoo_symbol(idx_ticker: str) -> str:
    """
    Convert IDX ticker (e.g. BBCA) to Yahoo Finance format (BBCA.JK).
    """
    return idx_ticker.strip().upper() + ".JK"

def fetch_ohlc(ticker_ya: str, interval: str, candles: int = 240) -> pd.DataFrame:
    if interval == "1h":
        period = "30d"
    elif interval == "1d":
        period = "2y"
    elif interval == "1wk":
        period = "10y"
    else:
        raise ValueError("Unsupported interval")

    df = yf.download(ticker_ya, period=period, interval=interval, auto_adjust=False)

    # ✅ Flatten MultiIndex columns (yfinance >= 0.2.40 uses MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.tail(candles).copy()
    df.dropna(inplace=True)
    df["AvgPrice"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)
    return df

def get_target_series(df: pd.DataFrame, target_col: str = "Close"):
    """
    Extract the target series (e.g. Close) as numpy array.
    """
    return df[target_col].values