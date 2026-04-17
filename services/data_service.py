import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
import cachetools
from config import settings

_price_cache = cachetools.TTLCache(maxsize=200, ttl=settings.prediction_cache_ttl_seconds)

def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV with MultiIndex flattening and 512MB RAM optimization."""
    cache_key = f"{ticker}_{period}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        # threads=False prevents CPU spikes that trigger Render's OOM killer
        if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
    except Exception as exc:
        logger.error(f"yFinance failed for {ticker}: {exc}")
        raise RuntimeError(f"Could not fetch data for {ticker}")

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # CRITICAL: Fix for yfinance 0.2.40+ MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize columns to standard casing
    df.columns = [c.capitalize() for c in df.columns]
    
    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in required if c in df.columns]].copy()
    df.dropna(inplace=True)

    _price_cache[cache_key] = df
    return df

def get_latest_sequence(ticker: str):
    """Retrieve the T=60 window for LSTM inference."""
    # Use 2y instead of 3y or 5y to stay under 512MB
    df = fetch_ohlcv(ticker, period="2y")
    from services.data_service import compute_features, normalise_features, FEATURE_COLS
    
    features = compute_features(df)
    norm = normalise_features(features)
    
    # Use the cleaned FEATURE_COLS list
    cols = [c for c in FEATURE_COLS if c in norm.columns]
    latest_window = norm[cols].iloc[-settings.lstm_lookback:].values
    
    if latest_window.shape[0] < settings.lstm_lookback:
        raise RuntimeError(f"Insufficient history for {ticker}")
        
    seq = latest_window.astype(np.float32)[np.newaxis, ...]
    return seq, float(df["Close"].iloc[-1]), df
