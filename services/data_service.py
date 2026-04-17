import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
import cachetools
from config import settings

# Shared cache instance
_price_cache = cachetools.TTLCache(maxsize=200, ttl=settings.prediction_cache_ttl_seconds)

def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV with MultiIndex flattening and 512MB RAM optimization."""
    cache_key = f"{ticker}_{period}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        # 1. Download first
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        # 2. Immediately check if empty
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        # 3. FIX: Flatten MultiIndex columns AFTER download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 4. Standardize columns to Title Case (Open, High, Low, Close, Volume)
        df.columns = [str(c).capitalize() for c in df.columns]
        
        required = ["Open", "High", "Low", "Close", "Volume"]
        # Filter only existing required columns
        df = df[[c for c in required if c in df.columns]].copy()
        df.dropna(inplace=True)

        _price_cache[cache_key] = df
        return df

    except Exception as exc:
        logger.error(f"yFinance failed for {ticker}: {exc}")
        raise RuntimeError(f"Could not fetch data for {ticker}")

def get_latest_sequence(ticker: str):
    """Retrieve the T=60 window for LSTM inference."""
    # Note: Ensure compute_features and normalise_features are defined in this file
    # or imported at the top to avoid circular imports.
    from services.data_service import compute_features, normalise_features, FEATURE_COLS
    
    df = fetch_ohlcv(ticker, period="2y")
    
    # Generate technical indicators
    features = compute_features(df)
    # Apply rolling Z-score normalization (Eq. 2)
    norm = normalise_features(features)
    
    # Ensure we use only the columns expected by the LSTM model
    cols = [c for c in FEATURE_COLS if c in norm.columns]
    
    # Get the most recent 60 days
    latest_window = norm[cols].iloc[-settings.lstm_lookback:].values
    
    if latest_window.shape[0] < settings.lstm_lookback:
        raise RuntimeError(f"Insufficient history for {ticker}. Need {settings.lstm_lookback} days.")
        
    # Reshape for PyTorch: (batch, seq_len, features)
    seq = latest_window.astype(np.float32)[np.newaxis, ...]
    
    return seq, float(df["Close"].iloc[-1]), df
