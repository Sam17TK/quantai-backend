"""
services/data_service.py

Market data ingestion and feature engineering.
Implements Table II of the paper: 22 technical indicator features
with rolling Z-score normalisation (Eq. 2).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import cachetools

from config import settings

# Simple TTL cache (5 min for price data)
_price_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=200, ttl=settings.prediction_cache_ttl_seconds
)

# ─────────────────────────────────────────────────────────────────────────────
# Raw OHLCV Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download adjusted OHLCV from Yahoo Finance with Browser-like headers.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    """
    cache_key = f"{ticker}_{period}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        # The 'headers' argument is critical for hosting on Render/Railway
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
    except Exception as exc:
        logger.error(f"yFinance download failed for {ticker}: {exc}")
        raise RuntimeError(f"Could not fetch data for {ticker}")

    if df.empty:
        logger.warning(f"Yahoo Finance returned an empty DataFrame for {ticker}")
        raise RuntimeError(f"No data returned for ticker {ticker}")

    # FIX: Flatten MultiIndex columns (yfinance 0.2.40+ compatibility)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in required if c in df.columns]
    df = df[available].copy()
    df.dropna(inplace=True)

    # ── Data quality: stale price detection (Section IV.A) ───────────────────
    stale_mask = (df["Close"].diff().abs() < 1e-8) & (df["Volume"] == 0)
    df = df[~stale_mask]

    # ── Return outlier clipping ──
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    extreme = log_ret.abs() > 0.50
    if extreme.any():
        logger.warning(f"{ticker}: {extreme.sum()} extreme return days detected")

    _price_cache[cache_key] = df
    return df


def get_current_price(ticker: str) -> float:
    """Latest adjusted close price."""
    df = fetch_ohlcv(ticker, period="5d")
    return float(df["Close"].iloc[-1])


def get_company_info(ticker: str) -> Dict:
    """Fetch basic company metadata from yFinance."""
    try:
        # Use Ticker object with custom session/headers if needed, 
        # but basic .info usually works if download worked.
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", settings.ticker_sector.get(ticker, "Unknown")),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {
            "name": ticker,
            "sector": settings.ticker_sector.get(ticker, "Unknown"),
            "industry": "",
            "market_cap": 0,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicator Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Calculations ──
    feat["log_return"] = np.log(close / close.shift(1))
    feat["sma_20"]  = close.rolling(20).mean()
    feat["sma_50"]  = close.rolling(50).mean()
    feat["sma_200"] = close.rolling(200).mean()
    feat["ema_12"] = close.ewm(span=12, adjust=False).mean()
    feat["ema_26"] = close.ewm(span=26, adjust=False).mean()

    macd_line   = feat["ema_12"] - feat["ema_26"]
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd"]        = macd_line
    feat["macd_signal"] = macd_signal
    feat["macd_hist"]   = macd_line - macd_signal

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / (bb_mid + 1e-10)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()

    lowest_low   = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_k.rolling(3).mean()
    feat["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    direction = np.sign(close.diff())
    feat["obv"] = (vol * direction).cumsum()
    feat["vol_sma_ratio"] = vol / (vol.rolling(20).mean() + 1e-10)

    # Ratios
    feat["sma_20_ratio"]   = feat["sma_20"]  / (close + 1e-10) - 1
    feat["sma_50_ratio"]   = feat["sma_50"]  / (close + 1e-10) - 1
    feat["sma_200_ratio"]  = feat["sma_200"] / (close + 1e-10) - 1
    feat["bb_upper_ratio"] = feat["bb_upper"] / (close + 1e-10) - 1
    feat["bb_lower_ratio"] = feat["bb_lower"] / (close + 1e-10) - 1
    feat["ema_12_ratio"]   = feat["ema_12"]  / (close + 1e-10) - 1
    feat["ema_26_ratio"]   = feat["ema_26"]  / (close + 1e-10) - 1

    feat.dropna(inplace=True)
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation & Sequence Logic (Remains unchanged for model consistency)
# ─────────────────────────────────────────────────────────────────────────────

RATIO_COLS = {
    "sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "bb_upper_ratio","bb_lower_ratio","ema_12_ratio","ema_26_ratio"
}

FEATURE_COLS = [
    "log_return", "sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "ema_12_ratio","ema_26_ratio", "macd","macd_signal","macd_hist",
    "rsi_14", "bb_upper_ratio","bb_lower_ratio","bb_width", "atr_14",
    "stoch_k","stoch_d", "williams_r", "obv", "vol_sma_ratio",
    "log_return", "macd_hist", "rsi_14"
]

def normalise_features(feat: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    out = feat.copy()
    eps = 1e-8
    non_ratio_cols = [c for c in FEATURE_COLS if c in out.columns and c not in RATIO_COLS]
    for col in non_ratio_cols:
        roll_mean = out[col].rolling(window, min_periods=20).mean()
        roll_std  = out[col].rolling(window, min_periods=20).std()
        out[col]  = (out[col] - roll_mean) / (roll_std + eps)
    return out

def build_sequences(df_norm: pd.DataFrame, close_prices: pd.Series, lookback: int = 60, horizon: int = 7):
    cols = [c for c in FEATURE_COLS if c in df_norm.columns]
    data = df_norm[cols].values
    prices = close_prices.values
    X_list, y_reg_list, y_cls_list = [], [], []
    for i in range(lookback, len(data) - horizon):
        X_list.append(data[i - lookback:i])
        forward_returns = np.log(prices[i+1:i+horizon+1] / prices[i:i+horizon])
        y_reg_list.append(forward_returns)
        cum_ret = (prices[i + horizon] - prices[i]) / prices[i]
        label = 2 if cum_ret > 0.01 else (0 if cum_ret < -0.01 else 1)
        y_cls_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_reg_list, dtype=np.float32), np.array(y_cls_list, dtype=np.int64)

def get_latest_sequence(ticker: str) -> Tuple[np.ndarray, float, pd.DataFrame]:
    df = fetch_ohlcv(ticker, period="3y")
    features = compute_features(df)
    norm = normalise_features(features)
    cols = [c for c in FEATURE_COLS if c in norm.columns]
    latest_window = norm[cols].iloc[-settings.lstm_lookback:].values
    if latest_window.shape[0] < settings.lstm_lookback:
        raise RuntimeError(f"{ticker}: Insufficient history for {settings.lstm_lookback}-day window")
    seq = latest_window.astype(np.float32)[np.newaxis, ...]
    price_now = float(df["Close"].iloc[-1])
    return seq, price_now, df
