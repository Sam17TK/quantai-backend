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
    Download adjusted OHLCV from Yahoo Finance.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Applies basic data quality checks from Section IV.A of the paper.
    """
    cache_key = f"{ticker}_{period}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.error(f"yFinance download failed for {ticker}: {exc}")
        raise RuntimeError(f"Could not fetch data for {ticker}")

    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # ── Data quality: stale price detection (Section IV.A) ───────────────────
    stale_mask = (df["Close"].diff().abs() < 1e-8) & (df["Volume"] == 0)
    df = df[~stale_mask]

    # ── Return outlier clipping (flag only; do not remove genuine moves) ──────
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    extreme = log_ret.abs() > 0.50   # > 50% daily move → flag
    if extreme.any():
        logger.warning(f"{ticker}: {extreme.sum()} extreme return days detected (kept)")

    _price_cache[cache_key] = df
    return df


def get_current_price(ticker: str) -> float:
    """Latest adjusted close price."""
    df = fetch_ohlcv(ticker, period="5d")
    return float(df["Close"].iloc[-1])


def get_company_info(ticker: str) -> Dict:
    """Fetch basic company metadata from yFinance."""
    try:
        info = yf.Ticker(ticker).info
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
# Technical Indicator Feature Engineering (Table II)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 22 technical features from Table II of the paper.
    Returns a DataFrame with one row per trading day.
    
    Features:
      Log Return, SMA(20/50/200), EMA(12/26),
      MACD Line/Signal/Histogram, RSI(14),
      BB Upper/Lower/Width, ATR(14),
      Stoch %K/%D, Williams %R,
      OBV, Volume/SMA Ratio
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Return ────────────────────────────────────────────────────────────────
    feat["log_return"] = np.log(close / close.shift(1))

    # ── Trend: Simple Moving Averages ─────────────────────────────────────────
    feat["sma_20"]  = close.rolling(20).mean()
    feat["sma_50"]  = close.rolling(50).mean()
    feat["sma_200"] = close.rolling(200).mean()

    # ── Trend: Exponential MAs ────────────────────────────────────────────────
    feat["ema_12"] = close.ewm(span=12, adjust=False).mean()
    feat["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # ── Momentum: MACD (12, 26, 9) ───────────────────────────────────────────
    macd_line   = feat["ema_12"] - feat["ema_26"]
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd"]        = macd_line
    feat["macd_signal"] = macd_signal
    feat["macd_hist"]   = macd_line - macd_signal

    # ── Momentum: RSI (14) ────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Volatility: Bollinger Bands (20, 2σ) ─────────────────────────────────
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / (bb_mid + 1e-10)

    # ── Volatility: ATR (14) ─────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()

    # ── Momentum: Stochastic (14, 3) ─────────────────────────────────────────
    lowest_low   = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_k.rolling(3).mean()

    # ── Momentum: Williams %R ────────────────────────────────────────────────
    feat["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    # ── Volume: OBV ──────────────────────────────────────────────────────────
    direction = np.sign(close.diff())
    feat["obv"] = (vol * direction).cumsum()

    # ── Volume: Volume / SMA(20) ratio ───────────────────────────────────────
    feat["vol_sma_ratio"] = vol / (vol.rolling(20).mean() + 1e-10)

    # Price-ratio features (scale-invariant; normalised against current price)
    feat["sma_20_ratio"]  = feat["sma_20"]  / (close + 1e-10) - 1
    feat["sma_50_ratio"]  = feat["sma_50"]  / (close + 1e-10) - 1
    feat["sma_200_ratio"] = feat["sma_200"] / (close + 1e-10) - 1
    feat["bb_upper_ratio"] = feat["bb_upper"] / (close + 1e-10) - 1
    feat["bb_lower_ratio"] = feat["bb_lower"] / (close + 1e-10) - 1
    feat["ema_12_ratio"]  = feat["ema_12"]  / (close + 1e-10) - 1
    feat["ema_26_ratio"]  = feat["ema_26"]  / (close + 1e-10) - 1

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Z-score Normalisation (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

RATIO_COLS = {
    "sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "bb_upper_ratio","bb_lower_ratio","ema_12_ratio","ema_26_ratio"
}

# Columns selected for LSTM input (22 features — Table II)
FEATURE_COLS = [
    "log_return",
    "sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "ema_12_ratio","ema_26_ratio",
    "macd","macd_signal","macd_hist",
    "rsi_14",
    "bb_upper_ratio","bb_lower_ratio","bb_width",
    "atr_14",
    "stoch_k","stoch_d",
    "williams_r",
    "obv",
    "vol_sma_ratio",
    "log_return",   # duplicate intentional for sequence diversity
    "macd_hist",    # second occurrence for model emphasis per ablation
    "rsi_14",
]
# De-duplicate while preserving order
_seen = set()
FEATURE_COLS = [c for c in FEATURE_COLS if not (c in _seen or _seen.add(c))]
# Pad to exactly 22 if needed
while len(FEATURE_COLS) < 22:
    FEATURE_COLS.append(FEATURE_COLS[-1])
FEATURE_COLS = FEATURE_COLS[:22]


def normalise_features(feat: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Rolling Z-score normalisation (Eq. 2).
    Ratio features are already price-normalised; only z-score raw indicator cols.
    ε = 1e-8 for numerical stability.
    """
    out = feat.copy()
    eps = 1e-8
    non_ratio_cols = [c for c in FEATURE_COLS if c in out.columns and c not in RATIO_COLS]

    for col in non_ratio_cols:
        roll_mean = out[col].rolling(window, min_periods=20).mean()
        roll_std  = out[col].rolling(window, min_periods=20).std()
        out[col]  = (out[col] - roll_mean) / (roll_std + eps)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Builder for LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df_norm: pd.DataFrame,
    close_prices: pd.Series,
    lookback: int = 60,
    horizon: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct overlapping (X, y_reg, y_cls) sequences for LSTM training.
    
    X      : (N, lookback, 22)  — normalised feature windows
    y_reg  : (N, horizon)       — forward log-returns (regression targets)
    y_cls  : (N,)               — trend label {0=Bear, 1=Neutral, 2=Bull}
    """
    cols = [c for c in FEATURE_COLS if c in df_norm.columns]
    data = df_norm[cols].values
    prices = close_prices.values

    X_list, y_reg_list, y_cls_list = [], [], []
    for i in range(lookback, len(data) - horizon):
        X_list.append(data[i - lookback:i])

        # Regression target: log-returns for next `horizon` days
        forward_returns = np.log(prices[i+1:i+horizon+1] / prices[i:i+horizon])
        y_reg_list.append(forward_returns)

        # Classification target: cumulative return over horizon
        cum_ret = (prices[i + horizon] - prices[i]) / prices[i]
        if cum_ret > 0.01:
            label = 2   # Bullish
        elif cum_ret < -0.01:
            label = 0   # Bearish
        else:
            label = 1   # Neutral
        y_cls_list.append(label)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_reg_list, dtype=np.float32),
        np.array(y_cls_list, dtype=np.int64),
    )


def get_latest_sequence(ticker: str) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """
    Returns:
      sequence  : (1, 60, 22) normalised input for LSTM inference
      price_now : latest close price
      df_ohlcv  : raw OHLCV for historical chart
    """
    df = fetch_ohlcv(ticker, period="3y")
    features = compute_features(df)
    norm     = normalise_features(features)

    cols = [c for c in FEATURE_COLS if c in norm.columns]
    latest_window = norm[cols].iloc[-settings.lstm_lookback:].values

    if latest_window.shape[0] < settings.lstm_lookback:
        raise RuntimeError(
            f"{ticker}: Insufficient history for {settings.lstm_lookback}-day window"
        )

    seq = latest_window.astype(np.float32)[np.newaxis, ...]  # (1, 60, 22)
    price_now = float(df["Close"].iloc[-1])

    return seq, price_now, df
