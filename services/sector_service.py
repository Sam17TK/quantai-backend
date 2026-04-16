"""
services/sector_service.py

Cross-sector cascade propagation (Section VI.F) and
Relative Performance Scoring (Section VI.E — Eq. 20).

The cascade model represents the economy as a directed weighted graph
G = (V, E, W) where W_ij is the partial correlation of sector i's
daily returns with sector j's next-day returns.
Propagation: I^(t+1) = W̃ · I^(t), capped at 3 iterations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from loguru import logger

import yfinance as yf
import cachetools

from config import settings

# ── Sector labels (11 GICS) ──────────────────────────────────────────────────
SECTORS = list(settings.sector_etfs.keys())
ETFS    = list(settings.sector_etfs.values())

# ── Cache ─────────────────────────────────────────────────────────────────────
_sector_cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=10, ttl=1800)

# ─────────────────────────────────────────────────────────────────────────────
# Sector ETF Data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sector_data(period: str = "1y") -> pd.DataFrame:
    """
    Download daily adjusted close prices for all 11 sector ETFs.
    Returns DataFrame (date index × sector ETF columns).
    """
    cache_key = f"sector_etfs_{period}"
    if cache_key in _sector_cache:
        return _sector_cache[cache_key]

    try:
        df = yf.download(
            ETFS, period=period, auto_adjust=True, progress=False, threads=True
        )["Close"]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[ETFS].dropna()
        _sector_cache[cache_key] = df
        return df
    except Exception as exc:
        logger.error(f"Sector ETF download failed: {exc}")
        raise


def get_sector_returns(period: str = "1y") -> pd.DataFrame:
    """Daily log-returns for sector ETFs."""
    prices = fetch_sector_data(period)
    return np.log(prices / prices.shift(1)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Correlation-weighted Dependency Graph (Section VI.F)
# ─────────────────────────────────────────────────────────────────────────────

def build_dependency_graph(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute W where W_ij = partial correlation of sector i returns
    with sector j NEXT-DAY returns (lead-lag correlation matrix).
    Row-normalise to get W̃.
    Returns (11, 11) row-normalised matrix.
    """
    n = len(SECTORS)
    W = np.zeros((n, n))

    for i, etf_i in enumerate(ETFS):
        for j, etf_j in enumerate(ETFS):
            if i == j:
                continue
            ri = returns[etf_i].values[:-1]    # day t
            rj = returns[etf_j].values[1:]     # day t+1
            if len(ri) < 10:
                continue
            corr = float(np.corrcoef(ri, rj)[0, 1])
            W[i, j] = max(corr, 0)   # keep positive spillovers only

    # Row-normalise: W̃_ij = W_ij / Σ_k W_ik
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm = np.where(row_sums > 1e-8, W / row_sums, 0)
    return W_norm


def propagate_cascade(
    initial_impacts: np.ndarray,
    W_norm: np.ndarray,
    n_iterations: int = 3,
) -> List[np.ndarray]:
    """
    Iterative cascade propagation (Eq. 21).
    Returns list of impact vectors for iterations 0, 1, 2.
    """
    steps = [initial_impacts.copy()]
    impact = initial_impacts.copy()
    for _ in range(n_iterations - 1):
        impact = W_norm @ impact
        steps.append(impact.copy())
    return steps


def estimate_event_sector_impact(
    category: str,
    sentiment: float,
) -> np.ndarray:
    """
    Map macro event category + sentiment to initial sector impact vector I^(0).
    Uses a handcrafted impact template derived from historical analogy matching
    (simplified proxy for FAISS analogy search — Section VI.D).
    Returns (11,) array of estimated % sector impacts.
    """
    # Base templates derived from paper's historical event database (Table XXI context)
    IMPACT_TEMPLATES: Dict[str, Dict[str, float]] = {
        "Interest Rate Decision": {
            "Financials": 0.8, "Real Estate": -0.7, "Utilities": -0.5,
            "Technology": -0.3, "Consumer Discretionary": -0.2,
        },
        "Inflation / CPI Release": {
            "Consumer Staples": -0.4, "Consumer Discretionary": -0.5,
            "Materials": 0.3, "Energy": 0.3, "Financials": 0.2,
        },
        "Geopolitical Conflict": {
            "Energy": 0.8, "Defense/Industrials": 0.4, "Technology": -0.3,
            "Consumer Discretionary": -0.3, "Utilities": 0.2,
        },
        "Earnings Surprise": {
            "Technology": 0.6, "Consumer Discretionary": 0.4,
        },
        "Market Crash / Rally": {
            "Technology": -0.8, "Financials": -0.7, "Energy": -0.4,
            "Utilities": 0.2, "Consumer Staples": 0.1,
        },
        "Merger & Acquisition": {
            "Technology": 0.3, "Healthcare": 0.2, "Financials": 0.2,
        },
        "Regulatory / Legal Action": {
            "Technology": -0.4, "Financials": -0.3,
        },
        "Natural Disaster": {
            "Energy": 0.3, "Industrials": -0.2, "Consumer Staples": -0.1,
        },
        "Trade Policy": {
            "Technology": -0.4, "Industrials": -0.3, "Materials": -0.2,
            "Consumer Discretionary": -0.2,
        },
        "Currency / FX Crisis": {
            "Technology": -0.3, "Industrials": -0.2,
            "Consumer Staples": -0.1, "Materials": -0.2,
        },
        "CEO / Leadership Change": {
            "Technology": 0.2,
        },
        "IPO / Delisting": {},
    }

    template = IMPACT_TEMPLATES.get(category, {})
    impact_vec = np.zeros(len(SECTORS))
    sign = 1 if sentiment >= 0 else -1

    for i, sector in enumerate(SECTORS):
        base = template.get(sector, 0.0)
        # Scale by sentiment magnitude
        impact_vec[i] = base * abs(sentiment) * sign

    return impact_vec


# ─────────────────────────────────────────────────────────────────────────────
# Relative Performance Score — Eq. 20
# ─────────────────────────────────────────────────────────────────────────────

def compute_rps(
    ticker: str,
    predicted_return: float,
    peer_returns: List[float],
    sentiment_score: float,
    atr_ratio: float = 1.0,        # ATR_i / ATR_max (volatility credibility weight)
    sector_event_impact: float = 0.0,
    alpha: float = 0.15,
    beta: float = 0.10,
) -> float:
    """
    Relative Performance Score (Eq. 20):
    RPS_i = (R̂_i - R̄_s) / σ_s · w_v + α·S_i + β·I_sector

    Args:
        predicted_return : LSTM-predicted 7-day return for ticker (%)
        peer_returns     : list of predicted returns for all sector peers
        sentiment_score  : aggregate FinBERT-equivalent sentiment
        atr_ratio        : volatility-adjusted credibility weight
        sector_event_impact: cascade-estimated sector impact
    """
    if not peer_returns or len(peer_returns) < 2:
        return 0.0

    R_bar = float(np.mean(peer_returns))
    sigma_s = float(np.std(peer_returns)) + 1e-6
    w_v = max(0.0, 1.0 - atr_ratio)   # lower weight for high-vol stocks

    rps = (predicted_return - R_bar) / sigma_s * w_v
    rps += alpha * sentiment_score
    rps += beta * sector_event_impact
    return round(float(rps), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Sector Overview (for GET /api/sectors/overview)
# ─────────────────────────────────────────────────────────────────────────────

def get_sector_summary() -> List[Dict]:
    """
    Returns current sector-level overview including ETF price + 1-day change.
    """
    cache_key = "sector_summary"
    if cache_key in _sector_cache:
        return _sector_cache[cache_key]

    try:
        df = fetch_sector_data("5d")
        summaries = []
        for sector, etf in settings.sector_etfs.items():
            if etf not in df.columns:
                continue
            series = df[etf].dropna()
            if len(series) < 2:
                continue
            price_now  = float(series.iloc[-1])
            price_prev = float(series.iloc[-2])
            chg_1d     = (price_now - price_prev) / price_prev * 100
            summaries.append({
                "sector":       sector,
                "etf_ticker":   etf,
                "current_price":round(price_now, 4),
                "change_1d_pct":round(chg_1d, 4),
            })
        _sector_cache[cache_key] = summaries
        return summaries
    except Exception as exc:
        logger.error(f"Sector summary failed: {exc}")
        return []
