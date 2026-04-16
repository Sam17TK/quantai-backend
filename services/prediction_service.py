"""
services/prediction_service.py

Central orchestration layer (Layer 3 + 4 of the 5-layer architecture).
Assembles:
  1. Data ingestion + feature engineering
  2. LSTM inference
  3. Monte Carlo simulation
  4. Sentiment scoring
  5. Sector RPS computation
  6. IOS ranking
"""
from __future__ import annotations

import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger

import cachetools

from config import settings
from services.data_service import (
    get_latest_sequence, get_company_info, get_current_price, FEATURE_COLS
)
from services.lstm_model import run_inference
from services.monte_carlo import run_monte_carlo
from services.sentiment_service import get_ticker_sentiment, fetch_recent_news, aggregate_sentiment
from services.sector_service import compute_rps, get_sector_summary
try:
    from services.llm_service import generate_narrative as _llm_narrative
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

# Per-ticker TTL cache
_pred_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=200, ttl=settings.prediction_cache_ttl_seconds
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-Stock Full Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_ticker(ticker: str, use_cache: bool = True) -> Dict:
    """
    End-to-end prediction for one ticker.
    Returns a dict matching FullPredictionPackage schema.
    """
    ticker = ticker.upper()
    cache_key = f"pred_{ticker}"

    if use_cache and cache_key in _pred_cache:
        logger.debug(f"{ticker}: Serving from cache")
        return _pred_cache[cache_key]

    logger.info(f"{ticker}: Running full prediction pipeline")

    # 1 ── Data & Features ────────────────────────────────────────────────────
    sequence, price_now, df_ohlcv = get_latest_sequence(ticker)
    info = get_company_info(ticker)
    sector = info.get("sector", settings.ticker_sector.get(ticker, "Unknown"))

    # Trailing log-returns for GARCH
    import numpy as np
    close = df_ohlcv["Close"]
    log_rets_history = np.log(close / close.shift(1)).dropna().values[-252:]

    # 2 ── LSTM Inference ─────────────────────────────────────────────────────
    lstm_result = run_inference(ticker, sequence, price_now)

    # 3 ── Monte Carlo ─────────────────────────────────────────────────────────
    # Disaggregate 7-day cumulative return into daily
    cum_log = math.log(1 + lstm_result["predicted_return"] / 100 + 1e-10)
    daily_log = [cum_log / settings.lstm_forecast_horizon] * settings.lstm_forecast_horizon

    mc_result = run_monte_carlo(
        price_now=price_now,
        predicted_daily_returns=daily_log,
        log_returns_history=log_rets_history,
        n_paths=settings.mc_paths,
    )

    # 4 ── Sentiment ───────────────────────────────────────────────────────────
    articles = fetch_recent_news()
    sentiment = aggregate_sentiment(articles, ticker)

    # 5 ── RPS ─────────────────────────────────────────────────────────────────
    # Peer returns: use same universe tickers in the sector (cached predictions)
    peer_tickers = [
        t for t, s in settings.ticker_sector.items()
        if s == sector and t != ticker
    ]
    # Use a quick momentum proxy for peer returns (avoid infinite recursion)
    peer_returns = _get_peer_returns_fast(peer_tickers)
    rps = compute_rps(
        ticker=ticker,
        predicted_return=lstm_result["predicted_return"],
        peer_returns=peer_returns,
        sentiment_score=float(sentiment),
        sector_event_impact=0.0,   # cascade impact injected separately
    )

    # 6 ── Historical OHLCV for chart (last 60 days) ───────────────────────────
    hist = df_ohlcv.iloc[-60:].copy()
    historical_prices = [
        {
            "date":   str(idx.date()),
            "open":   round(float(row["Open"]), 4),
            "high":   round(float(row["High"]), 4),
            "low":    round(float(row["Low"]), 4),
            "close":  round(float(row["Close"]), 4),
            "volume": float(row["Volume"]),
        }
        for idx, row in hist.iterrows()
    ]

    result = {
        "lstm": {
            "ticker":             ticker,
            "company_name":       info.get("name", ticker),
            "sector":             sector,
            "current_price":      round(price_now, 4),
            "predicted_prices":   lstm_result["predicted_prices"],
            "predicted_return_7d":lstm_result["predicted_return"],
            "trend": {
                "label":       lstm_result["trend_label"],
                "probability": lstm_result["trend_probs"][lstm_result["trend_label"]],
            },
            "trend_probabilities": lstm_result["trend_probs"],
            "confidence_score":    lstm_result["confidence"],
            "prediction_interval_60": [
                mc_result["pi_50"][0], mc_result["pi_50"][1]
            ],
            "prediction_interval_80": [
                mc_result["pi_80"][0], mc_result["pi_80"][1]
            ],
            "shap_features": lstm_result["shap_features"],
            "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
        },
        "monte_carlo": {
            "ticker": ticker,
            **mc_result,
        },
        "sentiment_score":  float(sentiment),
        "rps":              float(rps),
        "narrative":        None,   # populated by LLM layer below
        "historical_prices":historical_prices,
    }

    # 7 ── LLM Narrative ───────────────────────────────────────────────────────
    if _LLM_AVAILABLE:
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're inside an async context; schedule as fire-and-forget
                # narrative will be None on first call, cached on subsequent
            except RuntimeError:
                loop = None
            if loop is None:
                narrative = asyncio.run(_llm_narrative(result))
                result["narrative"] = narrative
        except Exception as exc:
            logger.warning(f"{ticker}: LLM narrative skipped — {exc}")

    _pred_cache[cache_key] = result
    logger.info(
        f"{ticker}: done — trend={lstm_result['trend_label']} "
        f"ret={lstm_result['predicted_return']:.2f}% "
        f"conf={lstm_result['confidence']:.2%}"
    )
    return result


def _get_peer_returns_fast(tickers: List[str]) -> List[float]:
    """
    Quick momentum-based return estimate for sector peers without full LSTM.
    Uses 5-day trailing return as proxy.
    """
    import yfinance as yf
    results = []
    if not tickers:
        return results
    try:
        data = yf.download(tickers[:10], period="10d", auto_adjust=True,
                           progress=False, threads=True)["Close"]
        if data.empty:
            return results
        for t in (tickers[:10] if not isinstance(tickers, list) else tickers[:10]):
            if t in data.columns:
                series = data[t].dropna()
                if len(series) >= 5:
                    ret = (series.iloc[-1] - series.iloc[-5]) / series.iloc[-5] * 100
                    results.append(float(ret))
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Investment Opportunity Score (Section XII — Eq. 31–32)
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(values: List[float]) -> List[float]:
    """Min-max normalisation (Eq. 32)."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-8:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_opportunities(
    tickers: Optional[List[str]] = None,
    confidence_threshold: float = settings.default_confidence_threshold,
    top_n: int = 10,
) -> List[Dict]:
    """
    Compute IOS for all tickers (or subset) and return top_n ranked.
    IOS_i = w1·R̂_i + w2·Conf_i + w3·RPS_i + w4·S_i + w5·MoR_i  (Eq. 31)
    """
    if tickers is None:
        tickers = settings.coverage_universe[:20]   # limit for speed

    predictions: List[Dict] = []
    for ticker in tickers:
        try:
            pred = predict_ticker(ticker)
            lstm = pred["lstm"]
            mc   = pred["monte_carlo"]
            if lstm["confidence_score"] < confidence_threshold:
                continue

            # Monte Carlo reward-to-risk ratio MoR = P_profit / |VaR95%|
            var_pct = abs(mc["var_95_pct"]) + 1e-6
            mor = mc["prob_profit"] / var_pct

            predictions.append({
                "ticker":         ticker,
                "company_name":   lstm["company_name"],
                "sector":         lstm["sector"],
                "current_price":  lstm["current_price"],
                "predicted_return":lstm["predicted_return_7d"],
                "confidence":     lstm["confidence_score"],
                "rps":            pred["rps"],
                "sentiment":      pred["sentiment_score"],
                "monte_carlo_reward_risk": mor,
                "trend":          lstm["trend"]["label"],
            })
        except Exception as exc:
            logger.warning(f"IOS: skip {ticker} — {exc}")
            continue

    if not predictions:
        return []

    # Normalise each component
    ret_norm   = _minmax([p["predicted_return"]           for p in predictions])
    conf_norm  = _minmax([p["confidence"]                 for p in predictions])
    rps_norm   = _minmax([p["rps"]                        for p in predictions])
    sent_norm  = _minmax([p["sentiment"]                  for p in predictions])
    mor_norm   = _minmax([p["monte_carlo_reward_risk"]    for p in predictions])

    w = settings
    for i, p in enumerate(predictions):
        ios = (
            w.ios_w1 * ret_norm[i]  +
            w.ios_w2 * conf_norm[i] +
            w.ios_w3 * rps_norm[i]  +
            w.ios_w4 * sent_norm[i] +
            w.ios_w5 * mor_norm[i]
        )
        p["ios"] = round(float(ios), 4)

    predictions.sort(key=lambda p: p["ios"], reverse=True)
    for rank, p in enumerate(predictions[:top_n], start=1):
        p["rank"] = rank

    return predictions[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Analysis (POST /api/portfolio/analyse)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_portfolio(holdings: List[Dict]) -> Dict:
    """
    Aggregate risk and return metrics for a user-submitted portfolio.
    """
    analysed = []
    total_value = 0.0

    for h in holdings:
        ticker = h["ticker"].upper()
        shares = float(h.get("shares", 1))
        try:
            price = get_current_price(ticker)
            mv = shares * price
            total_value += mv
            pred = predict_ticker(ticker)
            lstm = pred["lstm"]
            mc   = pred["monte_carlo"]
            analysed.append({
                "ticker":           ticker,
                "sector":           lstm["sector"],
                "current_price":    price,
                "market_value":     round(mv, 2),
                "weight":           0.0,   # filled below
                "predicted_return": lstm["predicted_return_7d"],
                "confidence":       lstm["confidence_score"],
                "var_95_pct":       mc["var_95_pct"],
                "trend":            lstm["trend"]["label"],
            })
        except Exception as exc:
            logger.warning(f"Portfolio: skip {ticker} — {exc}")

    if not analysed or total_value < 1e-6:
        return {"error": "No valid holdings analysed"}

    # Weights
    for h in analysed:
        h["weight"] = round(h["market_value"] / total_value, 4)

    # Weighted portfolio metrics
    wtd_ret  = sum(h["weight"] * h["predicted_return"] for h in analysed)
    wtd_conf = sum(h["weight"] * h["confidence"]       for h in analysed)

    # Approximate portfolio VaR (simplified: weighted avg)
    ptf_var  = sum(h["weight"] * h["var_95_pct"] for h in analysed)
    ptf_cvar = ptf_var * 1.35   # approx ES ≈ 1.35 × VaR for equity portfolios

    # Sector concentration
    sector_conc: Dict[str, float] = {}
    for h in analysed:
        sector_conc[h["sector"]] = sector_conc.get(h["sector"], 0) + h["weight"]

    n_sectors = len(sector_conc)
    hhi = sum(w**2 for w in sector_conc.values())
    diversity = 1 - hhi   # Herfindahl diversity score

    return {
        "total_value":              round(total_value, 2),
        "holdings":                 analysed,
        "portfolio_var_95":         round(ptf_var, 4),
        "portfolio_cvar_95":        round(ptf_cvar, 4),
        "sector_concentration":     {k: round(v, 4) for k, v in sector_conc.items()},
        "diversification_score":    round(diversity, 4),
        "weighted_predicted_return":round(wtd_ret, 4),
        "weighted_confidence":      round(wtd_conf, 4),
        "generated_at":             datetime.now(tz=timezone.utc).isoformat(),
    }
