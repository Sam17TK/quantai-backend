"""
config.py — Central configuration for QuantAI Backend.
Optimized for Render Free Tier (512MB RAM).
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "QuantAI Stock Analysis API"
    app_version: str = "1.0.0"
    debug: bool = True  # Set to True to see detailed logs in Render

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: List[str] = ["*"]

    # ── LSTM (Section V — Paper) ──────────────────────────────────────────────
    lstm_lookback: int = 60          # T = 60 trading days
    lstm_forecast_horizon: int = 7   # 7-day forecast
    lstm_layer1_units: int = 256
    lstm_layer2_units: int = 128
    lstm_shared_dense: int = 64
    lstm_dropout: float = 0.30
    lstm_recurrent_dropout: float = 0.10
    lstm_batch_size: int = 64
    lstm_lr: float = 0.001
    lstm_huber_delta: float = 1.0
    lstm_lambda_reg: float = 0.6    # λ_r regression weight
    lstm_lambda_cls: float = 0.4    # λ_c classification weight
    lstm_l2_decay: float = 1e-5

    # ── Monte Carlo (Section VII — Paper) ────────────────────────────────────
    # CRITICAL: Reduced from 10,000 to 500-1,000 to prevent OOM Crashes on Render
    mc_paths: int = 500 
    mc_mean_reversion_lambda: float = 0.05
    mc_student_t_fraction: float = 0.05   # 5% fat-tail draws
    mc_student_t_df: int = 5
    mc_var_level: float = 0.05            # 95% VaR

    # ── Feature Engineering (Table II — Paper) ───────────────────────────────
    feature_normalisation_window: int = 252   # 1-year rolling z-score
    num_features: int = 22

    # ── Confidence Score Weights (Section V.D) ───────────────────────────────
    conf_w1: float = 0.4   # classification certainty
    conf_w2: float = 0.4   # entropy complement
    conf_w3: float = 0.2   # historical accuracy proxy

    # ── IOS Weights (Section XII.A) ──────────────────────────────────────────
    ios_w1: float = 0.30   # predicted return
    ios_w2: float = 0.25   # confidence
    ios_w3: float = 0.20   # RPS
    ios_w4: float = 0.10   # sentiment
    ios_w5: float = 0.15   # Monte Carlo reward-to-risk

    # ── Confidence threshold (Table XVIII optimum) ───────────────────────────
    default_confidence_threshold: float = 0.65

    # ── Coverage Universe (Section XI.B) ─────────────────────────────────────
    coverage_universe: List[str] = [
        "AAPL","MSFT","NVDA","GOOGL","META","ORCL","ADBE","CRM","AMD","INTC","QCOM","TXN",
        "JNJ","PFE","UNH","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "JPM","BAC","GS","MS","WFC","C","BLK","AXP","SCHW","USB",
        "AMZN","TSLA","NKE","MCD","HD","SBUX","TGT","LOW","BKNG",
        "XOM","CVX","COP","SLB","OXY","EOG","MPC","PSX","VLO",
    ]

    # ── GICS Sector ETFs for cascade ─────────────────────────────────────────
    sector_etfs: Dict[str, str] = {
        "Technology":             "XLK",
        "Healthcare":             "XLV",
        "Financials":             "XLF",
        "Energy":                 "XLE",
        "Industrials":            "XLI",
        "Consumer Staples":       "XLP",
        "Consumer Discretionary": "XLY",
        "Utilities":              "XLU",
        "Real Estate":            "XLRE",
        "Materials":              "XLB",
        "Communication Services": "XLC",
    }

    ticker_sector: Dict[str, str] = {
        "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
        "GOOGL":"Technology","META":"Technology","ORCL":"Technology",
        "ADBE":"Technology","CRM":"Technology","AMD":"Technology",
        "INTC":"Technology","QCOM":"Technology","TXN":"Technology",
        "JNJ":"Healthcare","PFE":"Healthcare","UNH":"Healthcare",
        "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare",
        "ABT":"Healthcare","DHR":"Healthcare","BMY":"Healthcare","AMGN":"Healthcare",
        "JPM":"Financials","BAC":"Financials","GS":"Financials",
        "MS":"Financials","WFC":"Financials","C":"Financials",
        "BLK":"Financials","AXP":"Financials","SCHW":"Financials","USB":"Financials",
        "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary",
        "NKE":"Consumer Discretionary","MCD":"Consumer Discretionary",
        "HD":"Consumer Discretionary","SBUX":"Consumer Discretionary",
        "TGT":"Consumer Discretionary","LOW":"Consumer Discretionary",
        "BKNG":"Consumer Discretionary",
        "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
        "OXY":"Energy","EOG":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy",
    }

    # ── News / Sentiment ──────────────────────────────────────────────────────
    news_cache_ttl_seconds: int = 900
    prediction_cache_ttl_seconds: int = 300

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()
