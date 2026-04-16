"""
main.py — QuantAI Stock Analysis API
FastAPI application entry point.

Run with:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config import settings

# ── Logger setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="DEBUG" if settings.debug else "INFO",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────

_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  Universe: {len(settings.coverage_universe)} tickers")
    logger.info(f"  Sectors:  {len(settings.sector_etfs)}")
    logger.info("=" * 60)

    # Pre-warm: check if any model weights exist
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    if os.path.isdir(weights_dir):
        pt_files = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
        if pt_files:
            logger.info(f"Found {len(pt_files)} pre-trained model weight file(s)")
            _model_loaded = True
        else:
            logger.info("No weights found — statistical fallback will be used")
    else:
        logger.info("Weights directory absent — statistical fallback active")

    yield

    logger.info("QuantAI API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## QuantAI — AI-Powered Stock Analysis Platform

Backend implementing the architecture from  
**"An AI-Powered Stock Analysis and Probabilistic Forecasting Platform  
Integrating LSTM, Monte Carlo Simulation, and LLM Explainability"**  
— T. Sanath, IEEE

### Components
| Component | Implementation |
|---|---|
| **LSTM Forecasting** | 2-layer stacked LSTM (256→128 units), dual regression+classification heads |
| **Monte Carlo** | 10,000 GBM paths with GARCH(1,1) volatility + Student-t fat tails |
| **Sentiment** | VADER (drop-in for FinBERT) with exponential decay aggregation |
| **Sector Cascade** | Correlation-weighted directed dependency graph, 3-step propagation |
| **IOS Ranking** | 5-component composite Investment Opportunity Score |
| **Chatbot** | 7-intent WebSocket chatbot (Ollama/LLaMA-compatible) |

### Quick Start
1. `GET /api/predict/AAPL` — full prediction package
2. `GET /api/opportunities/top` — IOS-ranked opportunities
3. `WS /ws/chat` — conversational interface
""",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from routers.predict   import router as predict_router
from routers.market    import router as market_router
from routers.portfolio import router as portfolio_router
from routers.websocket import router as ws_router

# Services imported lazily inside endpoints to avoid circular imports at startup

app.include_router(predict_router)
app.include_router(market_router)
app.include_router(portfolio_router)
app.include_router(ws_router)


# ─────────────────────────────────────────────────────────────────────────────
# Core Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    settings.app_name,
        "version": settings.app_version,
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["Meta"], summary="Health check")
async def health():
    """
    Returns API status, universe size, and model load state.
    """
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    pt_files = []
    if os.path.isdir(weights_dir):
        pt_files = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]

    return {
        "status":       "ok",
        "version":      settings.app_version,
        "universe_size":len(settings.coverage_universe),
        "model_loaded": len(pt_files) > 0,
        "weights_found":len(pt_files),
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/universe", tags=["Meta"], summary="List coverage universe")
async def universe():
    """Returns the full ticker coverage universe with sector mapping."""
    return {
        "tickers": settings.coverage_universe,
        "sector_map": settings.ticker_sector,
        "sector_etfs": settings.sector_etfs,
        "total": len(settings.coverage_universe),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Endpoint (POST /api/train/{ticker})
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/train/{ticker}", tags=["Training"], summary="Train LSTM for a specific ticker")
async def train_ticker(ticker: str, period: str = "10y", epochs: int = 120):
    """
    Trains the LSTM model for a single ticker using walk-forward validation.
    Downloads `period` of data, computes features, trains, and saves weights.
    
    ⚠ This is a long-running operation (1–15 min depending on hardware).
    For production use: run as a background Celery task.
    """
    ticker = ticker.upper()
    if ticker not in set(settings.coverage_universe):
        return JSONResponse(
            status_code=400,
            content={"detail": f"{ticker} not in coverage universe"}
        )
    try:
        from services.data_service import fetch_ohlcv, compute_features, normalise_features, build_sequences  # noqa: E402
        from services.lstm_model import train_model  # noqa: E402

        logger.info(f"Training request for {ticker} ({period}, {epochs} epochs)")
        df = fetch_ohlcv(ticker, period=period)
        features = compute_features(df)
        norm = normalise_features(features)

        X, y_reg, y_cls = build_sequences(norm, df["Close"])
        if len(X) < 100:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Insufficient data for {ticker}: {len(X)} sequences"}
            )

        model = train_model(X, y_reg, y_cls, ticker=ticker, epochs=epochs)
        return {
            "status": "trained",
            "ticker": ticker,
            "sequences": len(X),
            "weights_path": f"weights/{ticker}.pt",
        }
    except Exception as exc:
        logger.exception(f"Training failed for {ticker}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Global Exception Handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
