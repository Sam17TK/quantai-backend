"""
main.py — QuantAI Stock Analysis API
FastAPI application entry point.
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

# Import your local settings
try:
    from config import settings
except ImportError:
    # Fallback settings if config.py is missing
    class MockSettings:
        app_name = "QuantAI"
        app_version = "1.0.0"
        debug = True
        allowed_origins = ["*"]
        coverage_universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
        sector_etfs = {"Technology": "XLK", "Energy": "XLE"}
        ticker_sector = {"AAPL": "Technology"}
    settings = MockSettings()

# ── Logger setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="DEBUG" if settings.debug else "INFO",
)

# ── Lifespan (Startup/Shutdown) ──────────────────────────────────────────────
_model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)

    # Pre-warm: check for model weights
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    if os.path.isdir(weights_dir):
        pt_files = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
        _model_loaded = len(pt_files) > 0
        logger.info(f"Models found: {_model_loaded} ({len(pt_files)} files)")
    
    yield
    logger.info("QuantAI API shutting down")

# ── App Initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# ── CORS (Crucial for Frontend Connection) ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to settings.allowed_origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers (Ensure these files exist in your 'routers' folder) ──────────────
try:
    from routers.predict import router as predict_router
    from routers.market import router as market_router
    from routers.portfolio import router as portfolio_router
    from routers.websocket import router as ws_router
    
    app.include_router(predict_router)
    app.include_router(market_router)
    app.include_router(portfolio_router)
    app.include_router(ws_router)
except ImportError as e:
    logger.warning(f"Router import failed: {e}. Ensure 'routers' folder exists.")

# ── Base Endpoints ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version[:6],
        "model_loaded": _model_loaded
    }

@app.get("/api/universe", tags=["Meta"])
async def universe():
    """Matches app.html expectation for sector and ticker mapping"""
    return {
        "tickers": settings.coverage_universe,
        "sector_map": getattr(settings, 'ticker_sector', {}),
        "sector_etfs": settings.sector_etfs,
        "total": len(settings.coverage_universe),
    }

# ── Training Endpoint ────────────────────────────────────────────────────────

@app.post("/api/train/{ticker}", tags=["Training"])
async def train_ticker(ticker: str, period: str = "10y", epochs: int = 120):
    ticker = ticker.upper()
    try:
        # These services must exist in your /services/ folder
        from services.data_service import fetch_ohlcv, compute_features, normalise_features, build_sequences
        from services.lstm_model import train_model

        logger.info(f"Training request for {ticker}")
        df = fetch_ohlcv(ticker, period=period)
        features = compute_features(df)
        norm = normalise_features(features)
        X, y_reg, y_cls = build_sequences(norm, df["Close"])
        
        model = train_model(X, y_reg, y_cls, ticker=ticker, epochs=epochs)
        
        return {"status": "trained", "ticker": ticker, "weights_path": f"weights/{ticker}.pt"}
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

# ── Global Error Handler ─────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )

# ── Entry Point for Render ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable automatically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
