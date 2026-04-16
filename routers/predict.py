"""
routers/predict.py — /api/predict/{ticker}
"""
from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from services.prediction_service import predict_ticker

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.get("/predict/{ticker}", summary="Full 7-day prediction package for a single ticker")
async def get_prediction(
    ticker: str,
    refresh: bool = Query(False, description="Bypass cache and recompute"),
):
    """
    Returns the complete prediction package:
    - LSTM 7-day forecast with confidence score
    - Monte Carlo VaR / CVaR / probability distribution
    - SHAP feature attributions
    - Sentiment score
    - Relative Performance Score (RPS)
    """
    try:
        result = predict_ticker(ticker.upper(), use_cache=not refresh)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Prediction failed for {ticker}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
