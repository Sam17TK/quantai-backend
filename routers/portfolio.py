"""
routers/portfolio.py — POST /api/portfolio/analyse
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from services.prediction_service import analyse_portfolio

router = APIRouter(prefix="/api", tags=["Portfolio"])


class HoldingInput(BaseModel):
    ticker: str
    shares: float = 1.0
    purchase_price: Optional[float] = None


class PortfolioRequest(BaseModel):
    holdings: List[HoldingInput]


@router.post("/portfolio/analyse", summary="Portfolio-level risk and return analysis")
async def analyse(req: PortfolioRequest):
    """
    Accepts a list of holdings (ticker + shares) and returns:
    - Per-holding LSTM prediction, confidence, VaR
    - Portfolio-level weighted VaR, CVaR
    - Sector concentration (Herfindahl diversity score)
    - Weighted predicted 7-day return
    """
    if not req.holdings:
        raise HTTPException(status_code=400, detail="holdings list is empty")
    try:
        holdings_dicts = [h.model_dump() for h in req.holdings]
        result = analyse_portfolio(holdings_dicts)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Portfolio analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
