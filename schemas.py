"""
schemas.py — All Pydantic v2 request/response models for the QuantAI API.
Structure mirrors the paper's output specification (Sections V, VII, XII).
"""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Literal
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────

class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TrendLabel(BaseModel):
    label: Literal["Bullish", "Neutral", "Bearish"]
    probability: float = Field(..., ge=0, le=1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Prediction (Section V)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMPrediction(BaseModel):
    ticker: str
    company_name: str
    sector: str
    current_price: float
    predicted_prices: List[float]           # 7 daily predicted prices
    predicted_return_7d: float              # cumulative % return
    trend: TrendLabel
    trend_probabilities: Dict[str, float]   # {Bullish, Neutral, Bearish}
    confidence_score: float = Field(..., ge=0, le=1)
    prediction_interval_60: List[float]     # [lo, hi] 60% PI
    prediction_interval_80: List[float]     # [lo, hi] 80% PI
    shap_features: List[ShapFeature]
    timestamp: datetime


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["positive", "negative"]
    description: str


LSTMPrediction.model_rebuild()


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo (Section VII)
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloResult(BaseModel):
    ticker: str
    current_price: float
    paths_simulated: int
    expected_price_median: float
    pi_50: List[float]   # [p25, p75]
    pi_80: List[float]   # [p10, p90]
    pi_90: List[float]   # [p5, p95]
    var_95: float        # absolute $ loss at 5th percentile
    var_95_pct: float    # as % of current price
    cvar_95: float       # expected tail loss $
    cvar_95_pct: float
    prob_profit: float   # P(S_T+7 > S_T)
    prob_loss_5pct: float  # P(S_T+7 < 0.95 * S_T)
    distribution_bins: List[float]   # histogram x-axis (price buckets)
    distribution_counts: List[int]   # histogram y-axis (path counts)
    garch_volatility_daily: float    # σ̂ from GARCH(1,1)
    garch_volatility_annual: float


# ─────────────────────────────────────────────────────────────────────────────
# Full Prediction Package (GET /api/predict/{ticker})
# ─────────────────────────────────────────────────────────────────────────────

class FullPredictionPackage(BaseModel):
    lstm: LSTMPrediction
    monte_carlo: MonteCarloResult
    sentiment_score: float              # aggregate FinBERT-equivalent score
    rps: float                          # Relative Performance Score (Section VI.E)
    narrative: Optional[str] = None     # LLM-generated text (Ollama layer)
    historical_prices: List[PricePoint]


# ─────────────────────────────────────────────────────────────────────────────
# Sector (Section VI)
# ─────────────────────────────────────────────────────────────────────────────

class SectorSummary(BaseModel):
    sector: str
    etf_ticker: str
    current_price: float
    change_1d_pct: float
    avg_predicted_return: float
    avg_confidence: float
    stock_count: int
    top_performer: str


class SectorOverview(BaseModel):
    sectors: List[SectorSummary]
    generated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Investment Opportunity Score (Section XII)
# ─────────────────────────────────────────────────────────────────────────────

class OpportunityScore(BaseModel):
    rank: int
    ticker: str
    company_name: str
    sector: str
    current_price: float
    predicted_return: float
    confidence: float
    rps: float
    sentiment: float
    monte_carlo_reward_risk: float
    ios: float = Field(..., ge=0, le=1)    # composite Investment Opportunity Score
    trend: str


class TopOpportunities(BaseModel):
    opportunities: List[OpportunityScore]
    confidence_threshold_used: float
    generated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# News / Events (Section VI.A–C)
# ─────────────────────────────────────────────────────────────────────────────

class NewsEvent(BaseModel):
    event_id: str
    headline: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    macro_category: Optional[str] = None
    macro_confidence: Optional[float] = None
    tickers_mentioned: List[str]


class MacroEvent(BaseModel):
    event_id: str
    headline: str
    category: str
    date: str
    estimated_sector_impacts: Dict[str, float]   # sector → % impact
    analogy_similarity: float
    top_analogues: List[str]


class RecentEventsResponse(BaseModel):
    events: List[NewsEvent]
    macro_events: List[MacroEvent]
    generated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Sector Cascade (Section VI.F)
# ─────────────────────────────────────────────────────────────────────────────

class CascadeStep(BaseModel):
    iteration: int   # 0, 1, 2 (primary, secondary, tertiary)
    sector_impacts: Dict[str, float]


class CascadeResult(BaseModel):
    event_id: str
    headline: str
    category: str
    cascade_steps: List[CascadeStep]
    final_impacts: Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio (POST /api/portfolio/analyse)
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioHolding(BaseModel):
    ticker: str
    shares: float
    purchase_price: Optional[float] = None


class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]


class HoldingAnalysis(BaseModel):
    ticker: str
    sector: str
    current_price: float
    market_value: float
    weight: float
    predicted_return: float
    confidence: float
    var_95_pct: float
    trend: str


class PortfolioAnalysis(BaseModel):
    total_value: float
    holdings: List[HoldingAnalysis]
    portfolio_var_95: float
    portfolio_cvar_95: float
    sector_concentration: Dict[str, float]
    diversification_score: float
    weighted_predicted_return: float
    weighted_confidence: float
    generated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Health / Meta
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    universe_size: int
    model_loaded: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    detail: str
    ticker: Optional[str] = None
