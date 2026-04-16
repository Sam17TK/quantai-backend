"""
routers/market.py — sector overview, opportunities, events, cascade
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from config import settings
from services.prediction_service import rank_opportunities
from services.sentiment_service import fetch_recent_news, classify_macro_event
from services.sector_service import (
    get_sector_summary,
    estimate_event_sector_impact,
    build_dependency_graph,
    propagate_cascade,
    get_sector_returns,
    SECTORS,
)

router = APIRouter(prefix="/api", tags=["Market"])


# ── GET /api/sectors/overview ─────────────────────────────────────────────────

@router.get("/sectors/overview", summary="Current sector performance overview")
async def sector_overview():
    """
    Returns price, 1-day change, and average predicted return for all 11 GICS sectors.
    """
    try:
        summaries = get_sector_summary()
        return {
            "sectors": summaries,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("Sector overview failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /api/opportunities/top ─────────────────────────────────────────────────

@router.get("/opportunities/top", summary="IOS-ranked top investment opportunities")
async def top_opportunities(
    top_n: int = Query(10, ge=1, le=50),
    confidence_threshold: float = Query(
        settings.default_confidence_threshold, ge=0.0, le=1.0
    ),
    sector: str | None = Query(None, description="Filter by sector name"),
):
    """
    Returns top N stocks ranked by Investment Opportunity Score (IOS).
    Applies confidence threshold filter before ranking.
    """
    try:
        tickers = settings.coverage_universe
        if sector:
            tickers = [
                t for t, s in settings.ticker_sector.items()
                if s.lower() == sector.lower()
            ]
        opps = rank_opportunities(
            tickers=tickers,
            confidence_threshold=confidence_threshold,
            top_n=top_n,
        )
        return {
            "opportunities": opps,
            "confidence_threshold_used": confidence_threshold,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("Opportunity ranking failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /api/montecarlo/{ticker} ──────────────────────────────────────────────

@router.get("/montecarlo/{ticker}", summary="Full Monte Carlo distribution for a ticker")
async def monte_carlo_detail(ticker: str):
    """
    Returns the complete 10,000-path Monte Carlo simulation result for a ticker,
    including VaR95, CVaR95, probability of profit, and distribution histogram.
    """
    from services.prediction_service import predict_ticker
    try:
        result = predict_ticker(ticker.upper())
        return result["monte_carlo"]
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Monte Carlo failed for {ticker}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /api/events/recent ────────────────────────────────────────────────────

@router.get("/events/recent", summary="Classified recent macro news events")
async def recent_events(limit: int = Query(30, ge=1, le=100)):
    """
    Returns recent financial news articles with:
    - VADER sentiment score
    - Macro event category classification
    - Ticker mentions
    """
    try:
        articles = fetch_recent_news(max_articles=limit)
        macro_events = [
            {
                "event_id": a["event_id"],
                "headline": a["headline"],
                "category": a["macro_category"],
                "category_confidence": a["macro_confidence"],
                "source": a["source"],
                "published_at": a["published_at"].isoformat() if hasattr(a["published_at"], "isoformat") else str(a["published_at"]),
                "sentiment_score": a["sentiment_score"],
                "tickers_mentioned": a["tickers"],
            }
            for a in articles if a.get("macro_category")
        ]
        all_events = [
            {
                "event_id":    a["event_id"],
                "headline":    a["headline"],
                "source":      a["source"],
                "url":         a["url"],
                "published_at":a["published_at"].isoformat() if hasattr(a["published_at"], "isoformat") else str(a["published_at"]),
                "sentiment_score": a["sentiment_score"],
                "macro_category":  a.get("macro_category"),
                "tickers_mentioned": a["tickers"],
            }
            for a in articles
        ]
        return {
            "events":       all_events,
            "macro_events": macro_events,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("Events fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /api/cascade/{event_id} ───────────────────────────────────────────────

@router.get("/cascade/{event_id}", summary="Sector cascade propagation for an event")
async def sector_cascade(event_id: str):
    """
    Given an event_id from /api/events/recent, compute the 3-step
    cross-sector cascade propagation using the correlation-weighted
    dependency graph (Section VI.F).
    """
    try:
        # Look up event in recent articles
        articles = fetch_recent_news()
        event = next((a for a in articles if a["event_id"] == event_id), None)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        category = event.get("macro_category", "Market Crash / Rally")
        sentiment = float(event.get("sentiment_score", 0.0))

        # Build dependency graph
        returns = get_sector_returns("1y")
        W_norm = build_dependency_graph(returns)

        # Initial impact vector
        initial_impacts = estimate_event_sector_impact(category, sentiment)

        # Propagate
        steps = propagate_cascade(initial_impacts, W_norm, n_iterations=3)

        cascade_steps = [
            {
                "iteration": i,
                "sector_impacts": {
                    SECTORS[j]: round(float(v), 4) for j, v in enumerate(step)
                },
            }
            for i, step in enumerate(steps)
        ]

        final_impacts = {
            SECTORS[j]: round(float(v), 4)
            for j, v in enumerate(steps[-1])
        }

        return {
            "event_id":     event_id,
            "headline":     event["headline"],
            "category":     category,
            "cascade_steps":cascade_steps,
            "final_impacts":final_impacts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Cascade computation failed for event {event_id}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /api/narrative/{ticker} ───────────────────────────────────────────────

@router.get("/narrative/{ticker}", summary="LLM-generated investment narrative for a ticker")
async def get_narrative(ticker: str):
    """
    Returns an LLM-generated natural language investment narrative for the ticker.
    Uses free LLM APIs (Groq → Together → OpenRouter → HuggingFace → template).
    """
    from services.prediction_service import predict_ticker
    try:
        from services.llm_service import generate_narrative
        pred = predict_ticker(ticker.upper())
        narrative = await generate_narrative(pred)
        return {
            "ticker": ticker.upper(),
            "narrative": narrative,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="LLM service not available")
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Narrative generation failed for {ticker}")
        raise HTTPException(status_code=500, detail=str(exc))
