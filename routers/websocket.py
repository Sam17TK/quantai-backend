"""
routers/websocket.py

WebSocket endpoints:
  /ws/prices  — Real-time price streaming (1-sec updates during market hours)
  /ws/chat    — Conversational chatbot (Section VIII.D of paper)
              Routes to Ollama if available, else uses built-in analytical engine.
"""
from __future__ import annotations

import asyncio
import json
import time
import re
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import yfinance as yf

from config import settings
from services.prediction_service import predict_ticker, rank_opportunities

try:
    from services.llm_service import generate_chatbot_response as _llm_chat
    _LLM_CHAT_AVAILABLE = True
except ImportError:
    _LLM_CHAT_AVAILABLE = False

router = APIRouter(tags=["WebSocket"])

# ─────────────────────────────────────────────────────────────────────────────
# /ws/prices — Live Price Stream
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/prices")
async def price_stream(ws: WebSocket):
    """
    Streams live prices for the coverage universe every 5 seconds.
    Payload: { "ticker": "AAPL", "price": 182.34, "timestamp": "..." }
    """
    await ws.accept()
    logger.info("Price stream WebSocket connected")
    tickers_to_stream = settings.coverage_universe[:10]  # first 10 for efficiency

    try:
        while True:
            try:
                data = yf.download(
                    tickers_to_stream,
                    period="1d",
                    interval="1m",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                prices = data["Close"].iloc[-1]
                payload = []
                for t in tickers_to_stream:
                    if t in prices.index:
                        payload.append({
                            "ticker":    t,
                            "price":     round(float(prices[t]), 4),
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        })
                await ws.send_text(json.dumps({"type": "price_batch", "data": payload}))
            except Exception as exc:
                logger.warning(f"Price stream error: {exc}")
                await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("Price stream WebSocket disconnected")


# ─────────────────────────────────────────────────────────────────────────────
# /ws/chat — Conversational Chatbot
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "single_stock":      r'\b([A-Z]{2,5})\b.*(?:outlook|view|forecast|predict|analysis)',
    "comparative":       r'compare|vs\.|versus',
    "sector_rotation":   r'sector|which sectors|rotation',
    "what_if":           r'if\s+(?:fed|rate|market|inflation)',
    "risk_assessment":   r'(?:risk|var|volatile|volatility)\s+(?:of|for|in)\s+([A-Z]{2,5})',
    "top_opportunities": r'top|best|opportunity|opportunities|recommend',
    "portfolio":         r'portfolio|holdings|my stocks',
}


def _detect_intent(message: str) -> str:
    msg_upper = message.upper()
    msg_lower = message.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, msg_upper if intent == "single_stock" else msg_lower):
            return intent
    return "general"


def _extract_ticker(message: str) -> Optional[str]:
    universe_set = set(settings.coverage_universe)
    matches = re.findall(r'\b([A-Z]{2,5})\b', message.upper())
    for m in matches:
        if m in universe_set:
            return m
    return None


async def _handle_query(message: str, history: list = None) -> str:
    """
    LLM-powered analytical response engine with built-in fallback.
    Tries free LLM APIs first; falls back to rule-based engine.
    """
    if history is None:
        history = []

    intent = _detect_intent(message)
    ticker = _extract_ticker(message)

    # Build context dict for LLM
    context = {"intent": intent}
    if ticker:
        try:
            pred = predict_ticker(ticker)
            context["stock_data"] = {
                "ticker": ticker,
                "trend":  pred["lstm"]["trend"]["label"],
                "return_7d": pred["lstm"]["predicted_return_7d"],
                "confidence": pred["lstm"]["confidence_score"],
                "current_price": pred["lstm"]["current_price"],
                "prob_profit": pred["monte_carlo"]["prob_profit"],
                "var_95_pct": pred["monte_carlo"]["var_95_pct"],
                "pi_80": pred["monte_carlo"]["pi_80"],
                "sentiment": pred["sentiment_score"],
                "rps": pred["rps"],
                "shap": pred["lstm"].get("shap_features", [])[:3],
            }
        except Exception:
            pass

    elif intent == "top_opportunities":
        try:
            opps = rank_opportunities(top_n=5)
            context["opportunities"] = opps
        except Exception:
            pass

    elif intent == "sector_rotation":
        try:
            from services.sector_service import get_sector_summary
            context["sectors"] = get_sector_summary()
        except Exception:
            pass

    # Try LLM first
    if _LLM_CHAT_AVAILABLE:
        try:
            llm_response = await _llm_chat(message, context, history)
            if llm_response and len(llm_response) > 30:
                return llm_response
        except Exception as exc:
            logger.warning(f"LLM chat failed: {exc}")

    # ── Rule-based fallback ──────────────────────────────────────────────────

    # ── Single stock outlook ─────────────────────────────────────────────────
    if intent == "single_stock" and ticker:
        try:
            pred = predict_ticker(ticker)
            lstm = pred["lstm"]
            mc   = pred["monte_carlo"]
            trend    = lstm["trend"]["label"]
            ret      = lstm["predicted_return_7d"]
            conf     = lstm["confidence_score"] * 100
            prob_pr  = mc["prob_profit"] * 100
            var_pct  = mc["var_95_pct"]
            pi_lo    = mc["pi_80"][0]
            pi_hi    = mc["pi_80"][1]
            drivers  = "; ".join(
                f'{f["feature"]} ({f["description"]})'
                for f in (lstm.get("shap_features") or [])[:2]
            )
            return (
                f"**{ticker} — {trend} outlook** (7-day, {conf:.0f}% confidence)\n\n"
                f"Predicted return: **{ret:+.2f}%** "
                f"| 80% price range: **${pi_lo:.2f}–${pi_hi:.2f}**\n"
                f"Probability of profit: **{prob_pr:.1f}%** "
                f"| VaR 95%: **{var_pct:.2f}%**\n\n"
                f"Key drivers: {drivers or 'momentum and volatility signals'}.\n\n"
                f"*This is an analytical forecast, not financial advice.*"
            )
        except Exception as exc:
            return f"Could not generate outlook for {ticker}: {exc}"

    # ── Top opportunities ────────────────────────────────────────────────────
    elif intent == "top_opportunities":
        try:
            opps = rank_opportunities(top_n=5)
            lines = [f"**Top 5 Investment Opportunities (IOS Ranking)**\n"]
            for o in opps:
                lines.append(
                    f"{o['rank']}. **{o['ticker']}** ({o['sector']}) — "
                    f"{o['trend']} | {o['predicted_return']:+.2f}% | "
                    f"Conf: {o['confidence']*100:.0f}% | IOS: {o['ios']:.3f}"
                )
            return "\n".join(lines) + "\n\n*Past performance does not guarantee future results.*"
        except Exception as exc:
            return f"Could not compute opportunities: {exc}"

    # ── Sector rotation ──────────────────────────────────────────────────────
    elif intent == "sector_rotation":
        from services.sector_service import get_sector_summary
        try:
            sectors = get_sector_summary()
            lines = ["**Current Sector Performance**\n"]
            for s in sorted(sectors, key=lambda x: x["change_1d_pct"], reverse=True):
                emoji = "🟢" if s["change_1d_pct"] >= 0 else "🔴"
                lines.append(
                    f"{emoji} **{s['sector']}** ({s['etf_ticker']}): "
                    f"{s['change_1d_pct']:+.2f}% today | ${s['current_price']:.2f}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Sector data unavailable: {exc}"

    # ── Risk assessment ──────────────────────────────────────────────────────
    elif intent == "risk_assessment" and ticker:
        try:
            pred = predict_ticker(ticker)
            mc = pred["monte_carlo"]
            return (
                f"**{ticker} Risk Profile (7-day Monte Carlo, {mc['paths_simulated']:,} paths)**\n\n"
                f"GARCH daily volatility: **{mc['garch_volatility_daily']*100:.2f}%** "
                f"(annualised: {mc['garch_volatility_annual']*100:.1f}%)\n"
                f"VaR 95%: **{mc['var_95_pct']:.2f}%** (worst-case 5th percentile)\n"
                f"CVaR 95%: **{mc['cvar_95_pct']:.2f}%** (expected tail loss)\n"
                f"Probability of loss > 5%: **{mc['prob_loss_5pct']*100:.1f}%**\n"
                f"80% price range: **${mc['pi_80'][0]:.2f}–${mc['pi_80'][1]:.2f}**"
            )
        except Exception as exc:
            return f"Risk data unavailable for {ticker}: {exc}"

    # ── Comparative ─────────────────────────────────────────────────────────
    elif intent == "comparative":
        tickers_found = [
            t for t in re.findall(r'\b([A-Z]{2,5})\b', message.upper())
            if t in set(settings.coverage_universe)
        ][:3]
        if len(tickers_found) < 2:
            return "Please mention at least two covered tickers to compare. E.g. 'Compare AAPL vs MSFT risk'"
        rows = []
        for t in tickers_found:
            try:
                pred = predict_ticker(t)
                lstm = pred["lstm"]
                mc   = pred["monte_carlo"]
                rows.append(
                    f"**{t}**: {lstm['trend']['label']} | "
                    f"return {lstm['predicted_return_7d']:+.2f}% | "
                    f"conf {lstm['confidence_score']*100:.0f}% | "
                    f"VaR {mc['var_95_pct']:.2f}%"
                )
            except Exception:
                rows.append(f"**{t}**: data unavailable")
        return "**Comparative Analysis**\n\n" + "\n".join(rows)

    # ── General / what-if ────────────────────────────────────────────────────
    else:
        suggestions = [
            "• 'What's your view on NVDA?' — single stock outlook",
            "• 'Compare AMD vs INTC risk' — comparative analysis",
            "• 'Which sectors look strong?' — sector rotation overview",
            "• 'How risky is XOM?' — VaR/CVaR risk profile",
            "• 'Show top opportunities' — IOS-ranked ideas",
        ]
        return (
            "I'm **QuantAI**, your AI-powered financial analysis assistant. "
            "I can help you with:\n\n" + "\n".join(suggestions) +
            "\n\nAsk about any covered ticker: " +
            ", ".join(settings.coverage_universe[:8]) + " and more."
        )


@router.websocket("/ws/chat")
async def chatbot(ws: WebSocket):
    """
    Conversational chatbot WebSocket.
    Supports intent classification across 7 query types (Table IX — paper).
    In production: forward to Ollama /api/generate for LLaMA 3 responses.
    """
    await ws.accept()
    logger.info("Chatbot WebSocket connected")
    conversation_history: list = []

    # Welcome message
    await ws.send_text(json.dumps({
        "type":    "assistant",
        "content": "Welcome to **QuantAI**. Ask me about stocks, sectors, risk, or top opportunities.",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg_data = json.loads(raw)
                message = msg_data.get("message", raw)
            except json.JSONDecodeError:
                message = raw

            logger.debug(f"Chatbot received: {message[:80]}")

            # Typing indicator
            await ws.send_text(json.dumps({"type": "typing"}))

            await asyncio.sleep(0.2)

            response = await _handle_query(message, conversation_history)

            # Maintain rolling history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})
            if len(conversation_history) > 12:
                conversation_history = conversation_history[-12:]

            await ws.send_text(json.dumps({
                "type":      "assistant",
                "content":   response,
                "intent":    _detect_intent(message),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }))
    except WebSocketDisconnect:
        logger.info("Chatbot WebSocket disconnected")
    except Exception as exc:
        logger.exception(f"Chatbot error: {exc}")
        try:
            await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            pass
