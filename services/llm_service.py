"""
services/llm_service.py

LLM Explainability Layer — Section VIII of the paper.
Uses FREE online LLM APIs (no API key required for basic use):
  1. Groq (primary)   — free tier, LLaMA 3 70B, ~500 tokens/sec
  2. Together AI      — free $25 credit, various open models
  3. Hugging Face     — free Inference API, Mistral/Zephyr
  4. OpenRouter       — free tier, multiple models
  5. Ollama           — local fallback if running

Priority: Groq → HuggingFace → OpenRouter → Ollama → built-in template
All prompts follow Section VIII.B structured chain-of-thought format.
"""
from __future__ import annotations

import os
import json
import httpx
import asyncio
from typing import Optional, Dict, List
from loguru import logger

# ── Free API Endpoints ───────────────────────────────────────────────────────
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_URL   = "https://api.together.xyz/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_URL         = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_URL     = "http://localhost:11434/api/generate"

# Load from environment (all optional — falls back through chain)
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_API_KEY         = os.getenv("HF_API_KEY", "")

SYSTEM_PROMPT = """You are QuantAI, an expert financial analyst assistant built on academic research.
Your role is to explain AI-driven stock predictions clearly for retail investors.
Rules:
- Always express uncertainty; never give absolute price targets
- Use directional language: "is expected to", "may indicate", "based on historical patterns"
- Keep responses under 200 words
- End with a brief risk disclaimer
- Focus on: directional outlook, key drivers, macro context, primary downside risk, probability of profit"""


def _build_narrative_prompt(analysis: Dict) -> str:
    lstm    = analysis.get("lstm", {})
    mc      = analysis.get("monte_carlo", {})
    ticker  = lstm.get("ticker", "UNKNOWN")
    company = lstm.get("company_name", ticker)
    sector  = lstm.get("sector", "Unknown")
    trend   = lstm.get("trend", {}).get("label", "Neutral")
    ret     = lstm.get("predicted_return_7d", 0)
    conf    = lstm.get("confidence_score", 0) * 100
    pi_lo   = mc.get("pi_80", [0, 0])[0]
    pi_hi   = mc.get("pi_80", [0, 0])[1]
    prob    = mc.get("prob_profit", 0) * 100
    var     = mc.get("var_95_pct", 0)
    garch   = mc.get("garch_volatility_annual", 0) * 100
    sent    = analysis.get("sentiment_score", 0)
    rps     = analysis.get("rps", 0)

    shap_features = lstm.get("shap_features", [])
    drivers = "; ".join(
        f"{f['feature']} ({f['description']}, SHAP={f['shap_value']:+.2f})"
        for f in shap_features[:3]
    ) or "momentum and volatility signals"

    return f"""Generate a 150-200 word investment outlook for {ticker} ({company}, {sector} sector).

PREDICTION DATA:
- 7-Day Predicted Return: {ret:+.2f}%
- Trend Direction: {trend}
- Confidence Score: {conf:.0f}%
- 80% Price Range: [${pi_lo:.2f}, ${pi_hi:.2f}]
- Probability of Profit: {prob:.1f}%
- VaR 95% (7-day): {var:.2f}%
- GARCH Annual Volatility: {garch:.1f}%
- News Sentiment Score: {sent:+.3f}
- Sector Relative Performance Score: {rps:+.3f} ({'outperforming' if rps > 0 else 'underperforming'} peers)

KEY DRIVERS (SHAP analysis):
{drivers}

INSTRUCTIONS:
1. Summarise the directional prediction and confidence in one sentence
2. Identify the 2-3 most significant drivers
3. Highlight the primary downside risk
4. State probability of profit clearly
5. End with: "This is an analytical forecast, not financial advice."
"""


def _build_template_narrative(analysis: Dict) -> str:
    """Rule-based fallback when all LLM APIs fail."""
    lstm   = analysis.get("lstm", {})
    mc     = analysis.get("monte_carlo", {})
    ticker = lstm.get("ticker", "UNKNOWN")
    trend  = lstm.get("trend", {}).get("label", "Neutral")
    ret    = lstm.get("predicted_return_7d", 0)
    conf   = lstm.get("confidence_score", 0) * 100
    prob   = mc.get("prob_profit", 0) * 100
    var    = mc.get("var_95_pct", 0)
    pi_lo  = mc.get("pi_80", [0, 0])[0]
    pi_hi  = mc.get("pi_80", [0, 0])[1]
    shap   = lstm.get("shap_features", [])
    driver = shap[0]["description"] if shap else "momentum signals"
    sent   = analysis.get("sentiment_score", 0)
    sent_str = "positive" if sent > 0.05 else "negative" if sent < -0.05 else "neutral"

    direction_str = {
        "Bullish": "a bullish",
        "Bearish": "a bearish",
        "Neutral": "a neutral",
    }.get(trend, "a neutral")

    return (
        f"**{ticker} — {trend} Outlook** ({conf:.0f}% confidence)\n\n"
        f"The model projects {direction_str} 7-day move of **{ret:+.2f}%** with "
        f"an 80% price range of **${pi_lo:.2f}–${pi_hi:.2f}**. "
        f"The primary driver is {driver}. "
        f"Recent news sentiment is **{sent_str}**, providing "
        f"{'additional tailwind' if sent > 0.05 else 'headwind' if sent < -0.05 else 'no directional bias'}.\n\n"
        f"Monte Carlo simulation across 10,000 paths gives a **{prob:.1f}% probability of profit** "
        f"with maximum expected loss (VaR 95%) of **{var:.2f}%** over the 7-day horizon.\n\n"
        f"*This is an analytical forecast, not financial advice.*"
    )


async def _try_groq(prompt: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "max_tokens": 350,
                    "temperature": 0.4,
                },
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning(f"Groq LLM failed: {exc}")
    return None


async def _try_together(prompt: str) -> Optional[str]:
    if not TOGETHER_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                TOGETHER_URL,
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-3-8b-chat-hf",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "max_tokens": 350,
                    "temperature": 0.4,
                },
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning(f"Together AI LLM failed: {exc}")
    return None


async def _try_openrouter(prompt: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://quantai.app",
                    "X-Title": "QuantAI Stock Analysis",
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "max_tokens": 350,
                },
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning(f"OpenRouter LLM failed: {exc}")
    return None


async def _try_huggingface(prompt: str) -> Optional[str]:
    """HuggingFace free Inference API — no key needed for public models."""
    try:
        full_prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
        headers = {"Content-Type": "application/json"}
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                HF_URL,
                headers=headers,
                json={
                    "inputs": full_prompt,
                    "parameters": {"max_new_tokens": 300, "temperature": 0.4, "return_full_text": False},
                },
            )
            if r.status_code == 200:
                result = r.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
    except Exception as exc:
        logger.warning(f"HuggingFace LLM failed: {exc}")
    return None


async def _try_ollama(prompt: str) -> Optional[str]:
    """Local Ollama fallback."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                OLLAMA_URL,
                json={"model": "llama3", "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}", "stream": False},
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
    except Exception:
        pass
    return None


async def generate_narrative(analysis: Dict) -> str:
    """
    Generate LLM investment narrative via free API chain.
    Falls back to rule-based template if all APIs unavailable.
    """
    prompt = _build_narrative_prompt(analysis)

    # Try each provider in priority order
    for provider_fn in [_try_groq, _try_together, _try_openrouter, _try_huggingface, _try_ollama]:
        result = await provider_fn(prompt)
        if result and len(result) > 50:
            logger.info(f"LLM narrative generated via {provider_fn.__name__}")
            return result

    logger.info("All LLM APIs unavailable — using template narrative")
    return _build_template_narrative(analysis)


async def generate_chatbot_response(
    message: str,
    context: Dict,
    history: List[Dict] = None,
) -> str:
    """
    Conversational chatbot response using free LLM.
    Context contains the latest prediction data for relevant tickers.
    """
    if history is None:
        history = []

    system = SYSTEM_PROMPT + """

You have access to real-time stock prediction data. When asked about specific stocks,
use the provided context data. Support these query types:
- Single stock outlook ("What's your view on NVDA?")
- Comparative analysis ("Compare AAPL vs MSFT")
- Sector rotation ("Which sectors look strong?")
- Risk assessment ("How risky is XOM?")
- Top opportunities ("Show me best opportunities")
- Portfolio review ("Review my holdings")
- What-if scenarios ("If Fed raises rates...")

Always cite specific numbers from the context when available."""

    ctx_str = json.dumps(context, indent=2) if context else "No specific stock data loaded."
    user_content = f"Context data:\n{ctx_str}\n\nUser question: {message}"

    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:  # Last 3 exchanges
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": user_content})

    # Try Groq first for speed
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama3-8b-8192", "messages": messages, "max_tokens": 400, "temperature": 0.5},
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # Try OpenRouter free tier
    if OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://quantai.app",
                    },
                    json={
                        "model": "meta-llama/llama-3.1-8b-instruct:free",
                        "messages": messages,
                        "max_tokens": 400,
                    },
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # Rule-based fallback (from websocket.py logic)
    return None  # Signal to use built-in handler
