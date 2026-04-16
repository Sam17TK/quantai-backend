"""
services/sentiment_service.py

Sentiment analysis and news ingestion — Section VI of the paper.
Architecture:
  • VADER replaces FinBERT for sentiment scoring (Eq. 17–18);
    can be swapped to transformers pipeline with one import change.
  • RSS feeds from Reuters, CNBC, MarketWatch (NewsAPI optional).
  • Source credibility weighting (w_source ∈ [0,1]).
  • Macro event category classification (rule-based proxy for BERT).
  • Exponential decay smoothing with 3-day half-life.
"""
from __future__ import annotations

import hashlib
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from loguru import logger

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cachetools

from config import settings

# ── VADER analyser (singleton) ────────────────────────────────────────────────
_analyser = SentimentIntensityAnalyzer()

# ── Simple TTL cache (15 min) ─────────────────────────────────────────────────
_news_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=50, ttl=settings.news_cache_ttl_seconds
)

# ─────────────────────────────────────────────────────────────────────────────
# Source credibility weights (Section VI.A)
# ─────────────────────────────────────────────────────────────────────────────

RSS_SOURCES: Dict[str, Tuple[str, float]] = {
    "reuters":     ("https://feeds.reuters.com/reuters/businessNews", 1.0),
    "cnbc":        ("https://search.cnbc.com/rs/search/combinedcombined/rss?q=stock+market", 0.90),
    "marketwatch": ("https://feeds.content.dowjones.io/public/rss/mw_marketpulse", 0.85),
    "seeking_alpha":("https://seekingalpha.com/market_currents.xml", 0.70),
}

# ─────────────────────────────────────────────────────────────────────────────
# Macro Event Categories (Table VI — rule-based proxy)
# ─────────────────────────────────────────────────────────────────────────────

MACRO_KEYWORDS: Dict[str, List[str]] = {
    "Inflation / CPI Release":   ["cpi", "inflation", "consumer price", "pce", "deflation"],
    "Interest Rate Decision":    ["federal reserve", "fed rate", "fomc", "rate hike", "rate cut",
                                  "basis points", "interest rate", "central bank"],
    "Geopolitical Conflict":     ["war", "conflict", "military", "sanction", "ukraine", "nato",
                                  "geopolit", "tension", "missile"],
    "Natural Disaster":          ["earthquake", "hurricane", "flood", "wildfire", "tsunami",
                                  "tornado", "drought"],
    "Earnings Surprise":         ["earnings", "eps", "revenue beat", "revenue miss",
                                  "guidance", "quarterly results", "profit"],
    "Regulatory / Legal Action": ["sec", "ftc", "doj", "antitrust", "lawsuit", "fine",
                                  "penalty", "regulatory", "compliance"],
    "Merger & Acquisition":      ["merger", "acquisition", "takeover", "buyout", "deal",
                                  "m&a", "acquires", "purchased by"],
    "IPO / Delisting":           ["ipo", "initial public offering", "goes public", "delist",
                                  "direct listing", "spac"],
    "CEO / Leadership Change":   ["ceo", "chief executive", "president", "resign", "appoint",
                                  "leadership", "board of directors", "steps down"],
    "Currency / FX Crisis":      ["dollar", "currency", "forex", "exchange rate", "devaluation",
                                  "fx", "yen", "euro", "pound"],
    "Trade Policy":              ["tariff", "trade war", "trade deal", "import", "export ban",
                                  "customs", "wto", "trade policy"],
    "Market Crash / Rally":      ["crash", "rally", "correction", "bear market", "bull market",
                                  "circuit breaker", "volatility spike", "sell-off"],
}


def classify_macro_event(text: str) -> Tuple[Optional[str], float]:
    """
    Rule-based macro event classifier.
    Returns (category, confidence) or (None, 0.0) if no match.
    Replace with BERT fine-tuned classifier for production use.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for category, keywords in MACRO_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in text_lower)
        if hit > 0:
            scores[category] = hit

    if not scores:
        return None, 0.0

    best_cat = max(scores, key=lambda c: scores[c])
    total_kw = len(MACRO_KEYWORDS[best_cat])
    confidence = min(scores[best_cat] / max(total_kw * 0.4, 1), 1.0)
    return best_cat, round(confidence, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Ticker tagging (regex entity linking)
# ─────────────────────────────────────────────────────────────────────────────

_TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')

_COMPANY_ALIASES: Dict[str, str] = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "NVIDIA": "NVDA",
    "Google": "GOOGL", "Alphabet": "GOOGL", "Meta": "META", "Facebook": "META",
    "Amazon": "AMZN", "Tesla": "TSLA", "JPMorgan": "JPM",
    "ExxonMobil": "XOM", "Chevron": "CVX", "Pfizer": "PFE",
    "Johnson & Johnson": "JNJ", "Goldman Sachs": "GS", "Morgan Stanley": "MS",
}


def tag_tickers(text: str, universe: List[str]) -> List[str]:
    """Extract relevant tickers from news text."""
    universe_set = set(universe)
    found = set()

    # Direct symbol extraction
    for match in _TICKER_PATTERN.finditer(text):
        sym = match.group(1)
        if sym in universe_set:
            found.add(sym)

    # Company name aliases
    for name, ticker in _COMPANY_ALIASES.items():
        if name.lower() in text.lower() and ticker in universe_set:
            found.add(ticker)

    return list(found)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment scoring (Eq. 17–18)
# ─────────────────────────────────────────────────────────────────────────────

def score_sentiment(text: str) -> float:
    """
    VADER compound score → rescaled to [-1, +1] (equivalent to p_pos - p_neg).
    To switch to FinBERT: replace this function body with HuggingFace pipeline.
    """
    scores = _analyser.polarity_scores(text)
    return round(float(scores["compound"]), 4)


def aggregate_sentiment(
    articles: List[Dict],
    ticker: str,
    decay_halflife_days: float = 3.0,
) -> float:
    """
    Credibility-weighted aggregate sentiment with exponential time decay.
    Eq. 18 with exponential decay replacing the simple moving average.
    """
    if not articles:
        return 0.0

    now = datetime.now(tz=timezone.utc)
    num, denom = 0.0, 0.0

    for art in articles:
        if ticker not in art.get("tickers", []):
            continue
        w_src  = art.get("credibility", 0.5)
        sent   = art.get("sentiment_score", 0.0)
        pub_ts = art.get("published_at")

        # Time decay
        if pub_ts:
            try:
                dt = pub_ts if isinstance(pub_ts, datetime) else datetime.fromisoformat(pub_ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max((now - dt).total_seconds() / 86400, 0)
                decay = 0.5 ** (age_days / decay_halflife_days)
            except Exception:
                decay = 1.0
        else:
            decay = 1.0

        w = w_src * decay
        num   += w * sent
        denom += w

    return round(num / denom if denom > 1e-8 else 0.0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RSS Feed Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _parse_feed(source_name: str, url: str, credibility: float) -> List[Dict]:
    """Parse a single RSS feed; return list of article dicts."""
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        logger.warning(f"RSS feed {source_name} failed: {exc}")
        return []

    articles = []
    for entry in feed.entries[:20]:
        title = getattr(entry, "title", "")
        summary = getattr(entry, "summary", "")
        text = f"{title}. {summary}"

        pub = getattr(entry, "published_parsed", None)
        if pub:
            try:
                pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
            except Exception:
                pub_dt = datetime.now(tz=timezone.utc)
        else:
            pub_dt = datetime.now(tz=timezone.utc)

        event_id = hashlib.md5(f"{title}{pub_dt}".encode()).hexdigest()[:12]
        macro_cat, macro_conf = classify_macro_event(text)
        sentiment = score_sentiment(text)

        articles.append({
            "event_id":       event_id,
            "headline":       title,
            "source":         source_name,
            "url":            getattr(entry, "link", ""),
            "published_at":   pub_dt,
            "sentiment_score":sentiment,
            "credibility":    credibility,
            "macro_category": macro_cat,
            "macro_confidence":macro_conf,
            "tickers":        tag_tickers(text, settings.coverage_universe),
        })

    return articles


def fetch_recent_news(max_articles: int = 50) -> List[Dict]:
    """
    Fetch and aggregate news from all RSS sources.
    Results are TTL-cached for 15 minutes.
    """
    cache_key = "recent_news"
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    all_articles: List[Dict] = []
    for name, (url, cred) in RSS_SOURCES.items():
        arts = _parse_feed(name, url, cred)
        all_articles.extend(arts)
        logger.debug(f"Fetched {len(arts)} articles from {name}")

    # Sort by publication time (newest first)
    all_articles.sort(key=lambda a: a["published_at"], reverse=True)
    result = all_articles[:max_articles]

    _news_cache[cache_key] = result
    return result


def get_ticker_sentiment(ticker: str) -> float:
    """Quick accessor — returns aggregated sentiment for a single ticker."""
    articles = fetch_recent_news()
    return aggregate_sentiment(articles, ticker)
