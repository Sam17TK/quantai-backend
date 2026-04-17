import hashlib
import socket
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from loguru import logger

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cachetools

from config import settings

# ── 1. NETWORK SAFETY SETTINGS ──────────────────────────────────────────────
# Prevent Render from hanging indefinitely on a slow RSS feed
socket.setdefaulttimeout(10) 

_analyser = SentimentIntensityAnalyzer()
_news_cache = cachetools.TTLCache(maxsize=50, ttl=settings.news_cache_ttl_seconds)

# ── 2. UPDATED RSS SOURCES ──────────────────────────────────────────────────
# Replaced deprecated Reuters feed with working alternatives
RSS_SOURCES: Dict[str, Tuple[str, float]] = {
    "cnbc":        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", 1.0),
    "marketwatch": ("http://feeds.marketwatch.com/marketwatch/marketpulse/", 0.90),
    "yahoo_fin":   ("https://finance.yahoo.com/news/rssindex", 0.85),
    "seeking_alpha":("https://seekingalpha.com/market_currents.xml", 0.70),
}

# (Keep MACRO_KEYWORDS and _COMPANY_ALIASES as you have them...)

# ── 3. OPTIMIZED FEED PARSER ────────────────────────────────────────────────
def _parse_feed(source_name: str, url: str, credibility: float) -> List[Dict]:
    """Parse RSS with timeout protection and memory-efficient limit."""
    try:
        # User-Agent is often required to prevent 403 Forbidden errors on cloud IPs
        feed = feedparser.parse(url, agent='Mozilla/5.0 (QuantAI-Bot/1.0)')
        
        # Check if feed is empty or failed
        if not feed.entries:
            logger.warning(f"No entries found for {source_name}")
            return []
            
    except Exception as exc:
        logger.warning(f"RSS feed {source_name} failed: {exc}")
        return []

    articles = []
    # Limit to 10 articles per source to stay under 512MB RAM
    for entry in feed.entries[:10]:
        title = getattr(entry, "title", "")
        summary = getattr(entry, "summary", "")
        # Remove HTML tags from summary to prevent VADER noise
        clean_summary = re.sub('<[^<]+?>', '', summary)
        text = f"{title}. {clean_summary}"

        pub = getattr(entry, "published_parsed", None)
        pub_dt = datetime(*pub[:6], tzinfo=timezone.utc) if pub else datetime.now(tz=timezone.utc)

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

# ── 4. AGGREGATE LOGIC ──────────────────────────────────────────────────────
def fetch_recent_news(max_articles: int = 30) -> List[Dict]:
    """Fetch news with a lower max_articles for Free Tier stability."""
    cache_key = "recent_news"
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    all_articles: List[Dict] = []
    for name, (url, cred) in RSS_SOURCES.items():
        arts = _parse_feed(name, url, cred)
        all_articles.extend(arts)
        logger.debug(f"Fetched {len(arts)} articles from {name}")

    all_articles.sort(key=lambda a: a["published_at"], reverse=True)
    result = all_articles[:max_articles]

    _news_cache[cache_key] = result
    return result
