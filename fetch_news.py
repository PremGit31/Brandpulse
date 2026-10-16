"""
fetch_news.py
-------------
Fetches financial headlines from two sources:
  1. NewsAPI  — live news articles by keyword
  2. yfinance — recent news for a given ticker

Returns a unified list of dicts:
  { "source": str, "ticker": str | None, "headline": str, "published_at": str }
"""

import os
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from newsapi import NewsApiClient
import yfinance as yf

# Load variables from .env file (looks in current dir and parent dirs)
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise EnvironmentError(
        "NEWSAPI_KEY not found. Make sure it's set in your .env file:\n"
        "  NEWSAPI_KEY=your_key_here"
    )

# Keywords to search on NewsAPI
NEWSAPI_QUERIES = [
    "Apple stock",
    "Tesla earnings",
    "NVIDIA AI",
    "Microsoft revenue",
    "Amazon market",
]

# Tickers to pull news for via yfinance
# NSE-listed Indian tickers need the ".NS" suffix — this is also what
# fetch_price.py in spark_consumer.py uses to route price lookups to
# yfinance instead of Finnhub, so keep that suffix on any Indian symbol
# you add here.
TICKERS = [
    "AAPL", "TSLA", "NVDA", "MSFT", "AMZN",
]

# How many days back to search (NewsAPI free tier: max 30 days)
DAYS_BACK = 1

# Max articles per NewsAPI query
MAX_ARTICLES_PER_QUERY = 5


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_newsapi_headlines() -> list[dict]:
    """Pull headlines from NewsAPI for each query keyword."""
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    results = []
    for query in NEWSAPI_QUERIES:
        try:
            response = client.get_everything(
                q=query,
                from_param=from_date,
                language="en",
                sort_by="publishedAt",
                page_size=MAX_ARTICLES_PER_QUERY,
            )
            for article in response.get("articles", []):
                headline = article.get("title", "").strip()
                if headline and headline != "[Removed]":
                    results.append({
                        "source": "newsapi",
                        "ticker": None,
                        "headline": headline,
                        "published_at": article.get("publishedAt", ""),
                        "query": query,
                    })
        except Exception as e:
            print(f"[NewsAPI] Error fetching '{query}': {e}")

    print(f"[NewsAPI] Fetched {len(results)} headlines across {len(NEWSAPI_QUERIES)} queries.")
    return results


def fetch_yfinance_headlines() -> list[dict]:
    """Recent news per ticker via yf.Search (Ticker.news has been
    returning [] since Yahoo withdrew its endpoint on 2026-10-02)."""
    results = []
    for ticker_symbol in TICKERS:
        try:
            search = yf.Search(ticker_symbol, news_count=MAX_ARTICLES_PER_QUERY * 2)
            items = search.news or []

            # Search is relevance-ranked, so keep only articles Yahoo
            # explicitly tags with this ticker.
            tagged = [i for i in items if ticker_symbol in (i.get("relatedTickers") or [])]

            if not tagged:
                print(f"[yfinance] WARNING: no tagged news for {ticker_symbol}")

            for item in tagged[:MAX_ARTICLES_PER_QUERY]:
                headline = (item.get("title") or "").strip()
                if headline:
                    results.append({
                        "source": "yfinance",
                        "ticker": ticker_symbol,
                        "headline": headline,
                        "published_at": str(item.get("providerPublishTime", "")),
                        "query": ticker_symbol,
                    })
        except Exception as e:
            print(f"[yfinance] Error fetching '{ticker_symbol}': {e}")

    print(f"[yfinance] Fetched {len(results)} headlines across {len(TICKERS)} tickers.")
    return results

# Ticker-specific NewsAPI queries for Indian stocks, reused from
# backfill_historical.py. yf.Search returns no tagged news for .NS symbols.
INDIAN_TICKER_QUERIES = {
    "RELIANCE.NS": '"Reliance Industries"',
    "TCS.NS": '"Tata Consultancy Services" OR "TCS"',
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": '"HDFC Bank"',
}


def fetch_indian_newsapi_headlines() -> list[dict]:
    """Ticker-tagged headlines for Indian stocks via NewsAPI title search."""
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    results = []
    for ticker_symbol, query in INDIAN_TICKER_QUERIES.items():
        try:
            response = client.get_everything(
                qintitle=query,
                from_param=from_date,
                language="en",
                sort_by="publishedAt",
                page_size=MAX_ARTICLES_PER_QUERY,
            )
            for article in response.get("articles", []):
                headline = (article.get("title") or "").strip()
                if headline and headline != "[Removed]":
                    results.append({
                        "source": "newsapi",
                        "ticker": ticker_symbol,
                        "headline": headline,
                        "published_at": article.get("publishedAt", ""),
                        "query": ticker_symbol,
                    })
        except Exception as e:
            print(f"[NewsAPI-IN] Error fetching '{ticker_symbol}': {e}")

    print(f"[NewsAPI-IN] Fetched {len(results)} headlines across {len(INDIAN_TICKER_QUERIES)} Indian tickers.")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def get_all_headlines() -> list[dict]:
    """Fetch from all sources and return combined, deduplicated headline list."""
    all_headlines = []
    all_headlines.extend(fetch_newsapi_headlines())
    all_headlines.extend(fetch_indian_newsapi_headlines())
    all_headlines.extend(fetch_yfinance_headlines())

    # Deduplicate by exact headline text
    seen = set()
    unique = []
    for item in all_headlines:
        if item["headline"] not in seen:
            seen.add(item["headline"])
            unique.append(item)

    print(f"\n[fetch_news] Total unique headlines: {len(unique)}")
    return unique


if __name__ == "__main__":
    headlines = get_all_headlines()
    print("\n── Sample Headlines ─────────────────────────────────────────")
    for h in headlines[:5]:
        print(f"  [{h['source'].upper()}] {h['headline']}")