"""
fetch_price.py
---------------
Price lookups for BrandPulse, routed by market:

  - US tickers (no suffix, e.g. "AAPL", "TSLA")   -> Finnhub /quote
  - Indian tickers ("RELIANCE.NS", "TCS.BO")      -> yfinance

Both are free and keyless-or-cheap-to-key. Neither is hammered per
headline — results are cached with a TTL, and each source is paced
independently so we stay well inside Finnhub's 60/min limit and don't
trip yfinance's anti-scraping defenses.

Requires FINNHUB_API_KEY in .env (free signup at finnhub.io).
yfinance needs no key at all.
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/quote"

CACHE_PATH = Path(__file__).parent / "price_cache.json"

TTL_MINUTES_DEFAULT = 30

# Minimum gap between consecutive *live* calls to each source. Finnhub's
# free tier allows 60/min — 1.5s keeps us well clear even with jitter.
# yfinance has no official limit, but it's an unofficial scraper, so we
# pace it deliberately conservatively to avoid getting IP-blocked.
SECONDS_BETWEEN_FINNHUB_CALLS = 1.5
SECONDS_BETWEEN_YFINANCE_CALLS = 3.0

_last_finnhub_call = 0.0
_last_yfinance_call = 0.0


# ── Cache persistence ────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ── Market routing ───────────────────────────────────────────────────────────

def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith((".NS", ".BO"))


# ── Source-specific fetchers ─────────────────────────────────────────────────

def _call_finnhub(ticker: str) -> dict | None:
    global _last_finnhub_call
    if not FINNHUB_KEY:
        print("[fetch_price] FINNHUB_API_KEY not set — skipping US lookup")
        return None

    wait = SECONDS_BETWEEN_FINNHUB_CALLS - (time.time() - _last_finnhub_call)
    if wait > 0:
        time.sleep(wait)

    try:
        resp = requests.get(
            FINNHUB_URL,
            params={"symbol": ticker, "token": FINNHUB_KEY},
            timeout=10,
        )
        _last_finnhub_call = time.time()
        data = resp.json()

        # Finnhub returns all-zero fields (c=0) for an invalid/unknown
        # symbol rather than an error — treat that as "no data".
        if not data or data.get("c") in (None, 0):
            print(f"[fetch_price] No Finnhub quote for {ticker}: {data}")
            return None

        return {
            "price": data["c"],
            "change": data["d"],
            "change_pct": round(data["dp"], 2) if data.get("dp") is not None else None,
            "source": "finnhub",
        }
    except (requests.RequestException, ValueError, KeyError) as e:
        _last_finnhub_call = time.time()
        print(f"[fetch_price] Finnhub call failed for {ticker}: {e}")
        return None


def _call_yfinance(ticker: str) -> dict | None:
    global _last_yfinance_call
    wait = SECONDS_BETWEEN_YFINANCE_CALLS - (time.time() - _last_yfinance_call)
    if wait > 0:
        time.sleep(wait)

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.fast_info
        _last_yfinance_call = time.time()

        price = info.get("last_price")
        prev_close = info.get("previous_close")

        # fast_info's real-time quote fields aren't reliably populated for
        # every exchange — NSE/BSE tickers especially, and more so outside
        # market hours (weekends, holidays). When that happens, fall back
        # to daily close history, which is always populated once a stock
        # has traded at all, regardless of exchange or session timing.
        if price is None or prev_close in (None, 0):
            hist = yf_ticker.history(period="5d")
            if len(hist) >= 2:
                price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2])
            elif len(hist) == 1:
                price = float(hist["Close"].iloc[-1])
                prev_close = price

        if price is None or prev_close in (None, 0):
            print(f"[fetch_price] No yfinance quote for {ticker}")
            return None

        change = price - prev_close
        change_pct = round((change / prev_close) * 100, 2)

        return {
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": change_pct,
            "source": "yfinance",
        }
    except Exception as e:
        # yfinance raises a variety of exception types depending on
        # what Yahoo's endpoint does that day — catch broadly so one
        # bad ticker can't take down the whole batch.
        _last_yfinance_call = time.time()
        print(f"[fetch_price] yfinance call failed for {ticker}: {e}")
        return None


def _fetch_live_price(ticker: str) -> dict | None:
    return _call_yfinance(ticker) if _is_indian_ticker(ticker) else _call_finnhub(ticker)


# ── Public API ───────────────────────────────────────────────────────────────

def get_price(ticker: str, ttl_minutes: int = TTL_MINUTES_DEFAULT) -> dict | None:
    """Returns {price, change, change_pct, source, status, fetched_at}
    for a ticker, or None if there's no cached data and the live call
    also failed.

    status is 'fresh' (within TTL), 'live' (just fetched), or 'stale'
    (older than TTL but the live call failed, so we're reusing it)."""
    cache = _load_cache()
    entry = cache.get(ticker)

    if entry:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if datetime.utcnow() - fetched_at < timedelta(minutes=ttl_minutes):
            return {**entry, "status": "fresh"}

    fresh = _fetch_live_price(ticker)
    if fresh:
        fresh["fetched_at"] = datetime.utcnow().isoformat()
        cache[ticker] = fresh
        _save_cache(cache)
        return {**fresh, "status": "live"}

    if entry:
        return {**entry, "status": "stale"}
    return None


def get_prices_for_tickers(tickers: list[str], ttl_minutes: int = TTL_MINUTES_DEFAULT) -> dict:
    """Batch version for use inside process_batch(). Dedupes tickers;
    pacing against each source happens automatically inside get_price()
    since cache hits never touch the network."""
    return {
        ticker: get_price(ticker, ttl_minutes)
        for ticker in sorted(set(t for t in tickers if t))
    }