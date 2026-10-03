"""
backfill_historical.py
-----------------------
One-time (or occasionally re-run) batch job — NOT part of the live
Kafka/Spark streaming pipeline. Pulls the last 30 days of ticker-tagged
headlines from NewsAPI (the max its free tier allows), scores them with
FinBERT, joins against daily closing prices from yfinance, and writes
a daily sentiment-vs-price aggregate per ticker to CSV.

This exists because a single headline's sentiment vs. a single price
tick is noisy and not very meaningful — but a rolling daily average
sentiment compared against that day's price move starts to show an
actual trend. Running this script costs a few minutes and a handful
of API calls, not an always-on service.

Run with:
    conda activate brandpulse
    python backfill_historical.py
"""

import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import torch
import yfinance as yf
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise EnvironmentError("NEWSAPI_KEY not found in .env")

DAYS_BACK = 30  # NewsAPI free tier's hard ceiling — don't raise this
MAX_ARTICLES_PER_TICKER = 100  # NewsAPI free tier's per-request max

# When a ticker has more than MAX_ARTICLES_PER_TICKER matching articles in
# the window, a single request only returns the newest 100 — the rest of
# the month gets silently squeezed out, and gets *more* squeezed out every
# day that passes as new articles keep outranking older ones by recency.
# Splitting the window into buckets and querying each separately gives each
# slice of the month its own 100-article cap instead of all slices
# competing for one global cap. Bounded at 5 to keep worst-case request
# count sane: 9 tickers x up to (1 + 5) requests each = 54 max, still well
# under NewsAPI's 100/day free-tier budget.
MAX_BUCKETS_PER_TICKER = 5

# One NewsAPI query per ticker — this is different from fetch_news.py's
# NEWSAPI_QUERIES, which are intentionally ticker-less for general market
# sentiment. Here we need every headline tied to a ticker so it can be
# joined against that ticker's price.
#
# Queries run against qintitle (title-only match, set below). Most company
# names are unambiguous on their own (NVIDIA, Infosys, "Reliance Industries",
# "HDFC Bank" don't collide with everyday words). "Apple", "Amazon", and
# "Tesla" do — they're also a fruit, a marketplace/river, and a unit/surname
# — so those three require a business-context qualifier in the title too,
# trading a little recall for a lot less noise. ("stock" is deliberately
# avoided as an Amazon qualifier since "back in stock" is itself a common
# false-positive source in deal-listing headlines.)
TICKER_QUERIES = {
    "AAPL": '"Apple" AND (iPhone OR Nasdaq OR shares OR "Tim Cook" OR MacBook OR earnings OR CEO)',
    "TSLA": '"Tesla" AND (Musk OR Nasdaq OR shares OR EV OR Model OR Autopilot OR earnings OR CEO)',
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": '"Amazon" AND (Bezos OR Nasdaq OR shares OR AWS OR e-commerce OR earnings OR CEO)',
    "RELIANCE.NS": '"Reliance Industries"',
    "TCS.NS": '"Tata Consultancy Services" OR "TCS"',
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": '"HDFC Bank"',
}

OUTPUT_PATH = "historical_sentiment_30d.csv"

# ── FinBERT (loaded once, same model as spark_consumer.py) ──────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[backfill] Loading FinBERT on {DEVICE}...")
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)
MODEL.eval()
LABEL_MAP = MODEL.config.id2label

# Numeric sentiment weight — used to compute a confidence-weighted daily
# average. NEUTRAL contributes 0 regardless of confidence, so a day full
# of neutral headlines nets out near zero rather than skewing either way.
SENTIMENT_WEIGHT = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}


def score_sentiment(headlines: list[str]) -> list[tuple[str, float]]:
    """Same approach as spark_consumer.py's score_sentiment — batched
    GPU inference, one forward pass for the whole list."""
    if not headlines:
        return []
    inputs = TOKENIZER(headlines, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    return [
        (LABEL_MAP[pred.item()].upper(), conf.item())
        for pred, conf in zip(predictions, confidences)
    ]


# ── NewsAPI backfill ─────────────────────────────────────────────────────────

def _looks_english(text: str) -> bool:
    """Lightweight heuristic, not a real language detector: FinBERT is
    English-only, and NewsAPI's language='en' filter is based on source
    metadata, not actual content — it lets non-English headlines through
    often enough to matter (Japanese, Gujarati, etc. have shown up in
    testing). Scoring those doesn't error, it just silently produces a
    confident-looking but meaningless sentiment label. A simple ASCII
    ratio catches the large majority of non-Latin-script cases cheaply,
    without pulling in a full language-detection dependency for what's
    ultimately a one-off batch script.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= 0.85


def _date_buckets(start: datetime, end: datetime, n: int) -> list[tuple[str, str]]:
    """Split [start, end] into n equal date-string sub-ranges for NewsAPI's
    from_param/to."""
    span = (end - start) / n
    return [
        (
            (start + i * span).strftime("%Y-%m-%d"),
            (start + (i + 1) * span).strftime("%Y-%m-%d"),
        )
        for i in range(n)
    ]


def _fetch_ticker_articles(client: NewsApiClient, query: str, from_date: str, to_date: str) -> tuple[list[dict], int, int]:
    """Fetches articles for one ticker, bucketing the date range if a
    single request wouldn't cover everything. Returns (articles,
    total_available, requests_used)."""
    response = client.get_everything(
        qintitle=query,
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="publishedAt",
        page_size=MAX_ARTICLES_PER_TICKER,
    )
    articles = response.get("articles", [])
    total_available = response.get("totalResults", len(articles))
    requests_used = 1

    if total_available <= MAX_ARTICLES_PER_TICKER:
        return articles, total_available, requests_used

    n_buckets = min(MAX_BUCKETS_PER_TICKER, math.ceil(total_available / MAX_ARTICLES_PER_TICKER))
    if n_buckets <= 1:
        return articles, total_available, requests_used

    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date, "%Y-%m-%d")
    bucketed_articles = []
    for bucket_from, bucket_to in _date_buckets(start_dt, end_dt, n_buckets):
        bucket_response = client.get_everything(
            qintitle=query,
            from_param=bucket_from,
            to=bucket_to,
            language="en",
            sort_by="publishedAt",
            page_size=MAX_ARTICLES_PER_TICKER,
        )
        bucketed_articles.extend(bucket_response.get("articles", []))
        requests_used += 1

    return bucketed_articles, total_available, requests_used


def fetch_historical_headlines() -> pd.DataFrame:
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    from_date = start_dt.strftime("%Y-%m-%d")
    to_date = end_dt.strftime("%Y-%m-%d")

    rows = []
    total_requests = 0
    for ticker, query in TICKER_QUERIES.items():
        try:
            articles, total_available, requests_used = _fetch_ticker_articles(client, query, from_date, to_date)
            total_requests += requests_used
            non_english_skipped = 0
            for article in articles:
                headline = " ".join(article.get("title", "").split())
                published_at = article.get("publishedAt", "")
                if not headline or headline == "[Removed]" or not published_at:
                    continue
                if not _looks_english(headline):
                    non_english_skipped += 1
                    continue
                rows.append({
                    "ticker": ticker,
                    "headline": headline,
                    "date": pd.to_datetime(published_at).date(),
                })
            status = f"{len(articles)} fetched over {requests_used} request(s)"
            if total_available > len(articles):
                status += f", {total_available} total match — {total_available - len(articles)} still missed"
            else:
                status += " (all available results captured)"
            if non_english_skipped:
                status += f", {non_english_skipped} skipped (non-English despite language=en filter)"
            print(f"[backfill] {ticker}: {status}")
        except Exception as e:
            print(f"[backfill] NewsAPI error for {ticker} ({query}): {e}")

    print(f"[backfill] Total NewsAPI requests used: {total_requests}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # NewsAPI frequently returns the exact same headline more than once —
    # syndication across sources, or the same listing re-surfacing across
    # pages. Same idea as make_message_id() in producer.py: dedupe by exact
    # headline text, keeping the earliest date it was seen under.
    before = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["ticker", "headline"], keep="first")
    removed = before - len(df)
    if removed:
        print(f"[backfill] Removed {removed} duplicate headlines across all tickers")

    return df.reset_index(drop=True)


# ── yfinance historical prices ───────────────────────────────────────────────

def fetch_historical_prices(tickers: list[str]) -> pd.DataFrame:
    """Daily closes for the same window, for every ticker — regardless
    of US/Indian market, since this is historical daily bars, not a
    live quote, and yfinance covers both for free."""
    end = datetime.utcnow()
    start = end - timedelta(days=DAYS_BACK)

    rows = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end)
            for date_idx, row in hist.iterrows():
                rows.append({
                    "ticker": ticker,
                    "date": date_idx.date(),
                    "close": round(row["Close"], 2),
                })
        except Exception as e:
            print(f"[backfill] yfinance history error for {ticker}: {e}")

    return pd.DataFrame(rows)


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(headlines_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if headlines_df.empty:
        print("[backfill] No headlines fetched — nothing to aggregate.")
        return pd.DataFrame()

    sentiments = score_sentiment(headlines_df["headline"].tolist())
    headlines_df["label"] = [s[0] for s in sentiments]
    headlines_df["confidence"] = [s[1] for s in sentiments]
    headlines_df["weighted_score"] = headlines_df["label"].map(SENTIMENT_WEIGHT) * headlines_df["confidence"]

    daily_sentiment = (
        headlines_df.groupby(["ticker", "date"])
        .agg(
            avg_sentiment=("weighted_score", "mean"),
            headline_count=("headline", "count"),
            headlines=("headline", lambda h: " || ".join(h)),
        )
        .reset_index()
    )

    # Sort prices so day-over-day % change can be computed per ticker.
    # Drop any duplicate (ticker, date) rows first — yfinance occasionally
    # returns one, usually from split/dividend adjustment handling — or a
    # merge below would fan out one sentiment row into two identical ones.
    prices_df = prices_df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    prices_df["price_change_pct"] = (
        prices_df.groupby("ticker")["close"].pct_change() * 100
    ).round(2)

    merged = pd.merge(daily_sentiment, prices_df, on=["ticker", "date"], how="left")
    # Non-trading days (weekends/holidays) won't have a price row — carry
    # the most recent known close forward per ticker rather than leaving
    # sentiment days with no price context at all.
    merged = merged.sort_values(["ticker", "date"])
    merged[["close", "price_change_pct"]] = (
        merged.groupby("ticker")[["close", "price_change_pct"]].ffill()
    )

    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[backfill] Pulling last {DAYS_BACK} days of headlines + prices for {len(TICKER_QUERIES)} tickers...\n")

    headlines_df = fetch_historical_headlines()
    prices_df = fetch_historical_prices(list(TICKER_QUERIES.keys()))

    print(f"\n[backfill] {len(headlines_df)} total headlines, {len(prices_df)} price rows fetched.")

    result = aggregate(headlines_df, prices_df)
    if result.empty:
        return

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[backfill] Wrote {len(result)} daily rows to {OUTPUT_PATH}")
    print("\n── Preview (headlines shortened for console width; full text is in the CSV) ──")
    preview = result.copy()
    preview["headlines"] = preview["headlines"].str.slice(0, 60) + "..."
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
