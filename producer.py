"""
producer.py
-----------
Kafka Producer for the Brand & Market Pulse Pipeline.

Polls NewsAPI + yfinance on a schedule and publishes each
headline as a JSON message to the 'financial-news' Kafka topic.

Flow:
  fetch_news.py → producer.py → Kafka topic: financial-news

Run:
  conda activate brandpulse
  python producer.py
"""

import json
import time
import hashlib
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import MetadataEmptyBrokerList
from fetch_news import get_all_headlines

# ── Configuration ─────────────────────────────────────────────────────────────

KAFKA_BROKER    = "localhost:9092"
TOPIC_NAME      = "financial-news"
POLL_INTERVAL   = 60   # seconds between fetch cycles (15 minutes)
                        # reduce to 60 during testing

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_message_id(headline: str) -> str:
    """Stable unique ID for a headline — used to avoid reprocessing duplicates."""
    return hashlib.md5(headline.encode()).hexdigest()


def connect_producer(retries: int = 5, delay: int = 5) -> KafkaProducer:
    """
    Connect to Kafka with retry logic.
    Kafka can take ~10s to be ready after docker-compose up.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",           # wait for broker acknowledgement
                retries=3,
                linger_ms=10,         # small batch window for efficiency
            )
            print(f"[Producer] Connected to Kafka at {KAFKA_BROKER} ✓")
            return producer
        except MetadataEmptyBrokerList:
            print(f"[Producer] Kafka not ready yet (attempt {attempt}/{retries}). Retrying in {delay}s...")
            time.sleep(delay)

    raise ConnectionError(
        f"[Producer] Could not connect to Kafka after {retries} attempts.\n"
        "Make sure Docker is running: docker-compose up -d"
    )


def on_send_success(record_metadata):
    print(
        f"  [✓] topic={record_metadata.topic} "
        f"partition={record_metadata.partition} "
        f"offset={record_metadata.offset}"
    )


def on_send_error(exc):
    print(f"  [✗] Failed to send message: {exc}")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_producer():
    producer = connect_producer()
    seen_ids = set()   # track published headlines within this session
    cycle = 0

    print(f"[Producer] Starting publish loop. Poll interval: {POLL_INTERVAL}s\n")

    try:
        while True:
            cycle += 1
            timestamp = datetime.utcnow().isoformat()
            print(f"\n[Producer] ── Cycle {cycle} at {timestamp} ──────────────────")

            # Fetch headlines from all sources
            headlines = get_all_headlines()
            new_count = 0

            for item in headlines:
                msg_id = make_message_id(item["headline"])

                # Skip headlines already published this session
                if msg_id in seen_ids:
                    continue

                # Enrich message with pipeline metadata
                message = {
                    **item,
                    "message_id":   msg_id,
                    "ingested_at":  timestamp,
                    "pipeline":     "brandpulse-phase2",
                }

                # Publish — keyed by ticker (or "general") for Kafka partitioning
                key = item["ticker"] or "general"
                producer.send(TOPIC_NAME, key=key, value=message) \
                         .add_callback(on_send_success) \
                         .add_errback(on_send_error)

                seen_ids.add(msg_id)
                new_count += 1

            producer.flush()   # ensure all buffered messages are sent
            print(f"[Producer] Published {new_count} new headlines (skipped {len(headlines) - new_count} duplicates)")
            print(f"[Producer] Sleeping {POLL_INTERVAL}s until next cycle...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Producer] Shutting down gracefully...")
    finally:
        producer.close()
        print("[Producer] Connection closed.")


if __name__ == "__main__":
    run_producer()