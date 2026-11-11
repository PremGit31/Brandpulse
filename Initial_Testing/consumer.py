"""
consumer.py
-----------
Kafka Consumer for the Brand & Market Pulse Pipeline.

Reads headline messages from the 'financial-news' Kafka topic,
runs each through FinBERT, and prints scored results to console.

Flow:
  Kafka topic: financial-news → consumer.py → FinBERT → console

Run (in a separate terminal from producer.py):
  conda activate brandpulse
  python consumer.py
"""

import json
import torch
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import MetadataEmptyBrokerList
from transformers import BertTokenizer, BertForSequenceClassification

# ── Configuration ─────────────────────────────────────────────────────────────

KAFKA_BROKER    = "localhost:9092"
TOPIC_NAME      = "financial-news"
CONSUMER_GROUP  = "brandpulse-sentiment-group"
MODEL_NAME      = "ProsusAI/finbert"
BATCH_SIZE      = 8     # process N messages at a time before scoring
POLL_TIMEOUT_MS = 3000  # ms to wait for new messages before looping

LABELS = ["positive", "negative", "neutral"]

SENTIMENT_COLORS = {
    "positive": "\033[92m",
    "negative": "\033[91m",
    "neutral":  "\033[93m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"

# ── FinBERT ───────────────────────────────────────────────────────────────────

def load_finbert(device: torch.device):
    print(f"[Consumer] Loading FinBERT on {device}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("[Consumer] FinBERT ready ✓\n")
    return tokenizer, model


def score_batch(batch: list[dict], tokenizer, model, device) -> list[dict]:
    """Run FinBERT over a batch of message dicts. Returns enriched dicts."""
    texts = [msg["headline"] for msg in batch]

    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu()

    results = []
    for i, msg in enumerate(batch):
        prob_vec = probs[i].tolist()
        predicted_idx = prob_vec.index(max(prob_vec))
        results.append({
            **msg,
            "sentiment":       LABELS[predicted_idx],
            "score_positive":  round(prob_vec[0], 4),
            "score_negative":  round(prob_vec[1], 4),
            "score_neutral":   round(prob_vec[2], 4),
            "confidence":      round(max(prob_vec), 4),
            "scored_at":       datetime.utcnow().isoformat(),
        })
    return results


# ── Display ───────────────────────────────────────────────────────────────────

def print_scored(result: dict):
    color = SENTIMENT_COLORS.get(result["sentiment"], "")
    ticker = result.get("ticker") or "—"
    headline = result["headline"][:65] + "…" if len(result["headline"]) > 65 else result["headline"]
    conf = result["confidence"]

    print(
        f"  {color}{result['sentiment'].upper():<10}{RESET} "
        f"{conf:<7.1%} "
        f"[{result['source']:<8}] "
        f"{ticker:<6}  "
        f"{headline}"
    )


# ── Kafka Connection ──────────────────────────────────────────────────────────

def connect_consumer(retries: int = 5, delay: int = 5) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                auto_offset_reset="earliest",   # start from beginning if no committed offset
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=-1,          # block indefinitely — keep listening
                max_poll_records=BATCH_SIZE,
            )
            print(f"[Consumer] Connected to Kafka at {KAFKA_BROKER} ✓")
            print(f"[Consumer] Subscribed to topic: '{TOPIC_NAME}'")
            print(f"[Consumer] Consumer group: '{CONSUMER_GROUP}'\n")
            return consumer
        except MetadataEmptyBrokerList:
            print(f"[Consumer] Kafka not ready (attempt {attempt}/{retries}). Retrying in {delay}s...")
            import time; time.sleep(delay)

    raise ConnectionError(
        "[Consumer] Could not connect to Kafka.\n"
        "Make sure Docker is running: docker-compose up -d"
    )


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_consumer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[Consumer] GPU: {torch.cuda.get_device_name(0)} ✓")

    tokenizer, model = load_finbert(device)
    consumer = connect_consumer()

    print(f"[Consumer] Listening for messages...\n")
    print(f"{'─'*85}")
    print(f"{BOLD}  SENTIMENT   CONF    SOURCE     TICKER  HEADLINE{RESET}")
    print(f"{'─'*85}")

    batch = []
    total_scored = 0

    try:
        for message in consumer:
            msg = message.value
            batch.append(msg)

            # Score when batch is full
            if len(batch) >= BATCH_SIZE:
                results = score_batch(batch, tokenizer, model, device)
                for r in results:
                    print_scored(r)
                total_scored += len(results)
                batch = []

        # Score any remaining messages
        if batch:
            results = score_batch(batch, tokenizer, model, device)
            for r in results:
                print_scored(r)
            total_scored += len(results)

    except KeyboardInterrupt:
        # Score whatever's left in the buffer before exiting
        if batch:
            results = score_batch(batch, tokenizer, model, device)
            for r in results:
                print_scored(r)
            total_scored += len(results)

        print(f"\n{'─'*85}")
        print(f"[Consumer] Shutting down. Total headlines scored this session: {total_scored}")
    finally:
        consumer.close()
        print("[Consumer] Connection closed.")


if __name__ == "__main__":
    run_consumer()