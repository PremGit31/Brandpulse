"""
phase1_pipeline.py
------------------
Phase 1 of the Brand & Market Pulse Pipeline.

Flow:
  fetch_news (NewsAPI + yfinance)
      → FinBERT sentiment analysis (GPU-accelerated via RTX 4050)
          → pretty-print results to console

No Kafka, no Delta Lake — just validating the end-to-end logic.

Run from the brandpulse conda environment:
  conda activate brandpulse
  python phase1_pipeline.py
"""

import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fetch_news import get_all_headlines

# ── FinBERT Setup ─────────────────────────────────────────────────────────────

MODEL_NAME = "ProsusAI/finbert"  # uses your local HuggingFace cache automatically

# Labels as defined by FinBERT
LABELS = ["positive", "negative", "neutral"]


def load_finbert(device: torch.device):
    """Load tokenizer and model. Uses local cache — no download needed."""
    print(f"[FinBERT] Loading model on {device}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"[FinBERT] Model ready.\n")
    return tokenizer, model


def run_sentiment(headlines: list[dict], tokenizer, model, device: torch.device) -> list[dict]:
    """
    Run FinBERT over each headline.
    Returns headline dicts enriched with sentiment fields.
    """
    texts = [h["headline"] for h in headlines]
    results = []

    # Batch inference (all at once — fine for Phase 1 scale)
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

    for i, headline_dict in enumerate(headlines):
        prob_vec = probs[i].tolist()
        predicted_idx = prob_vec.index(max(prob_vec))
        enriched = {
            **headline_dict,
            "sentiment": LABELS[predicted_idx],
            "score_positive": round(prob_vec[0], 4),
            "score_negative": round(prob_vec[1], 4),
            "score_neutral":  round(prob_vec[2], 4),
            "confidence":     round(max(prob_vec), 4),
        }
        results.append(enriched)

    return results


# ── Display ───────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "positive": "\033[92m",  # green
    "negative": "\033[91m",  # red
    "neutral":  "\033[93m",  # yellow
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def print_results(results: list[dict]):
    """Print sentiment results in a readable table to the console."""
    print(f"\n{'─'*90}")
    print(f"{BOLD}{'#':<4} {'SENTIMENT':<12} {'CONF':<8} {'SRC':<10} {'TICKER':<8} HEADLINE{RESET}")
    print(f"{'─'*90}")

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for i, r in enumerate(results, 1):
        color = SENTIMENT_COLORS.get(r["sentiment"], "")
        ticker_display = r["ticker"] or "—"
        headline_short = r["headline"][:60] + "…" if len(r["headline"]) > 60 else r["headline"]

        print(
            f"{i:<4} "
            f"{color}{r['sentiment'].upper():<12}{RESET} "
            f"{r['confidence']:<8.2%} "
            f"{r['source']:<10} "
            f"{ticker_display:<8} "
            f"{headline_short}"
        )
        sentiment_counts[r["sentiment"]] += 1

    print(f"{'─'*90}")

    # Summary
    total = len(results)
    print(f"\n{BOLD}── Sentiment Summary {'─'*30}{RESET}")
    for label, count in sentiment_counts.items():
        color = SENTIMENT_COLORS[label]
        bar = "█" * count
        pct = count / total * 100 if total else 0
        print(f"  {color}{label.capitalize():<10}{RESET}  {bar:<30}  {count:>3} ({pct:.1f}%)")
    print(f"  {'Total':<10}  {'':30}  {total:>3}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Brand & Market Pulse Pipeline — Phase 1")
    print(f"{'='*60}\n")

    # Device selection — use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU detected: {gpu_name} ✓")
    else:
        print("[Device] No GPU found — running on CPU (slower but works fine for Phase 1)")

    # Step 1: Fetch headlines
    print("\n[Step 1] Fetching headlines...\n")
    headlines = get_all_headlines()

    if not headlines:
        print("[ERROR] No headlines fetched. Check your NEWSAPI_KEY and internet connection.")
        return

    # Step 2: Load FinBERT
    print("\n[Step 2] Loading FinBERT...")
    tokenizer, model = load_finbert(device)

    # Step 3: Run sentiment
    print(f"[Step 3] Running sentiment analysis on {len(headlines)} headlines...")
    t0 = time.time()
    results = run_sentiment(headlines, tokenizer, model, device)
    elapsed = time.time() - t0
    print(f"[Step 3] Done in {elapsed:.2f}s  ({elapsed/len(results)*1000:.1f}ms per headline)")

    # Step 4: Display
    print("\n[Step 4] Results:\n")
    print_results(results)

    print(f"\n{'='*60}")
    print("  Phase 1 complete. Pipeline validated end-to-end ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()