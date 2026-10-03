"""
spark_consumer.py — PySpark Structured Streaming consumer for BrandPulse (Phase 3).

Reads from the 'financial-news' Kafka topic, parses JSON into structured
columns, and runs FinBERT sentiment inference on each micro-batch via
foreachBatch. Prints sentiment-scored results to console. Delta Lake
writing will be added inside process_batch() in Phase 4.

Run with:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 spark_consumer.py

Requires:
  - Java 17 or 21 (JAVA_HOME set)
  - HADOOP_HOME set, with winutils.exe on PATH
  - PYSPARK_PYTHON set to your conda env's python.exe
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType

# Windows' default console encoding (cp1252) can't represent many Unicode
# characters that show up in real headlines (currency symbols like ₹, curly
# quotes, em dashes, etc). Without this, a single such character in any
# headline crashes the print() call and kills the entire streaming query.
# Forcing UTF-8 here makes stdout able to handle virtually any character.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Schema of the JSON messages your producer publishes.
#    Adjust field names here if they don't match producer.py's payload —
#    a mismatch shows up as NULL columns, not an error.
# ---------------------------------------------------------------------------
news_schema = StructType([
    StructField("headline", StringType()),
    StructField("ticker", StringType()),
    StructField("source", StringType()),
    StructField("published_at", StringType()),
])

# ---------------------------------------------------------------------------
# 2. Load FinBERT ONCE at module import time — not inside process_batch().
#    If it were loaded inside process_batch(), Spark would reload the model
#    from disk/cache on every micro-batch, which would be very slow. Loading
#    it here means it loads once when the job starts, and is reused for
#    every batch for the rest of the job's lifetime.
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[spark_consumer] Loading FinBERT on {DEVICE}...")
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)
MODEL.eval()

# FinBERT's label order — printed at startup so you can double check it
# against your existing consumer.py in case the order differs.
LABEL_MAP = MODEL.config.id2label
print(f"[spark_consumer] FinBERT ready. Label map: {LABEL_MAP}")


def score_sentiment(headlines):
    """Run FinBERT on a list of headlines in one batched GPU forward pass.
    Returns a list of (label, confidence_pct) tuples, one per headline."""
    if not headlines:
        return []
    inputs = TOKENIZER(
        headlines, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    return [
        (LABEL_MAP[pred.item()].upper(), round(conf.item() * 100, 1))
        for pred, conf in zip(predictions, confidences)
    ]


def process_batch(batch_df, batch_id):
    """Called by Spark once per micro-batch. batch_df is a normal Spark
    DataFrame containing just the new rows since the last batch. We convert
    it to pandas, run FinBERT on all headlines in the batch at once, and
    print the results. This is also where the Delta Lake write will go
    in Phase 4."""
    pdf = batch_df.toPandas()
    if pdf.empty:
        print(f"[Batch {batch_id}] No rows.")
        return

    sentiments = score_sentiment(pdf["headline"].tolist())
    pdf["sentiment"] = [s[0] for s in sentiments]
    pdf["confidence"] = [s[1] for s in sentiments]

    print(f"\n[Batch {batch_id}] {len(pdf)} headlines scored")
    print("-" * 100)
    for _, row in pdf.iterrows():
        try:
            ticker = row["ticker"] if row["ticker"] else "-"
            headline = row["headline"][:70] if row["headline"] else ""
            print(
                f"  {row['sentiment']:<9} {row['confidence']:>5.1f}%  "
                f"[{str(row['source']):<9}] {str(ticker):<6}  {headline}"
            )
        except Exception as e:
            # Defense in depth: one bad row should never take down the
            # whole streaming query. Log it and keep going.
            print(f"  [!] Could not print one row: {e}")


def main():
    spark = (
        SparkSession.builder
        .appName("BrandPulseSparkConsumer")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "financial-news")
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_stream = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), news_schema).alias("data"))
        .select("data.*")
    )

    query = (
        parsed_stream.writeStream
        .foreachBatch(process_batch)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()