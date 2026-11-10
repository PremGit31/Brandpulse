"""
spark_consumer.py — PySpark Structured Streaming consumer for BrandPulse (Phase 3).

Reads from the 'financial-news' Kafka topic, parses JSON into structured
columns, runs FinBERT sentiment inference on each micro-batch via
foreachBatch, and enriches ticker-level rows with current price data
(see fetch_price.py — Finnhub for US tickers, yfinance for Indian
".NS"/".BO" tickers, both cached with a TTL). Prints sentiment+price
results to console and stores each batch idempotently in Delta Lake.

Run with:
    python -m pip install --upgrade pyspark==4.1.3
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.3,io.delta:delta-spark_4.1_2.13:4.3.0 spark_consumer.py

Requires:
  - Java 17 or 21 (JAVA_HOME set)
  - HADOOP_HOME set, with winutils.exe on PATH
  - PYSPARK_PYTHON set to your conda env's python.exe
"""

import sys
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from fetch_price import get_prices_for_tickers

DELTA_PATH = r"C:\Users\Prem\Brandpulse\delta\sentiment"
CHECKPOINT_PATH = r"C:\Users\Prem\Brandpulse\checkpoints\sentiment"

# Windows can write UTF-8 from Python while the attached console still
# decodes output using a legacy code page, producing mojibake. Set the
# console's output code page as well as Python's stream encoding.
if sys.platform == "win32" and sys.stdout.isatty():
    import ctypes

    if not ctypes.windll.kernel32.SetConsoleOutputCP(65001):
        raise ctypes.WinError()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    StructField("message_id", StringType()),
    StructField("ingested_at", StringType()),
])

delta_schema = StructType([
    StructField("message_id", StringType()),
    StructField("ticker", StringType()),
    StructField("headline", StringType()),
    StructField("source", StringType()),
    StructField("published_at", StringType()),
    StructField("ingested_at", StringType()),
    StructField("sentiment", StringType()),
    StructField("confidence", DoubleType()),
    StructField("price", DoubleType()),
    StructField("price_change_pct", DoubleType()),
    StructField("price_status", StringType()),
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
    it to pandas, run FinBERT on all headlines in the batch at once,
    enrich prices, store results in Delta Lake, and print them."""
    pdf = batch_df.toPandas()
    if pdf.empty:
        print(f"[Batch {batch_id}] No rows.")
        return

    print(
        f"[Batch {batch_id}] message_id nulls={pdf['message_id'].isna().sum()}, "
        f"ingested_at nulls={pdf['ingested_at'].isna().sum()}"
    )

    sentiments = score_sentiment(pdf["headline"].tolist())
    pdf["sentiment"] = [s[0] for s in sentiments]
    pdf["confidence"] = [s[1] for s in sentiments]

    # ── Price enrichment ────────────────────────────────────────────────
    # Ticker-tagged rows from yfinance and ticker-specific NewsAPI searches
    # carry a ticker; broad NewsAPI market headlines are untagged and skipped.
    # get_prices_for_tickers() routes each ticker to Finnhub (US) or yfinance
    # (.NS/.BO Indian tickers) and handles caching internally.
    tickers_in_batch = pdf["ticker"].dropna().unique().tolist()
    prices = get_prices_for_tickers(tickers_in_batch)

    def _price_field(ticker, field):
        info = prices.get(ticker)
        return info.get(field) if info else None

    def _currency_symbol(ticker):
        # Indian tickers carry the exchange suffix everywhere in this
        # pipeline (fetch_news.py appends it, fetch_price.py routes on
        # it) — reuse that same signal here rather than tracking
        # currency as a separate field.
        return "₹" if isinstance(ticker, str) and ticker.upper().endswith((".NS", ".BO")) else "$"

    pdf["price"] = pdf["ticker"].apply(lambda t: _price_field(t, "price"))
    pdf["price_change_pct"] = pdf["ticker"].apply(lambda t: _price_field(t, "change_pct"))
    pdf["price_status"] = pdf["ticker"].apply(lambda t: _price_field(t, "status"))

    delta_pdf = pdf[delta_schema.fieldNames()].astype(object)
    delta_pdf = delta_pdf.where(pd.notna(delta_pdf), None)
    delta_df = batch_df.sparkSession.createDataFrame(delta_pdf, schema=delta_schema)

    print(f"\n[Batch {batch_id}] {len(pdf)} headlines scored")
    print("-" * 100)
    for _, row in pdf.iterrows():
        try:
            ticker = row["ticker"] if pd.notna(row["ticker"]) else "-"
            headline = row["headline"][:70] if row["headline"] else ""
            if pd.notna(row["price"]):
                currency = _currency_symbol(row["ticker"])
                price_change_pct = (
                    row["price_change_pct"]
                    if pd.notna(row["price_change_pct"])
                    else "N/A"
                )
                price_str = f"{currency}{row['price']:.2f} ({price_change_pct}%, {row['price_status']})"
            else:
                price_str = "-"
            print(
                f"  {row['sentiment']:<9} {row['confidence']:>5.1f}%  "
                f"[{str(row['source']):<9}] {str(ticker):<12}  {price_str:<28}  {headline}"
            )
        except Exception as e:
            # Defense in depth: one bad row should never take down the
            # whole streaming query. Log it and keep going.
            print(f"  [!] Could not print one row: {e}")

    spark = batch_df.sparkSession
    batch_rows = len(pdf)
    unique_delta_df = delta_df.dropDuplicates(["message_id"])

    if DeltaTable.isDeltaTable(spark, DELTA_PATH):
        delta_table = DeltaTable.forPath(spark, DELTA_PATH)
        merge_started = time.perf_counter()
        delta_table.alias("t").merge(
            unique_delta_df.alias("s"),
            "t.message_id = s.message_id",
        ).whenNotMatchedInsertAll().execute()
        operation_seconds = time.perf_counter() - merge_started
        operation_metrics = (
            delta_table.history(1).select("operationMetrics").first()["operationMetrics"]
        )
        inserted_rows = int(operation_metrics["numTargetRowsInserted"])
        operation_name = "merge"
    else:
        unique_rows = unique_delta_df.count()
        write_started = time.perf_counter()
        unique_delta_df.write.format("delta").mode("append").save(DELTA_PATH)
        operation_seconds = time.perf_counter() - write_started
        inserted_rows = unique_rows
        operation_name = "initial write"

    print(
        f"[Batch {batch_id}] rows in batch={batch_rows}, "
        f"rows inserted={inserted_rows}, "
        f"{operation_name} time={operation_seconds:.2f}s"
    )


def main():
    spark = (
        SparkSession.builder
        .appName("BrandPulseSparkConsumer")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
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
        .option("checkpointLocation", CHECKPOINT_PATH)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()