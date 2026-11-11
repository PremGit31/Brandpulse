"""
spark_consumer.py — Minimal PySpark Structured Streaming consumer for BrandPulse.

Purpose: validate that Spark can read from the 'financial-news' Kafka topic and
parse each message into structured columns, before FinBERT inference is added.
No FinBERT, no Delta Lake yet — this just proves the plumbing works.

Run with (adjust the Kafka package version to match your Spark version if needed):

    spark-submit ^
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 ^
      spark_consumer.py

On Windows PowerShell, use `` ` `` instead of `^` for line continuation, or put it
all on one line.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType

# ---------------------------------------------------------------------------
# 1. Define the schema of the JSON messages your producer publishes.
#
#    IMPORTANT: These field names are a best guess based on your consumer.py
#    output (headline, ticker, source). Open producer.py and check exactly
#    what keys the JSON payload uses (e.g. look at the dict passed into
#    producer.send(...)) and adjust the field names below to match exactly.
#    If a field name is wrong, that column will come back as NULL — it won't
#    error, so this is the most common thing to double-check first.
# ---------------------------------------------------------------------------
news_schema = StructType([
    StructField("headline", StringType()),
    StructField("ticker", StringType()),
    StructField("source", StringType()),
    StructField("published_at", StringType()),
])

def main():
    spark = (
        SparkSession.builder
        .appName("BrandPulseSparkConsumer")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # quiet down Spark's default noisy logs

    # -----------------------------------------------------------------------
    # 2. Read the raw stream from Kafka.
    #    Each row here has columns like key, value, topic, partition, offset —
    #    'value' is the raw JSON bytes your producer sent.
    # -----------------------------------------------------------------------
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "financial-news")
        .option("startingOffsets", "earliest")  # read from the beginning of the topic
        .load()
    )

    # -----------------------------------------------------------------------
    # 3. Parse the JSON 'value' column into structured columns using the schema.
    # -----------------------------------------------------------------------
    parsed_stream = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), news_schema).alias("data"))
        .select("data.*")  # flatten data.headline -> headline, etc.
    )

    # -----------------------------------------------------------------------
    # 4. Write the parsed stream to the console so you can see it working.
    #    outputMode("append") = only show new rows as they arrive.
    # -----------------------------------------------------------------------
    query = (
        parsed_stream.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()