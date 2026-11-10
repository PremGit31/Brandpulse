from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip, DeltaTable
import time

PATH = "delta_test/hello"

builder = (
    SparkSession.builder.appName("MergeTest")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "4")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# id1 already exists in the table (a duplicate). id4 is new.
incoming = spark.createDataFrame(
    [
        ("id1", "AAPL", "Apple shares rise", 0.8),
        ("id4", "MSFT", "Microsoft cloud growth", 0.7),  # new, inserted
    ],
    ["message_id", "ticker", "headline", "score"],
)

target = DeltaTable.forPath(spark, PATH)

t0 = time.time()
(
    target.alias("t")
    .merge(incoming.alias("s"), "t.message_id = s.message_id")
    .whenNotMatchedInsertAll()
    .execute()
)
print(f"MERGE took {time.time() - t0:.1f}s")

print("=== TABLE AFTER MERGE ===")
spark.read.format("delta").load(PATH).orderBy("message_id").show(truncate=False)

print("=== HISTORY ===")
target.history().select("version", "operation").show(truncate=False)

spark.stop()