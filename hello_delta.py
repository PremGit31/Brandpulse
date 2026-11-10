from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

PATH = "delta_test/hello"

builder = (
    SparkSession.builder.appName("HelloDelta")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

data = [
    ("id1", "AAPL", "Apple shares rise", 0.8),
    ("id2", "TSLA", "Tesla deliveries miss", -0.6),
]
df = spark.createDataFrame(data, ["message_id", "ticker", "headline", "score"])

df.write.format("delta").mode("overwrite").save(PATH)

spark.read.format("delta").load(PATH).show(truncate=False)
spark.stop()