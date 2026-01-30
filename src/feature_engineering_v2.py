from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = (
    SparkSession
    .builder
    .appName("Build_ML_Table")
    .getOrCreate()
)

gold_path = "gs://bia678-final/parquet/cleanedevents2"
events = spark.read.parquet(gold_path)

print("===== Cleaned Table Schema =====")
events.printSchema()

print("===== Total Events =====")
print(events.count())


# labeling
events_labeled = (
    events
    .withColumn(
        "label",
        F.when(F.col("event_type") == "purchase", 1).otherwise(0)
    )
)


# ML Table (feature selection only)
ml_table = (
    events_labeled
    .select(
        # identifiers
        F.col("user_id"),
        F.col("product_id"),

        # label
        F.col("label"),

        # temporal features (from cleaning layer)
        F.col("is_weekend"),
        F.col("is_weekend").cast("int").alias("is_weekend_int"),
        F.col("day_of_week"),
        F.col("hour_of_day"),
        F.col("is_morning"),
        F.col("is_evening"),

        # price features
        F.col("price").cast("double").alias("price"),
        F.when(F.col("price").isNotNull(), F.log1p(F.col("price")))
        .otherwise(0.0)
        .alias("log_price")
    )
)

print("===== ML Table Schema =====")
ml_table.printSchema()

print("===== Label Distribution =====")
ml_table.groupBy("label").count().show()

print("===== Weekend vs Purchase =====")
(
    ml_table
    .groupBy("is_weekend", "label")
    .count()
    .show()
)

print("===== Morning/Evening vs Purchase =====")
(
    ml_table
    .groupBy("is_morning", "is_evening", "label")
    .count()
    .show()
)

ml_output_path = "gs://bia678-final/parquet/ml_events_table2"

(
    ml_table
    .write
    .mode("overwrite")
    .parquet(ml_output_path)
)

print("===== ML Table Written Successfully =====")
print(ml_output_path)

spark.stop()