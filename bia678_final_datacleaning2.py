from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def main():
    spark = (
        SparkSession
        .builder
        .appName("Build Cleaned Data")
        .getOrCreate()
    )

    silver_path = "gs://bia678-final/parquet/events"
    gold_path = "gs://bia678-final/parquet/cleanedevents2"

    df = spark.read.parquet(silver_path)

    df = (
        df
        # basic casting
        .withColumn("price", F.col("price").cast("double"))

        # temporal features
        .withColumn("day_of_week", F.dayofweek("event_time"))  # 1=Sun ... 7=Sat
        .withColumn("hour_of_day", F.hour("event_time"))

        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("event_time").isin([1, 7]), 1).otherwise(0)
            .cast("boolean")
        )

        # NEW: time buckets
        .withColumn(
            "is_morning",
            F.when((F.col("hour_of_day") >= 6) & (F.col("hour_of_day") < 12), 1).otherwise(0)
        )
        .withColumn(
            "is_evening",
            F.when((F.col("hour_of_day") >= 16) & (F.col("hour_of_day") < 22), 1).otherwise(0)
        )

        # select final columns
        .select(
            "event_time",
            "event_type",
            "product_id",
            "price",
            "user_id",
            "day_of_week",
            "hour_of_day",
            "is_weekend",
            "is_morning",
            "is_evening"
        )
    )

    (
        df.write
        .mode("overwrite")
        .parquet(gold_path)
    )

    spark.stop()


if __name__ == "__main__":
    main()