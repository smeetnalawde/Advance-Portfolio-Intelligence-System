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
    gold_path = "gs://bia678-final/parquet/cleanedevents"

    df = spark.read.parquet(silver_path)

    df = (
        df
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("day_of_week", F.dayofweek("event_time"))  # 1=Sun ... 7=Sat
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("event_time").isin([1, 7]), 1).otherwise(0)
            .cast("boolean")
        )
        .select(
            "event_time",
            "event_type",
            "product_id",
            "price",
            "user_id",
            "day_of_week",
            "is_weekend"
        )
    )

    # write to parquet
    (
        df.write
        .mode("overwrite")
        .parquet(gold_path)
    )

    spark.stop()


if __name__ == "__main__":
    main()