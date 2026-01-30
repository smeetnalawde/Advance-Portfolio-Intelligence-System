from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def main():
    spark = (
        SparkSession
        .builder
        .appName("EDA")
        .getOrCreate()
    )

    gold_path = "gs://bia678-final/parquet/cleanedevents"
    df = spark.read.parquet(gold_path)

    print("===== Event Schema =====")
    df.printSchema()

    # EDA 1: Event Type Distribution
    print("===== Total Events =====")
    print("Number of events:", df.count())
    print("===== Event Type Distribution =====")
    (
        df.groupBy("event_type")
        .count() # auto generate a column named 'count'
        .withColumn(
            "percentage",
            F.col("count") / F.sum("count").over(Window.partitionBy()) * 100
        )
        .orderBy(F.desc("count"))
        .show()
    )


    # EDA 2: Weekend vs Weekday
    print("===== Weekend vs Weekday =====")
    (
        df.groupBy("is_weekend")
        .count()
        .withColumn(
        "day_type",
        F.when(F.col("is_weekend"), "Weekend").otherwise("Weekday")
        )
        .withColumn(
            "percentage",
            F.col("count") / F.sum("count").over(Window.partitionBy()) * 100
        )
        .select("day_type", "count", "percentage")
        .orderBy(F.desc("count"))
        .show()
    )


    # EDA 3: Day of Week Distribution
    print("===== Day of Week Distribution =====")
    (
        df.groupBy("day_of_week")
        .count()
        .orderBy("day_of_week")
        .show()
    )


    # EDA 4: Overall Conversion Rate
    print("===== Overall Funnel Conversion =====")

    funnel_counts = (
        df.agg(
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("num_views"),
            F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("num_carts"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("num_purchases")
        )
    )

    funnel_rates = (
        funnel_counts
        .withColumn(
            "view_to_cart_rate",
            F.when(F.col("num_views") > 0,
                F.col("num_carts") / F.col("num_views") * 100)
            .otherwise(0.0)
        )
        .withColumn(
            "cart_to_purchase_rate",
            F.when(F.col("num_carts") > 0,
                F.col("num_purchases") / F.col("num_carts") * 100)
            .otherwise(0.0)
        )
        .withColumn(
            "view_to_purchase_rate",
            F.when(F.col("num_views") > 0,
                F.col("num_purchases") / F.col("num_views") * 100)
            .otherwise(0.0)
        )
    )

    funnel_rates.show(truncate=False)


    # EDA 5: Events Type with Distinct user_id
    (
    df.groupBy("user_id")
        .agg(
            F.count("*").alias("total_events"),
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("num_views"),
            F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("num_carts"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("num_purchases"),
            F.avg("price").alias("avg_price"),
            F.max("price").alias("max_price")
        )
    ).show()


    # EDA 6: User-level Behavior (Tempy Aggregation)
    user_stats = (
        df.groupBy("user_id")
        .agg(
            F.count("*").alias("total_events"),
            F.avg(F.col("is_weekend").cast("int")).alias("weekend_ratio"),
            F.avg(
                F.when(F.col("event_type") == "purchase", 1).otherwise(0)
            ).alias("purchase_rate"),
            F.avg("price").alias("avg_price")
        )
    )

    print("===== User-level Summary =====")
    user_stats.select(
        "total_events",
        "weekend_ratio",
        "purchase_rate",
        "avg_price"
    ).summary().show()

    spark.stop()


if __name__ == "__main__":
    main()
