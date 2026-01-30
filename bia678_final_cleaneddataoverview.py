from pyspark.sql import SparkSession

def main():
    spark = (
        SparkSession
        .builder
        .appName("Inspect Cleaned Events Table")
        .getOrCreate()
    )

    gold_path = "gs://bia678-final/parquet/cleanedevents"

    # read cleaned events
    df = spark.read.parquet(gold_path)

    print("===== Cleaned Events Schema =====")
    df.printSchema()

    print("\n===== Total Rows =====")
    print(df.count())

    print("\n===== Sample Rows (Top 20) =====")
    df.show(20, truncate=False)

    print("\n===== Event Type Distribution =====")
    df.groupBy("event_type").count().show()

    print("\n===== Weekend Distribution =====")
    df.groupBy("is_weekend").count().show()

    print("\n===== Day of Week Distribution =====")
    df.groupBy("day_of_week").count().orderBy("day_of_week").show()

    spark.stop()


if __name__ == "__main__":
    main()