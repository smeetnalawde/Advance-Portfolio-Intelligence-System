from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

def main():
    spark = (
        SparkSession
        .builder
        .appName("Raw Data to Parquet")
        .getOrCreate()
    )

    # set time as UTC
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # GCP path
    raw_data = "gs://bia678-final/data/2019-Oct.csv"
    silver_path = "gs://bia678-final/parquet/events"

    # 1. read raw data
    df = spark.read.csv(
        path=raw_data,
        sep=",",
        header=True,
        inferSchema=False)
    
    # 2. parse timestamp
    df = df.withColumn(
        "event_time",
        F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss z"))

    # 3. basic cleaning & standardization
    valid_event_types = ["view", "cart", "purchase"]
    df_clean = (
        df
        .dropna(subset=["event_time", "event_type", "product_id", "user_id"])
        .withColumn("event_type", F.lower(F.col("event_type")))
        .filter(F.col("event_type").isin(valid_event_types))
        .filter((F.col("price").isNull()) | (F.col("price") >= 0))
    )

    # 4. write to parquet (partitioned by event_type)
    (
        df_clean.write
        .mode("overwrite")
        .partitionBy("event_type")
        .parquet(silver_path)
    )

    spark.stop()

if __name__ == "__main__":
    main()