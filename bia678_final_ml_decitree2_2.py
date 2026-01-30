from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier


def main():
    spark = (
        SparkSession
        .builder
        .appName("Decision Tree Feature Analysis")
        .getOrCreate()
    )

    ml_path = "gs://bia678-final/parquet/ml_events_table2"
    df = spark.read.parquet(ml_path)

    print("Total events:", df.count())

    FULL_FEATURES = [
        "is_weekend_int",
        "day_of_week",
        "hour_of_day",
        "is_morning",
        "is_evening",
        "price",
        "log_price"
    ]

    pipeline = Pipeline(stages=[
        VectorAssembler(
            inputCols=FULL_FEATURES,
            outputCol="features"
        ),
        DecisionTreeClassifier(
            labelCol="label",
            featuresCol="features",
            maxDepth=6,
            seed=13
        )
    ])

    train, _ = df.randomSplit([0.7, 0.3], seed=13)

    model = pipeline.fit(train)
    dt_model = model.stages[-1]


    # feature importance
    print("\n===== Decision Tree Feature Importance =====")

    importance_df = spark.createDataFrame(
        zip(FULL_FEATURES, dt_model.featureImportances),
        ["feature", "importance"]
    )

    importance_df.orderBy(F.desc("importance")).show(truncate=False)


    # purchase rate feature analysis
    print("\n===== Purchase Rate: Weekend =====")
    (
        df.groupBy("is_weekend")
        .agg(
            F.avg("label").alias("purchase_rate"),
            F.count("*").alias("events")
        )
        .show()
    )

    print("\n===== Purchase Rate: Morning =====")
    (
        df.groupBy("is_morning")
        .agg(
            F.avg("label").alias("purchase_rate"),
            F.count("*").alias("events")
        )
        .show()
    )

    print("\n===== Purchase Rate: Evening =====")
    (
        df.groupBy("is_evening")
        .agg(
            F.avg("label").alias("purchase_rate"),
            F.count("*").alias("events")
        )
        .show()
    )

    spark.stop()


if __name__ == "__main__":
    main()