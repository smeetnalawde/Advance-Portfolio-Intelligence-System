from pyspark.sql import SparkSession
import time

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = (
        SparkSession
        .builder
        .appName("LogReg Scaling Up + Hour Baseline")
        .getOrCreate()
    )

    ml_path = "gs://bia678-final/parquet/ml_events_table"
    df = spark.read.parquet(ml_path)

    print("Total events:", df.count())

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # =====================================================
    # FULL MODEL (Scaling Up)
    # =====================================================
    FULL_FEATURES = [
        "is_weekend_int",
        "day_of_week",
        "hour_of_day",
        "is_morning",
        "is_evening",
        "price",
        "log_price"
    ]

    full_pipeline = Pipeline(stages=[
        VectorAssembler(
            inputCols=FULL_FEATURES,
            outputCol="features_raw"
        ),
        MinMaxScaler(
            inputCol="features_raw",
            outputCol="features"
        ),
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=20
        )
    ])

    SAMPLE_RATIOS = [0.25, 0.50, 0.75, 1.00]
    results = []

    print("\n===== Scaling Up Experiment (Full Feature Model) =====")

    for ratio in SAMPLE_RATIOS:
        print(f"\n--- Data Size: {int(ratio * 100)}% ---")

        sampled_df = df.sample(False, ratio, seed=13)
        train, test = sampled_df.randomSplit([0.7, 0.3], seed=13)

        train.cache()

        start = time.time()
        model = full_pipeline.fit(train)
        preds = model.transform(test)
        auc = evaluator.evaluate(preds)
        elapsed = time.time() - start

        train.unpersist()

        print(f"AUC: {auc:.4f}")
        print(f"Time: {elapsed:.2f} sec")

        results.append((ratio, train.count(), auc, elapsed))

    result_df = spark.createDataFrame(
        results,
        ["data_fraction", "train_rows", "auc", "time_seconds"]
    )

    print("\n===== Scaling-Up Summary =====")
    result_df.show(truncate=False)

    # =====================================================
    # HOUR-OF-DAY BASELINE
    # =====================================================
    print("\n===== Hour-of-Day Baseline =====")

    hour_pipeline = Pipeline(stages=[
        VectorAssembler(
            inputCols=["hour_of_day"],
            outputCol="features"
        ),
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=20
        )
    ])

    train, test = df.randomSplit([0.7, 0.3], seed=13)

    start = time.time()
    hour_model = hour_pipeline.fit(train)
    hour_preds = hour_model.transform(test)
    hour_auc = evaluator.evaluate(hour_preds)
    hour_time = time.time() - start

    print(f"AUC: {hour_auc:.4f}")
    print(f"Training + Inference Time: {hour_time:.2f} sec")

    spark.stop()


if __name__ == "__main__":
    main()