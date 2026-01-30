from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import time

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = (
        SparkSession
        .builder
        .appName("Spark ML - LogReg Purchase Prediction (Scaling Up + Baseline)")
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


    # Model 1: FULL Logistic Regression
    FULL_FEATURES = [
        "is_weekend_int",
        "day_of_week",
        "hour_of_day",
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
    scaling_results = []

    for ratio in SAMPLE_RATIOS:
        print(f"\n===== Scaling Up: {int(ratio*100)}% Data =====")

        sampled_df = df.sample(False, ratio, seed=13)
        train, test = sampled_df.randomSplit([0.7, 0.3], seed=13)

        start_time = time.time()
        model = full_pipeline.fit(train)
        preds = model.transform(test)
        auc = evaluator.evaluate(preds)
        elapsed = time.time() - start_time

        print(f"AUC: {auc:.4f}")
        print(f"Time: {elapsed:.2f} sec")

        scaling_results.append(
            (ratio, train.count(), auc, elapsed)
        )

    scaling_df = spark.createDataFrame(
        scaling_results,
        ["data_fraction", "train_rows", "auc", "time_seconds"]
    )

    print("\n===== Scaling-Up Summary (Full Model) =====")
    scaling_df.show(truncate=False)


    # Model 2: Hour-of-Day ONLY (Baseline)
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

    start_time = time.time()
    hour_model = hour_pipeline.fit(train)
    hour_preds = hour_model.transform(test)
    hour_auc = evaluator.evaluate(hour_preds)
    hour_time = time.time() - start_time

    print("\n===== HOUR-ONLY BASELINE =====")
    print(f"AUC: {hour_auc:.4f}")
    print(f"Training + Inference Time: {hour_time:.2f} sec")

    spark.stop()


if __name__ == "__main__":
    main()