from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import time

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (
    SparkSession
    .builder
    .appName("Spark ML - GBT Purchase Prediction with Scaling Up")
    .getOrCreate()
)

ml_path = "gs://bia678-final/parquet/ml_events_table"
df = spark.read.parquet(ml_path)

print("===== ML Input Schema =====")
df.printSchema()

print("===== Total Events =====")
print(df.count())

FEATURE_COLS = [
    "is_weekend_int",
    "day_of_week",
    "hour_of_day",
    "price",
    "log_price"
]

df = (
    df
    .withColumn("is_weekend_int", F.col("is_weekend_int").cast("int"))
    .withColumn("day_of_week", F.col("day_of_week").cast("int"))
    .withColumn("hour_of_day", F.col("hour_of_day").cast("int"))
    .withColumn("price", F.col("price").cast("double"))
    .withColumn("log_price", F.col("log_price").cast("double"))
)

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features"
)

gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxIter=30,        # number of trees
    maxDepth=5,        # tree depth
    stepSize=0.1,      # learning rate
    subsamplingRate=0.8,
    seed=13
)

gbt_pipeline = Pipeline(
    stages=[
        assembler,
        gbt
    ]
)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

SAMPLE_RATIOS = [0.25, 0.50, 0.75, 1.00]
results_summary = []

for ratio in SAMPLE_RATIOS:
    print("\n==============================================")
    print(f"Running GBT Scaling-Up Experiment: {int(ratio * 100)}% Data")
    print("==============================================")

    sampled_df = df.sample(
        withReplacement=False,
        fraction=ratio,
        seed=13
    )

    train, test = sampled_df.randomSplit([0.7, 0.3], seed=13)
    train.cache()

    print("Train size:", train.count())
    print("Test size:", test.count())

    start_time = time.time()

    model = gbt_pipeline.fit(train)
    predictions = model.transform(test)

    auc = evaluator.evaluate(predictions)
    elapsed_time = time.time() - start_time

    print(f"AUC: {auc}")
    print(f"Training + Inference Time: {elapsed_time:.2f} seconds")

    predictions.groupBy("label").pivot("prediction").count().show()

    results_summary.append(
        (
            ratio,
            train.count(),
            auc,
            elapsed_time
        )
    )

summary_df = spark.createDataFrame(
    results_summary,
    ["data_fraction", "train_rows", "auc", "time_seconds"]
)

print("===== GBT Scaling-Up Summary =====")
summary_df.show(truncate=False)

spark.stop()