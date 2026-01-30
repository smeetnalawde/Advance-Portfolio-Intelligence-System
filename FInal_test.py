from pyspark.sql import SparkSession
import sys

# 1. spark session
spark = SparkSession.builder.appName("bia678-etl-test").getOrCreate()

# 2. read input path from args
input_path = sys.argv[1]

print(f"Reading data from: {input_path}")

# 3. read CSV from GCS
df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

# 4. basic check
print("Schema:")
df.printSchema()

print("Row count:")
print(df.count())

print("Sample rows:")
df.show(5, truncate=False)

spark.stop()