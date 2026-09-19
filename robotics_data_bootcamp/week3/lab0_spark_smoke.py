# lab0_spark_smoke.py
import sys, pyspark
from pyspark.sql import SparkSession

print("python:", sys.version.split()[0], " pyspark:", pyspark.__version__)
spark = (SparkSession.builder
         .master("local[*]")          # every core on the M4 Max; no cluster
         .appName("day4-smoke")
         .getOrCreate())
print("spark:", spark.version, " cores:", spark.sparkContext.defaultParallelism)
print(spark.range(10).selectExpr("sum(id) AS s").collect())
spark.stop()