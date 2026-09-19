from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.master("local[*]").appName("day4-lab1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
raw = spark.read.parquet(str(ROOT / "data" / "*" / "*.parquet"))
# print("rows:", raw.count())
# raw.printSchema()
# raw.select(F.col("`observation.state`")).show(2)
episodes = (
    raw.groupBy("episode_index")
       .agg(
           F.count("*").alias("num_frames"),
           (F.max("timestamp") - F.min("timestamp")).alias("duration_s"),   # the roadmap's formula, verbatim
           F.first("task_index").alias("task_index"),
       )
       .orderBy("episode_index")
)
episodes.show(5)
import json
info = json.loads((ROOT / "meta/info.json").read_text())
FPS = info["fps"]                                   # 50; also note info["robot_type"] for Lab 3's robots table

meta_eps = spark.read.parquet(str(ROOT / "meta/episodes" / "*" / "*.parquet"))
meta_eps.printSchema()                              # LOOK at this: column names with '/' in them — Spark is fine with those in backticks too

episodes = episodes.withColumn("duration_s", F.col("num_frames") / F.lit(FPS))

mismatch = (episodes.join(meta_eps.select("episode_index", "length"), "episode_index")
                    .filter(F.col("num_frames") != F.col("length")))
n_bad = mismatch.count()
print("episodes:", episodes.count(), " reconcile mismatches:", n_bad)
assert n_bad == 0, "episodes table does not reconcile against meta/episodes — do not trust downstream"