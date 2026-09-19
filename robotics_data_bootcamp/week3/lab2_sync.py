# lab2_sync.py
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.master("local[*]").appName("day4-lab2").getOrCreate()
FPS = 50
EXPECTED_DT = 1.0 / FPS

toy = spark.createDataFrame(
    [(0, 0, 0.00), (0, 1, 0.02), (0, 2, 0.04), (0, 3, 0.08), (0, 4, 0.10),   # ep 0: frame missing between 2 and 3
     (1, 0, 0.00), (1, 1, 0.02), (1, 2, 0.02), (1, 3, 0.04)],                # ep 1: duplicate timestamp at frame 2
    ["episode_index", "frame_index", "timestamp"],
)

w = Window.partitionBy("episode_index").orderBy("frame_index")
deltas = toy.withColumn("dt", F.col("timestamp") - F.lag("timestamp").over(w))
deltas.show()
def build_sync(frames_df, fps: int):
    expected = 1.0 / fps
    w = Window.partitionBy("episode_index").orderBy("frame_index")
    d = frames_df.withColumn("dt", F.col("timestamp") - F.lag("timestamp").over(w))
    return (
        d.groupBy("episode_index").agg(
            F.count("*").alias("num_frames"),
            F.max("dt").alias("max_gap_s"),
            F.sum((F.col("dt") > F.lit(1.5 * expected)).cast("int")).alias("n_dropped"),
            F.sum((F.abs(F.col("dt")) < F.lit(1e-6)).cast("int")).alias("n_duplicates"),
            # drift: how far the wall span is from what num_frames at this fps implies, in ms
            (((F.max("timestamp") - F.min("timestamp")) - (F.count("*") - 1) / F.lit(fps)) * 1000)
                .alias("drift_ms"),
        )
        .withColumn("sync_error_flag", (F.col("n_dropped") > 0) | (F.col("n_duplicates") > 0)
                                        | (F.abs(F.col("drift_ms")) > F.lit(0.5 * expected * 1000)))
    )

build_sync(toy, FPS).orderBy("episode_index").show()
from pathlib import Path
ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
frames = spark.read.parquet(str(ROOT / "data" / "*" / "*.parquet")).select("episode_index", "frame_index", "timestamp")
sync = build_sync(frames, FPS)
sync.filter("sync_error_flag").show()
sync.select(F.min("max_gap_s"), F.max("max_gap_s"), F.min("drift_ms"), F.max("drift_ms")).show()