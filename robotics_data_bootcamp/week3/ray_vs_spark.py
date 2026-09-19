#!/usr/bin/env python3
"""
Week 3 Day 1 - Mini Project: Ray Data vs. Spark, Same Transform, Timed
=========================================================================
Run the IDENTICAL action-magnitude computation through Ray Data
(ray_bronze_ingest.py's core logic) and a Spark equivalent, and time both
on your own machine. At aloha_mobile_cabinet's scale (127,500 rows) don't
expect Ray to "win" -- that's not the point. The point is having a real,
honest number instead of trusting either tool's reputation.

One known asymmetry worth predicting BEFORE you run this: the Spark path
below uses a plain Python row-at-a-time UDF (the simplest possible Spark
equivalent), not a vectorized pandas_udf. Ray's map_batches is vectorized
numpy by construction. Predict which one that asymmetry favors, then check.

Both engines now read only the "action" column (columns=[...] for Ray,
.select() for Spark) -- confirmed via Ray's Loading Data / Performance
Tips docs that read_parquet(columns=[...]) pushes column selection down
to the file scan itself. Spark's own Catalyst optimizer does the
equivalent column pruning when you .select() right after .read.parquet()
without touching the other columns first. Without this, one engine could
be doing more I/O than the other for reasons that have nothing to do with
the actual computation being timed.

Requires: pip install pyspark (already in Week 3's Day 0 setup)
"""
import time
from pathlib import Path

import numpy as np

from tensor_utils import to_2d_float_array

DATA_DIR = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet/data"
ACTION_DIM = 14


def run_ray() -> float:
    import ray

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    t0 = time.perf_counter()

    def add_magnitude(batch):
        action = to_2d_float_array(batch, "action", ACTION_DIM)  # see tensor_utils.py
        magnitude = np.linalg.norm(action, axis=-1).astype(np.float32)
        return {**batch, "action_magnitude": magnitude}  # new dict, not mutated

    ds = ray.data.read_parquet(str(DATA_DIR), columns=["action"])
    ds = ds.map_batches(
        add_magnitude, batch_format="numpy", batch_size=256, udf_modifying_row_count=False
    )
    n = ds.count()
    dt = time.perf_counter() - t0
    ray.shutdown()
    print(f"Ray Data:  {n} rows in {dt:.2f}s")
    return dt


def run_spark() -> float:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import FloatType

    spark = SparkSession.builder.appName("day1-timing-comparison").getOrCreate()
    t0 = time.perf_counter()

    @udf(returnType=FloatType())
    def magnitude(action):
        # Row-wise UDF -- `action` here is one row's plain 14-element
        # list/array, never a batched/object-dtype column. This is why the
        # Ray path needed tensor_utils.to_2d_float_array() and this one
        # never did: Spark's scalar UDF never presents a whole batch at
        # once, so there's no axis/shape ambiguity to get wrong.
        return float(np.linalg.norm(action))

    df = spark.read.parquet(str(DATA_DIR)).select("action")
    df = df.withColumn("action_magnitude", magnitude(col("action")))
    n = df.count()
    dt = time.perf_counter() - t0
    spark.stop()
    print(f"Spark:     {n} rows in {dt:.2f}s")
    return dt


if __name__ == "__main__":
    ray_time = run_ray()
    spark_time = run_spark()
    print()
    print(f"Ray Data: {ray_time:.2f}s   Spark: {spark_time:.2f}s")
    print("Report the real ratio on your machine -- don't round it toward")
    print("whichever story you expected going in.")