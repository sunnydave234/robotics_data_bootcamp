#!/usr/bin/env python3
"""
debug_challenge_dtypes.py -- Week 3 Day 4 Debugging Challenge
==================================================================
Prints episode_index/num_frames/duration_s/robot as seen by Spark,
Ray Data, and pandas -- for BOTH spark_output/episodes/ (Mini Project
output) and ray_output_day1/ (Day 1's Ray output, action/action_
magnitude/action_magnitude_outlier instead).

Run this AFTER:
  1. spark_tabular_warehouse.py has been run at least once (produces
     spark_output/episodes/, partitioned by robot)
  2. ray_bronze_ingest_simplified.py has been run for real (produces
     ray_output_day1/)

Usage:
    python debug_challenge_dtypes.py \
        --spark-episodes spark_output/episodes \
        --ray-output ray_output_day1
"""
import argparse

import pandas as pd
import ray
from pyspark.sql import SparkSession


def show_spark(spark, path: str, label: str):
    print(f"\n--- Spark printSchema() :: {label} ---")
    spark.read.parquet(path).printSchema()


def show_ray(path: str, label: str):
    print(f"\n--- Ray schema() :: {label} ---")
    print(ray.data.read_parquet(path).schema())


def show_pandas(path: str, label: str):
    print(f"\n--- pandas dtypes :: {label} ---")
    try:
        print(pd.read_parquet(path).dtypes)
    except Exception as e:
        print(f"pandas read_parquet FAILED on this path: {e!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spark-episodes", default="spark_output/episodes")
    p.add_argument("--ray-output", default="ray_output_day1")
    args = p.parse_args()

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    spark = SparkSession.builder.master("local[*]").appName("debug_challenge_dtypes").getOrCreate()

    for path, label in [
        (args.spark_episodes, "episodes (Spark/Mini-Project output, partitioned by robot)"),
        (args.ray_output, "ray_output_day1 (Ray Data output)"),
    ]:
        show_spark(spark, path, label)
        show_ray(path, label)
        show_pandas(path, label)

    spark.stop()


if __name__ == "__main__":
    main()
