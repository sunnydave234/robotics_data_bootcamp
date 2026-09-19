#!/usr/bin/env python3
"""
coding_exercise_episode_quality.py -- Week 3 Day 4 Coding Exercise
======================================================================
Answers the roadmap's Checkpoint Question with real numbers instead of
a hypothetical: Day 1's Ray Data pipeline produced a per-FRAME quality
signal (action_magnitude_outlier, written to ray_output_day1/). Lab 2's
Spark `sync` table is a per-EPISODE quality signal. This script IS the
seam between them: it rolls Ray's frame-grain flags up to episode
grain, joins them against Spark's own episode-grain tables, and prints
the numbers Day 1 never reported.

Run this AFTER:
  1. ray_bronze_ingest_simplified.py has been run for real (produces
     ray_output_day1/ in the current directory)

Usage:
    python coding_exercise_episode_quality.py \
        --root ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet \
        --ray-output ray_output_day1
"""
import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def bt(name: str) -> str:
    """Backtick a LeRobot column name -- dotted keys need this in Spark."""
    return f"`{name}`"


def build_episodes(frames, fps: int):
    return (
        frames.groupBy("episode_index")
        .agg(
            F.count("*").alias("num_frames"),
            F.first("task_index").alias("task_index"),
        )
        .withColumn("duration_s", F.col("num_frames") / F.lit(fps))
    )


def build_sync(frames, fps: int):
    expected = 1.0 / fps
    w = Window.partitionBy("episode_index").orderBy("frame_index")
    d = frames.withColumn("dt", F.col("timestamp") - F.lag("timestamp").over(w))
    return (
        d.groupBy("episode_index")
        .agg(
            F.max("dt").alias("max_gap_s"),
            F.sum((F.col("dt") > F.lit(1.5 * expected)).cast("int")).alias("n_dropped"),
            F.sum((F.abs(F.col("dt")) < F.lit(1e-6)).cast("int")).alias("n_duplicates"),
            (
                ((F.max("timestamp") - F.min("timestamp")) - (F.count("*") - 1) / F.lit(fps))
                * 1000
            ).alias("drift_ms"),
        )
        .withColumn(
            "sync_error_flag",
            (F.col("n_dropped") > 0)
            | (F.col("n_duplicates") > 0)
            | (F.abs(F.col("drift_ms")) > F.lit(0.5 * expected * 1000)),
        )
    )


def rollup_frame_flags(spark, ray_out_dir: str):
    """Ray produced one row per FRAME. Quality decisions are made per
    EPISODE. This is the grain change -- it has to happen somewhere,
    and it's cheap tabular work once the data is Parquet."""
    frames = spark.read.parquet(ray_out_dir).select(
        "episode_index", "action_magnitude", "action_magnitude_outlier"
    )
    return frames.groupBy("episode_index").agg(
        F.sum(F.col("action_magnitude_outlier").cast("int")).alias("n_outlier_frames"),
        F.mean(F.col("action_magnitude_outlier").cast("int")).alias("outlier_frame_rate"),
        F.max("action_magnitude").alias("max_action_magnitude"),
    )


def normalize_keys(df):
    """The seam. Join keys get an explicit type HERE and nowhere else."""
    return df.withColumn("episode_index", F.col("episode_index").cast("long"))


def build_episode_quality(episodes, sync, frame_flags):
    q = (
        episodes.select("episode_index", "num_frames", "duration_s", "task_index")
        .join(
            sync.select(
                "episode_index", "n_dropped", "n_duplicates", "drift_ms", "sync_error_flag"
            ),
            "episode_index",
            "left",
        )
        .join(frame_flags, "episode_index", "left")
    )
    return q.withColumn(
        "any_flag",
        F.col("sync_error_flag") | (F.coalesce(F.col("n_outlier_frames"), F.lit(0)) > 0),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                    help="LeRobot dataset root, e.g. ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet")
    p.add_argument("--ray-output", required=True,
                    help="Directory ray_bronze_ingest_simplified.py wrote to, e.g. ray_output_day1")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("coding_exercise_episode_quality")
        .getOrCreate()
    )

    frames = spark.read.parquet(str(root / "data" / "*" / "*.parquet")).select(
        "episode_index", "frame_index", "timestamp", "task_index"
    )

    episodes = normalize_keys(build_episodes(frames, fps))
    sync = normalize_keys(build_sync(frames, fps))
    frame_flags = normalize_keys(rollup_frame_flags(spark, args.ray_output))

    quality = build_episode_quality(episodes, sync, frame_flags).cache()

    n_episodes = quality.count()
    n_sync_flagged = quality.filter("sync_error_flag").count()
    n_outlier_episodes = quality.filter(F.coalesce(F.col("n_outlier_frames"), F.lit(0)) > 0).count()
    n_any = quality.filter("any_flag").count()
    max_rate = quality.agg(F.max("outlier_frame_rate")).first()[0]

    print(f"episodes total:                     {n_episodes}")
    print(f"episodes with sync_error_flag:      {n_sync_flagged}")
    print(f"episodes with n_outlier_frames > 0: {n_outlier_episodes}")
    print(f"max outlier_frame_rate:             {max_rate}")
    print(f"episodes with ANY flag (unified):   {n_any}")

    quality.orderBy("episode_index").show(10)
    quality.write.mode("overwrite").parquet("spark_output/episode_quality")

    spark.stop()


if __name__ == "__main__":
    main()