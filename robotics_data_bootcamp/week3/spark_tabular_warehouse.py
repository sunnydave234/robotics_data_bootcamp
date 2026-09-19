#!/usr/bin/env python3
"""
spark_tabular_warehouse.py -- Week 3 Day 4 Mini Project
===========================================================
Pure-tabular LeRobot v3.0 metadata warehouse. No pixels, no GPU, ever.
Everything that needs a decoded frame lives in the Ray Data pipeline
(partition_utils.py, ray_bronze_ingest_simplified.py); this tool
consumes that pipeline's OUTPUT as a Parquet input, never its video.

Builds up to four tables under --out:
  episodes         -- per-episode frame count, duration, task
  sync             -- per-episode dropped/duplicate/drift detector
  robots           -- one-row dimension table from meta/info.json
  episode_quality  -- (only if --ray-output given) sync + Ray's
                        per-frame outlier flags, rolled up and joined

Refuses to write anything if `episodes` doesn't reconcile against
LeRobot's own meta/episodes index (exit code 2) -- same CI-gate
pattern as Week 1's week1_report.py.

Usage:
    python spark_tabular_warehouse.py \
        --root ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet \
        --out spark_output \
        [--ray-output ray_output_day1] \
        [--no-partition]
"""
import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

TABLES = ("episodes", "robots", "sync", "episode_quality")


def bt(name: str) -> str:
    """Backtick a LeRobot column name -- every dotted key needs this in Spark."""
    return f"`{name}`"


def read_frames(spark, root: Path, columns):
    cols = [bt(c) if "." in c else c for c in columns]
    return spark.read.parquet(str(root / "data" / "*" / "*.parquet")).select(*cols)


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


def build_robots(spark, info: dict):
    return spark.createDataFrame(
        [(info["robot_type"], info["codebase_version"], info["fps"], "unknown")],
        ["robot", "codebase_version", "fps", "firmware"],
    )


def rollup_frame_flags(spark, ray_out_dir: str):
    frames = spark.read.parquet(ray_out_dir).select(
        "episode_index", "action_magnitude", "action_magnitude_outlier"
    )
    return frames.groupBy("episode_index").agg(
        F.sum(F.col("action_magnitude_outlier").cast("int")).alias("n_outlier_frames"),
        F.mean(F.col("action_magnitude_outlier").cast("int")).alias("outlier_frame_rate"),
        F.max("action_magnitude").alias("max_action_magnitude"),
    )


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


def normalize_keys(df):
    """The seam. Join keys get an explicit type HERE and nowhere else."""
    out = df.withColumn("episode_index", F.col("episode_index").cast("long"))
    if "robot" in df.columns:
        out = out.withColumn("robot", F.col("robot").cast("string"))
    return out


def reconcile(episodes, meta_eps) -> int:
    """Return the number of episodes whose frame count disagrees with
    LeRobot's own index. 0 means the warehouse can be trusted."""
    j = episodes.join(meta_eps.select("episode_index", "length"), "episode_index", "full")
    return j.filter(
        (F.col("num_frames") != F.col("length"))
        | F.col("num_frames").isNull()
        | F.col("length").isNull()
    ).count()


def write(df, out: Path, name: str, partition: bool):
    w = df.write.mode("overwrite")
    if partition and "robot" in df.columns:
        w = w.partitionBy("robot")
    w.parquet(str(out / name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="spark_output")
    p.add_argument("--ray-output", default=None,
                    help="ray_bronze_ingest_simplified.py's output dir -- builds episode_quality if given")
    p.add_argument("--no-partition", action="store_true")
    args = p.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out)
    info = json.loads((root / "meta/info.json").read_text())
    fps, robot = info["fps"], info["robot_type"]
    partition = not args.no_partition

    spark = SparkSession.builder.master("local[*]").appName("spark_tabular_warehouse").getOrCreate()

    frames = read_frames(spark, root, ["episode_index", "frame_index", "timestamp", "task_index"])
    meta_eps = spark.read.parquet(str(root / "meta" / "episodes" / "*" / "*.parquet"))

    episodes = normalize_keys(build_episodes(frames, fps).withColumn("robot", F.lit(robot)))

    n_bad = reconcile(episodes, meta_eps)
    print(f"[reconcile] episodes vs meta/episodes mismatches: {n_bad}")
    if n_bad:
        print(
            "[reconcile] FAILED -- refusing to write a warehouse that disagrees "
            "with the dataset's own index",
            file=sys.stderr,
        )
        sys.exit(2)

    sync = normalize_keys(build_sync(frames, fps).withColumn("robot", F.lit(robot)))
    robots = build_robots(spark, info)

    write(episodes, out, "episodes", partition)
    print(f"[episodes] rows: {episodes.count()}")

    write(sync, out, "sync", partition)
    print(f"[sync] rows: {sync.count()}  flagged: {sync.filter('sync_error_flag').count()}")

    write(robots, out, "robots", partition=False)  # one row -- partitioning it is Lab 3's trap
    print(f"[robots] rows: {robots.count()}")

    if args.ray_output:
        frame_flags = normalize_keys(rollup_frame_flags(spark, args.ray_output))
        quality = build_episode_quality(episodes, sync, frame_flags)
        quality = normalize_keys(quality).withColumn("robot", F.lit(robot))
        write(quality, out, "episode_quality", partition)
        print(f"[episode_quality] rows: {quality.count()}  any_flag: {quality.filter('any_flag').count()}")
    else:
        print("[episode_quality] skipped -- pass --ray-output to build this table")

    spark.stop()


if __name__ == "__main__":
    main()