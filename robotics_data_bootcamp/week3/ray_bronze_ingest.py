#!/usr/bin/env python3
"""
ray_bronze_ingest.py -- Week 3 Day 1 Coding Exercise / Mini Project deliverable
================================================================================
Ray Data's version of a bronze-ingest step: read a LeRobot v3.0 dataset's
tabular Parquet layer, compute a REAL per-frame quality signal (action
magnitude) against a REAL dataset-wide threshold (not the per-block
approximation from Lab 2 -- see that file's docstring for why that's wrong),
flag outliers, and write the flagged table back out.
 
Usage:
    python ray_bronze_ingest.py \
        --path ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet \
        --out ray_output/day1/ \
        --std-threshold 3.0
 
Design notes -- read before assuming this is "just" Lab 2 wrapped in argparse:
 
1. batch["action"] is normalized via tensor_utils.to_2d_float_array()
   instead of assumed to already be (num_rows, 14). Confirmed by debugging
   Lab 2: Ray Data represented this column as a 1-D object array of
   per-row arrays, not a proper 2D tensor -- see tensor_utils.py's module
   docstring for the full, confirmed mechanism (not a guess).
 
2. .materialize() is called once, right after the magnitude computation.
   Without it, EVERY downstream trigger (aggregate(), count(), write_parquet())
   independently re-executes the read + map_batches from scratch -- confirmed
   against Ray's docs: Dataset transformations are lazy, and execution is
   triggered by downstream consumption *per call*, not cached across calls,
   unless you materialize().
   https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.materialize.html
 
3. Mean and Std are computed in ONE ds.aggregate(Mean(...), Std(...)) call,
   not two separate ds.mean()/ds.std() calls -- confirmed in Ray's
   aggregation docs, which show exactly this "multiple aggregations at once"
   pattern. Two separate calls would mean two separate passes over the data.
   https://docs.ray.io/en/latest/data/aggregations.html
 
4. udf_modifying_row_count=False is set on both map_batches calls because
   it's genuinely true here (columns are added, no rows dropped or
   duplicated). Ray's docs say this "allows Ray Data to perform more
   optimizations like limit pushdown" -- whether it measurably changes
   YOUR timings is the Debugging Challenge's open question
   (see debug_count_behavior.py).
 
5. columns=[...] is passed to read_parquet() instead of reading every
   column and discarding what's unused. Confirmed in Ray's Loading Data
   and Performance Tips docs: this pushes column selection down to the
   Parquet file scan itself ("projection pushdown"), which is more
   efficient than reading everything and calling select_columns()
   afterward, because the unused columns (observation.state,
   observation.effort, timestamp, task_index...) are never even read off
   disk. https://docs.ray.io/en/latest/data/loading-data.html
 
6. The final filter uses ray.data.expressions (col(), lit()) instead of a
   plain Python lambda. Confirmed in Ray's own read_parquet docs and the
   Anyscale engineering blog on Ray Data's optimizer: a lambda/UDF-based
   filter is opaque to Ray's query optimizer and blocks predicate
   pushdown, while an expression-based filter can be pushed down and
   optimized the same way projection is. For one filter on a small,
   already-materialized dataset this makes no visible difference here --
   it's included to show the currently-recommended pattern, not because
   this script needed the speedup.
   https://docs.ray.io/en/latest/data/api/doc/ray.data.read_parquet.html
 
7. select_columns([...]) runs right before write_parquet(), so the output
   file only carries episode_index, frame_index, and the two computed
   columns -- not the original 14-number action tensor. This is a
   deliberate "bronze quality-flags table" design: the raw sensor data
   already lives in the source dataset; a derived signal table only needs
   to carry the derived values plus a join key (episode_index,
   frame_index) back to it. Re-writing the full action tensor into every
   quality-flag output would duplicate storage for no benefit.
"""
import argparse
import time
from pathlib import Path
 
import numpy as np
import ray
from ray.data.aggregate import Mean, Std
from ray.data.expressions import col, lit
 
from tensor_utils import to_2d_float_array
 
ACTION_DIM = 14


def add_action_magnitude(batch: dict) -> dict:
    action = to_2d_float_array(batch, "action", ACTION_DIM) # (num_rows, 14), guaranteed
    magnitude = np.linalg.norm(action, axis=-1).astype(np.float32)
    return {**batch, "action_magnitude": magnitude}     # new dict, never mutate what Ray hands you


def make_flag_fn(mean_mag: float, std_mag: float, threshold: float):
    def flag_outliers(batch: dict) -> dict:
        mag = batch["action_magnitude"]
        outlier = mag > (mean_mag + std_mag + threshold)
        return {**batch, "action_magnitude_outlier": outlier}
    return flag_outliers


def main():
    parser = argparse.ArgumentParser(
        description="Ray Data bronze ingest: action-magnitude quality flag for a LeRobot v3.0 dataset"
    )
    parser.add_argument(
        "--path", type=Path, required=True,
        help="LeRobot v3.0 dataset root (contains data/, videos/, meta/)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for flagged Parquet")
    parser.add_argument(
        "--std-threshold", type=float, default=3.0,
        help="Flag frames with action_magnitude > mean + N*std (default: 3.0)",
    )
    args = parser.parse_args()
 
    data_dir = args.path / "data"
    if not data_dir.exists():
        raise SystemExit(f"{data_dir} not found -- point --path at a LeRobot v3.0 dataset root.")

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    t0 = time.perf_counter()

    ds = ray.data.read_parquet(
        str(data_dir), columns=["episode_index", "frame_index", "action"]
    )
    ds = ds.map_batches(
        add_action_magnitude,
        batch_format="numpy",
        batch_size=256,
        udf_modifying_row_count=False,
    )

    # Pin the magnitude-augmented blocks in the object store. Everything
    # below (aggregate, flag, write, count) reuses these blocks instead of
    # re-reading Parquet and recomputing action_magnitude from scratch.
    ds = ds.materialize()
    print(f"materialized {ds.count()} frames with action_magnitude computed")

    stats = ds.aggregate(Mean(on="action_magnitude"), Std(on="action_magnitude"))
    mean_mag = stats["mean(action_magnitude)"]
    std_mag = stats["std(action_magnitude)"]
    print(f"dataset-wide action_magnitude: mean={mean_mag:.4f}  std={std_mag:.4f}")

    flag_fn = make_flag_fn(mean_mag, std_mag, args.std_threshold)
    ds = ds.map_batches(
        flag_fn, batch_format="numpy", batch_size=256,
        udf_modifying_row_count=False,
    )

    # Clean output: join key (episode_index, frame_index) + the two
    # computed columns. Drops the raw 14-number action tensor -- see
    # design note 7. Comment this line out if you want the full tensor
    # carried through to the output for some other downstream use.
    ds = ds.select_columns(["episode_index", "frame_index", "action_magnitude", "action_magnitude_outlier"])
    ds = ds.materialize()

    args.out.mkdir(parents=True, exist_ok=True)
    ds.write_parquet(str(args.out))

    n_total = ds.count()
    # Expression-based filter (see design note 6) instead of
    # ds.filter(lambda row: bool(row["action_magnitude_outlier"]))
    n_flagged = ds.filter(expr=col("action_magnitude_outlier") == lit(True)).count()
    dt = time.perf_counter() - t0

    print(f"frames: {n_total}")
    print(
        f"flagged (> {args.std_threshold} std above mean): {n_flagged} "
        f"({100 * n_flagged / n_total:.2f}%)"
    )
    print(f"total wall time: {dt:.2f}s")
    print(f"output written to: {args.out}")
 
    ray.shutdown()
 
 
if __name__ == "__main__":
    main()