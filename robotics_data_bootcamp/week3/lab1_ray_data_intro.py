#!/usr/bin/env python3
"""
Week 3, Day 1 - Lab 1: Ray Data Fundamentals
=============================================
Read lerobot/aloha_mobile_cabinet's tabular Parquet layer through Ray Data
and find out -- empirically, on your own machine -- exactly where the
"free metadata" line sits. Predict each timing before you run it.

Verified against Ray Data docs (docs.ray.io, Ray 2.56.0, checked Aug 2026):
  "For Datasets which only read Parquet files (created with read_parquet()),
   this [count()] method reads the file metadata to efficiently count the
   number of rows without reading in the entire data."
  "If this dataset consists of more than a read ... this operation will
   trigger execution of the lazy transformations."
  https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.count.html

This directly corrects the original assumption that count() would behave
like a full scan the way `ds.hf_dataset` scalar reads did NOT in Week 1-2 --
it doesn't, but not for the reason you'd guess either. Read Lab 1's write-up
in the cheatsheet before assuming you understand the result.

Prereqs:
    pip install "ray[data,train]" pyarrow pandas --break-system-packages
    python -c "import ray; ray.init(); print(ray.cluster_resources())"
    # Week 1's LeRobotDataset("lerobot/aloha_mobile_cabinet") must have run
    # at least once, to populate the local cache this script reads from.
"""
import argparse
import time
from pathlib import Path

import ray

DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
EXPECTED_FRAMES = 127_500  # lerobot/aloha_mobile_cabinet, confirmed meta/info.json, Week 2 Day 4


def timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[{dt:7.3f}s] {label}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Ray Data fundamentals lab")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="LeRobot v3.0 local dataset root (contains data/, videos/, meta/)",
    )
    args = parser.parse_args()

    data_dir = args.root / "data"
    if not data_dir.exists():
        raise SystemExit(
            f"{data_dir} not found.\n"
            f"Run LeRobotDataset('lerobot/aloha_mobile_cabinet') once first "
            f"(Week 1 Lab 2) so the local cache is populated, then re-run this script."
        )

    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    # --- Step 1: build the lazy read plan. Nothing executes yet -- Ray's own
    # docs: "Dataset transformations are lazy, with execution ... triggered
    # by downstream consumption." ---
    ds = ray.data.read_parquet(str(data_dir))
    print("Schema:")
    print(ds.schema())
    print()

    # --- Step 2: PREDICT before you run each line below. Write your guess
    # down first. ---

    # Q1: bare read_parquet().count() -- full scan, or metadata-only?
    n1 = timed("count() on bare read_parquet()", ds.count)
    print(f"  -> {n1} rows\n")

    # Q2: run it again on the SAME dataset object. Still fast?
    n2 = timed("count() on bare read_parquet(), again", ds.count)
    print(f"  -> {n2} rows\n")

    assert n1 == n2 == EXPECTED_FRAMES, (
        f"Expected {EXPECTED_FRAMES} frames both times, got {n1} then {n2}. "
        "If this fails, your local cache may be stale or partial -- re-check "
        "Week 1's LeRobotDataset download."
    )

    # Q3: now chain ANY map_batches -- even a total no-op -- before count().
    # Does Ray still answer instantly from Parquet footer metadata, or does
    # it now have to run the actual pipeline?
    noop_ds = ds.map_batches(lambda batch: batch, batch_format="numpy", batch_size=512)
    n3 = timed("count() AFTER a no-op map_batches()", noop_ds.count)
    print(f"  -> {n3} rows (same number as before -- very different cost to get it)\n")

    print("=" * 72)
    print("Compare n1/n2 above against Week 1's LeRobotDatasetMetadata.total_frames:")
    print("that number costs ZERO file opens -- it's one field already sitting")
    print("in meta/info.json, which you fetched once. n1/n2 are NOT that free:")
    print("Ray still opened every Parquet file under data/ to read its footer.")
    print("On this dataset (chunks_size=1000, 85 episodes) that's probably a")
    print("single file, so the cost is invisible here. At DROID scale")
    print("(thousands of chunk files) it would not be -- that's Day 2's")
    print("file-group partitioning problem, previewed.")
    print("=" * 72)

    ray.shutdown()


if __name__ == "__main__":
    main()