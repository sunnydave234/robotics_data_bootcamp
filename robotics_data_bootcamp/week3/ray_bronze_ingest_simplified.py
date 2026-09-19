#!/usr/bin/env python3
"""
ray_bronze_ingest.py -- SIMPLIFIED core version
=================================================
This is Lab 2's transform, fixed to compute a threshold using the WHOLE
dataset instead of one small pile of rows at a time. Nothing else.
Command-line arguments, reading only some columns, a fancier filter, a
trimmed-down output -- all removed on purpose. They come back later.
"""
from pathlib import Path

import numpy as np
import ray
from ray.data.aggregate import Mean, Std

from tensor_utils import to_2d_float_array
 
DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
ACTION_DIM = 14
STD_THRESHOLD = 3.0
OUT_DIR = "ray_output_day1"


def add_action_magnitude(batch:dict) -> dict:
    action = to_2d_float_array(batch, "action", ACTION_DIM)
    magnitude = np.linalg.norm(action, axis=-1).astype(np.float32)
    return {**batch, "action_magnitude": magnitude}


def main():
    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    # STEP 1: read the data, compute one magnitude number per frame.
    # Identical to Lab 2's add_action_magnitude -- you already know this part.
    data_dir = DEFAULT_ROOT / "data"
    ds = ray.data.read_parquet(data_dir)
    ds = ds.map_batches(add_action_magnitude, batch_format="numpy", batch_size=256)
    ds = ds.materialize()   # run it once, keep the result -- don't redo this below

    # STEP 2: find the REAL average and spread, across ALL 127,500 frames.
    # This is the one thing Lab 2 couldn't do -- it only ever saw ~256 rows
    # at a time. This step deliberately looks at everything before deciding
    # what "normal" means.
    stats = ds.aggregate(Mean(on="action_magnitude"), Std(on="action_magnitude"))
    mean_mag = stats["mean(action_magnitude)"]
    std_mag = stats["std(action_magnitude)"]
    print(f"dataset-wide action_magnitude: mean={mean_mag:.4f}  std={std_mag:.4f}")

    # STEP 3: now that we know the REAL average, flag anything unusually big.
    def flag_outliers(batch: dict) -> dict:
        mag = batch["action_magnitude"]
        outlier = mag > (mean_mag + STD_THRESHOLD * std_mag)
        return {**batch, "action_magnitude_outlier": outlier}
    
    ds = ds.map_batches(flag_outliers, batch_format="numpy", batch_size=256)
    ds = ds.materialize()

    # STEP 4: report the result, and save it.
    n_total = ds.count()
    n_flagged = ds.filter(lambda row: row["action_magnitude_outlier"]).count()
    print(f"frames: {n_total}")
    print(
        f"flagged   (> {STD_THRESHOLD}  std above mean): {n_flagged}    "
        f"({100 * n_flagged / n_total:.2f}%)"
    )

    ds.write_parquet(OUT_DIR)
    print(f"wrote flagged data to: {OUT_DIR}")

    ray.shutdown()

if __name__ == "__main__":
    main()
