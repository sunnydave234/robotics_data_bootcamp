#!/usr/bin/env python3
"""
Week 3, Day 1 - Lab 2: A Real map_batches Transform
====================================================
Computes a genuine per-frame quality signal -- action magnitude --
vectorized across a Ray Data batch, in the spirit of Week 1's
action_following_error, but running through Ray's execution model instead
of a Python loop over ds.hf_dataset.
 
Grounded in your own project knowledge: aloha_mobile_cabinet's `action` is
float32[14], an ABSOLUTE joint-space target (not a delta, not Cartesian --
Week 1 confirmed this from the dataset card). Both arms, 6 joints + gripper
each: left_waist ... left_gripper, right_waist ... right_gripper.
 
CONFIRMED ROOT CAUSE (found via debug prints, then reproduced standalone
with plain numpy -- see tensor_utils.py's module docstring for the full
mechanism): Ray's docs say fixed-shape tensor columns come back as regular
(num_rows, dim) ndarrays. For THIS dataset's `action` column, they don't --
Ray hands back a 1-D array of length num_rows, dtype=object, where each
element is itself a 14-length array. `np.linalg.norm(action, axis=-1)` on
that shape silently reduces over the wrong axis (num_rows, not 14) instead
of raising an error, producing a 14-length result that means nothing. Two
earlier fix attempts (a LimitOperator theory, an in-place-mutation theory)
were both wrong -- they were reasonable hypotheses given the traceback, but
neither one was the actual mechanism, and the identical error after each
"fix" is what proved that. The shape normalization in tensor_utils.py is
the fix that actually addresses the confirmed mechanism.
"""
from pathlib import Path

import numpy as np
import ray

from tensor_utils import to_2d_float_array

DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
ACTION_DIM = 14
FLAG_STD = 3.0  # flag frames > N std devs above THIS BATCH's own mean -- see
                 # the docstring on flag_outliers_per_block for why that's wrong.


def add_action_magnitude(batch: dict) -> dict:
    action = to_2d_float_array(batch, "action", ACTION_DIM)  # (num_rows, 14), guaranteed
    magnitude = np.linalg.norm(action, axis=-1).astype(np.float32)
    return {**batch, "action_magnitude": magnitude}  # new dict, never mutate what Ray hands you


def flag_outliers_per_block(batch: dict) -> dict:
    """
    DELIBERATELY NAIVE: flags relative to THIS BATCH's mean/std, not the
    dataset's. Ray Data batch boundaries don't align with anything
    semantically meaningful (not episode boundaries, not a representative
    sample of the whole distribution) -- so a per-block z-score is closer
    to noise than signal. A frame could be flagged in one batch and not
    flagged if the batch boundaries had fallen differently, for identical
    data. Kept here on purpose so you see the wrong version before
    ray_bronze_ingest.py fixes it with a real dataset-wide ds.aggregate()
    pass computed once and threaded through a second map_batches call.
    """
    mag = batch["action_magnitude"]
    mu, sigma = float(mag.mean()), float(mag.std())
    outlier = mag > (mu + FLAG_STD * sigma)
    return {**batch, "action_magnitude_outlier_PER_BLOCK_NAIVE": outlier}


def main():
    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    data_dir = DEFAULT_ROOT / "data"
    # columns=[...] pushes column selection down to the Parquet file scan
    # itself -- confirmed in Ray's docs (Loading Data, Performance Tips):
    # "By default, read_parquet() reads all columns... If you only need a
    # subset, specify the list explicitly to avoid loading unnecessary
    # data (projection pushdown)." This dataset also carries
    # observation.state, observation.effort, frame_index, timestamp,
    # task_index -- none of which this script touches, so there's no
    # reason to pull them through memory.
    ds = ray.data.read_parquet(str(data_dir), columns=["episode_index", "action"])
    ds = ds.map_batches(
        add_action_magnitude, batch_format="numpy", batch_size=256,
        udf_modifying_row_count=False,  # true here -- we only add columns
    )
    ds = ds.map_batches(
        flag_outliers_per_block, batch_format="numpy", batch_size=256,
        udf_modifying_row_count=False,
    )

    # materialize() once, before inspecting: runs the pipeline cleanly and
    # pins the result, so take_batch() below slices an already-computed
    # result instead of re-triggering execution. (Not required to avoid the
    # ArrowInvalid error anymore -- that was fixed at the source in
    # tensor_utils.py -- this is just good practice on its own, per Lab 1's
    # ray_bronze_ingest.py writeup.)
    ds = ds.materialize()

    sample = ds.take_batch(batch_size=5, batch_format="numpy")
    print("episode_index:       ", sample["episode_index"])
    print("action_magnitude:    ", np.round(sample["action_magnitude"], 4))
    print("naive per-block flag:", sample["action_magnitude_outlier_PER_BLOCK_NAIVE"])

    ray.shutdown()


if __name__ == "__main__":
    main()