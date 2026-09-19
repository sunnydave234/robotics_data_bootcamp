#!/usr/bin/env python3
"""Diff observation/action schemas between two Open X-Embodiment sub-datasets,
reading directly from the public GCS bucket (no HF mirror dependency)."""
import argparse
import numpy as np
import tensorflow_datasets as tfds

def oxe_builder(name, version="0.1.0"):
    return tfds.builder_from_directory(builder_dir=f"gs://gresearch/robotics/{name}/{version}")


def first_step(name, version="0.1.0"):
    builder = oxe_builder(name, version)
    ds = builder.as_dataset(split="train")
    episode = next(iter(ds.take(1)))
    step = next(iter(episode["steps"].take(1)))
    return step, ds.info.splits["train"].num_examples


def dims(feature):
    """
    Sum leaf tensor sizes. Handles both a flat tensor (aloha_mobile-style)
    and a dict of named sub-fields (the OXE-standard case).
    """
    if isinstance(feature, dict):
        return sum(np.asarray(v).size for v in feature.values())
    return int(np.asarray(feature).size)


def key_set(feature):
    return set(feature.keys()) if isinstance(feature, dict) else set()


def shape_dtype(v):
    arr = np.asarray(v)
    return arr.shape, arr.dtype


def diff(name_a, name_b, version_a="0.1.0", version_b="0.1.0"):
    step_a, total_a = first_step(name_a, version_a)
    step_b, total_b = first_step(name_b, version_b)

    obs_a, obs_b = step_a["observation"], step_b["observation"]
    act_a, act_b = step_a["action"], step_b["action"]

    obs_keys_a, obs_keys_b = set(obs_a.keys()), set(obs_b.keys())
    act_keys_a, act_keys_b = key_set(act_a), key_set(act_b)

    print(f"=== {name_a} (total_episodes={total_a}) vs {name_b} (total_episodes={total_b}) ===")
    print(f"observation keys only in {name_a}: {sorted(obs_keys_a - obs_keys_b)}")
    print(f"observation keys only in {name_b}: {sorted(obs_keys_b - obs_keys_a)}")
    print(f"action keys only in {name_a}: {sorted(act_keys_a - act_keys_b)}")
    print(f"action keys only in {name_b}: {sorted(act_keys_b - act_keys_a)}")

    for k in sorted(act_keys_a & act_keys_b):
        sa, da = shape_dtype(act_a[k])
        sb, db = shape_dtype(act_b[k])

        if (sa, da) != (sb, db):
            print(f"  action['{k}'] shape/dtype mismatch: "
                  f"{name_a}={sa}/{da}  {name_b}={sb}/{db}")
    print(f"total action dims: {name_a}={dims(act_a)}  {name_b}={dims(act_b)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-a", required=True)
    p.add_argument("--dataset-b", required=True)
    p.add_argument("--version-a", default="0.1.0")
    p.add_argument("--version-b", default="0.1.0")
    args = p.parse_args()
    diff(args.dataset_a, args.dataset_b, args.version_a, args.version_b)