"""
Sample N episodes of an OXE sub-dataset from the public GCS bucket, report
the free corpus-wide total plus sampled step-count/schema stats.
"""
import argparse
import statistics
import numpy as np
import tensorflow_datasets as tfds


def oxe_builder(name, version="0.1.0"):
    return tfds.builder_from_directory(builder_dir=f"gs://gresearch/robotics/{name}/{version}")


def dims(action):
    if isinstance(action, dict):
        return sum(np.asarray(v).size for v in action.values()), sorted(action.keys())
    return int(np.asarray(action).size), []


def inspect(name, n_episodes, version="0.1.0"):
    