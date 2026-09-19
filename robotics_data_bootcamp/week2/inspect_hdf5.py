#!/usr/bin/env python3
"""
inspect_hdf5.py -- Week 2 Day 5 Coding Exercise

Walks an ALOHA-native HDF5 episode file and reports, for every dataset:
shape, dtype, chunk shape, and compression codec. Also flags whether
observations/effort and base_action are present -- both survive raw HDF5
capture on real hardware, but this week's verified finding is that the
LeRobot conversion only carries action, state (<-qpos), and effort
forward. qvel and base_action never make it to the Parquet layer, so if
you're reconstructing an HDF5 file FROM a LeRobot dataset (Lab 1), you
structurally cannot repopulate them -- that data is already gone by the
time it reaches you.
"""
import argparse
from typing import Optional, Dict

def walk_hdf5(path: str, assumed_fps: Optional[float] = None) -> dict:
    """
    Inspect one ALOHA-style HDF5 episode file.
    Returns a dict per-dataset shape/dtype/chunks/compression, a
    best-effort duration, and explicit flags for fields whose absence is 
    meaningful (effort, base_action) rather than silently omitting them.
    """
    import h5py

    datasets: Dict[str, dict] = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = {
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "chunks": obj.chunks,
                "compression": obj.compression,
            }
    
    with h5py.File(path, "r") as f:
        f.visititems(_visit)
        attrs = dict(f.attrs)
    
    action_shape = datasets.get("action", {}).get("shape")
    n_frames = action_shape[0] if action_shape else None