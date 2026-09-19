#!/usr/bin/env python3
"""
mcap_from_lerobot.py -- write one LeRobot v3.0 episode out as an MCAP file.

Design notes (the parts that are not obvious):
  * Episode frame range is resolved by the `episode_index` VALUE, never by
    position. Week 1 lesson: dataset_from_index/dataset_to_index are GLOBAL,
    and LeRobotDataset(repo_id, episodes=[...]) re-indexes locally.
  * Tabular reads go through ds.hf_dataset as ONE batched slice. Per-row
    `ds.hf_dataset[i]` in a loop is ~N Arrow reads instead of 1; `ds[i]` would
    additionally decode video for every camera.
  * LeRobot `timestamp` is float32 SECONDS FROM EPISODE START (it resets to
    0.0 on every episode). MCAP `log_time` is an int64 nanosecond count on a
    monotonic, file-wide clock. Writing `timestamp * 1e9` straight through
    makes every episode in a multi-episode file occupy the same time window.
    --time-base global fixes that by adding a per-episode offset.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

NS_PER_S = 1_000_000_000


# -----------------------------------------------------------------------------
# tabular extraction (no mcap dependency -- importable and testable on its own)
# -----------------------------------------------------------------------------
def _to_pylist(v):
    """hf_dataset rows come back as torch tensors (hf_transform_to_torch),
    but stay list/ndarray under some versions. Normalise without assuming."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _to_pyfloat(v):
    if hasattr(v, "item"):
        return float(v.item())
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)


def episode_bound(ds, episode_index: int) -> tuple[int, int]:
    """Global [from, to) row range for one episode, by VALUE not position."""
    eps = ds.meta.episodes
    eps = eps.to_pandas() if hasattr(eps, "to_pandas") else eps
    row = eps[eps["episode_index"] == episode_index]
    if len(row) == 0:
        raise ValueError(f"episode_index {episode_index} not in meta.episodes")
    return int(row["dataset_from_index"].iloc[0]), int(row["dataset_to_index"].iloc[0])


def read_episode_tabular(ds, episode_index: int) -> list[dict]:
    """One batched Arrow slice -> list of plain-python row dicts. No video decode."""
    from_idx, to_idx = episode_bound(ds, episode_index)
    cols = ds.hf_dataset[from_idx:to_idx]
    n = len(cols["timestamp"])
    out = []
    for j in range(n):
        out.append({
            "frame_index": int(_to_pyfloat(cols["frame_index"][j])),
            "episode_index": int(_to_pyfloat(cols["episode_index"][j])),
            "timestamp": int(_to_pyfloat(cols["timestamp"][j])),
            "state": _to_pylist(cols["observation.state"][j]),
            "action": _to_pylist(cols["action"][j]),
        })
    return out


def episode_time_offset_ns(ds, episode_index:int) -> int:
    """
    Cumulative nanosecond offset so episodes don't overlap in a shared file.
    Uses declared length/fps rather than reading every earlier episode's
    timestamps -- cheap, and only needs to be monotonic, not physically true.
    """
    eps = ds.meta.episodes
    eps = eps.to_pandas() if hasattr(eps, "to_pandas") else eps
    eps = eps.sort_values("episode_index")
    fps = float(ds.meta.fps)
    prior = eps[eps["episode_index"] < episode_index]
    if "length" in prior.columns:
        frames_before = int(prior["length"].sum())
    else:
        frames_before = int((prior["dataset_to_index"] - prior["dataset_from_index"]).sum())
    return int(round(frames_before / fps * NS_PER_S))


# --------------------------------------------------------------------------
# mcap writing
# --------------------------------------------------------------------------
JOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "frame_index": {"type": "integer"},
        "episode_index": {"type": "integer"},
        "state": {"type": "array", "items": {"type": "number"}},
        "action": {"type": "array", "items": {"type": "number"}}
    },
}

VECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "frame_index": {"type": "integer"},
        "values": {"type": "array", "items": {"type": "number"}}
    },
}


def _compression(name: str):
    from mcap.writer import CompressionType
    return {
        "zstd": CompressionType.ZSTD,
        "lz4": CompressionType.LZ4,
        "none": CompressionType.NONE
    }[name]


def write_mcap(rows, out_path, *, fps, repo_id, episode_index, motor_names,
               split_topics=False, time_offset_ns=0, compression="zstd",
               chunk_size=1024 * 1024):
    from mcap.writer import Writer

    with open(out_path, "wb") as f:
        writer = Writer(f, chunk_size=chunk_size, compression=_compression(compression))
        writer.start(profile="x-lerobot", library="robot-data-forge/mcap_from_lerobot")

        # provenance travels WITH the data -- the whole point of the format
        writer.add_metadata("source", {
            "repo_id": str(repo_id),
            "episode_index": str(episode_index),
            "fps": str(fps),
            "motor_names": json.dumps(motor_names),
            "time_base": "global" if time_offset_ns else "episode",
        })

        if split_topics:
            schema_id = writer.register_schema(
                name="lerobot.JointVector", encoding="jsonschema",
                data=json.dumps(VECTOR_SCHEMA).encode()
            )
            channels = {
                "/observation/state": writer.register_channel(
                    schema_id=schema_id, topic="/observation/state",
                    message_encoding="json"
                ),
                "/action": writer.register_channel(
                    schema_id=schema_id, topic="/action",
                    message_encoding="json"
                )
            }
        else:
            schema_id = writer.register_schema(
                name="lerobot.JointState", encoding="jsonschema",
                data=json.dumps(JOINT_SCHEMA).encode()
            )
            channels = {
                "/joint_state": writer.register_channel(
                    schema_id=schema_id, topic="joint_state",
                    message_encoding="json"
                )
            }
        
        for seq, row in enumerate(rows):
            log_time = int(round(row["timestamp"] * NS_PER_S)) + time_offset_ns
            if split_topics:
                for topic, key in (("/observation/state", "state"), ("/action", "action")):
                    payload = {"frame_index": row["frame_index"], "values": row[key]}
                    writer.add_message(
                        channel_id=channels[topic], log_time=log_time,
                        publish_time=log_time, sequence=seq,
                        data=json.dumps(payload).encode("utf-8")
                    )
            else:
                payload = {
                    "frame_index": row["frame_index"],
                    "episode_index": row["episode_index"],
                    "state": row["state"],
                    "action": row["action"],
                }
                writer.add_message(
                    channel_id=channels["/joint_state"], log_time=log_time,
                    publish_time=log_time, sequence=seq,
                    data=json.dumps(payload).encode("utf-8"))
        writer.finish()         # summary + index live here. Skiip it, lose fast seek.
    return len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="LeRobot episode -> MCAP")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--split-topics", action="store_true",
                   help="write /observation/state and /action as separate topics")
    p.add_argument("--time-base", choices=["episode", "global"], default="episode",
                   help="'episode' = log_time starts at 0 (collides across episodes); "
                        "'global' = offset by prior episodes' duration")
    p.add_argument("--compression", choices=["zstd", "lz4", "none"], default="zstd")
    p.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = p.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id)
    rows = read_episode_tabular(ds, args.episode)

    out_target = args.out or f"episode{args.episode}.mcap"
    out = os.path.abspath(out_target)

    offset = episode_time_offset_ns(ds, args.episode) if args.time_base == "global" else 0
    # out = args.out or f"episode{args.episode}.mcap"

    motor_names = ds.meta.features["observation.state"].get("names") or []
    if isinstance(motor_names, dict):
        motor_names = motor_names.get("motors", [])
    
    print(f"Target file path: {out}")

    n = write_mcap(rows, out, fps=ds.meta.fps, repo_id=args.repo_id,
                   episode_index=args.episode, motor_names=motor_names,
                   split_topics=args.split_topics, time_offset_ns=offset,
                   compression=args.compression, chunk_size=args.chunk_size)
    span = rows[-1]["timestamp"] - rows[0]["timestamp"]
    print(f"wrote {out}: {n} frames, {span:.2f}s @ {ds.meta.fps}fps, "
          f"time_base={args.time_base}, offset={offset}ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())