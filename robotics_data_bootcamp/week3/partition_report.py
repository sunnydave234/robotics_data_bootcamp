#!/usr/bin/env python3
"""
partition_report.py -- Week 3 Day 2 Mini Project
====================================================
CLI that reports, for any LeRobot v3.0 dataset, how much cheaper it is to
read episodes grouped by shared video file instead of one-episode-at-a-time.

--verify-read uses PyAV to actually open the smallest partition's video
and confirm real frame counts -- NOT ray.data.read_videos(), which
depends on decord (broken here by a Homebrew FFmpeg upgrade removing the
libavformat version decord's binary needs), and not torchcodec either
(worked, but Anyscale's own published reference implementation for this
exact job uses PyAV, so this matches it -- see partition_utils.py's
get_episode_frame_counts_pyav() for the full story).

Usage:
    python partition_report.py --repo-id lerobot/aloha_mobile_cabinet
    python partition_report.py --repo-id lerobot/aloha_mobile_cabinet --verify-read
"""
import argparse

from partition_utils import build_file_group_partitions, get_episode_frame_counts_pyav


def verify_read(meta, episodes_df, partitions, camera: str):
    """Open the smallest partition's video for ONE camera with PyAV and
    confirm real frame counts against meta.episodes. One camera only --
    every camera in a group shares the same episode/frame boundaries by
    construction, so this is representative without tripling the runtime.

    No Ray Data here on purpose: this checks ONE file. Ray Data's job
    (Day 1) is fanning this same per-partition read out across MANY
    partitions in parallel -- worth wiring up once this single-file
    version is solid, not before.
    """
    sizes = [to_idx - from_idx for from_idx, to_idx in partitions]
    target_idx = sizes.index(min(sizes))
    target_from, target_to = partitions[target_idx]

    group = episodes_df[
        (episodes_df["dataset_from_index"] >= target_from)
        & (episodes_df["dataset_to_index"] <= target_to)
    ].reset_index(drop=True)

    if group.empty:
        print(f"no episodes found for partition range ({target_from}, {target_to})")
        return

    row0 = group.iloc[0]
    chunk_i = row0[f"videos/{camera}/chunk_index"]
    file_i = row0[f"videos/{camera}/file_index"]
    video_path = meta.root / f"videos/{camera}/chunk-{chunk_i:03d}/file-{file_i:03d}.mp4"

    print(f"\nsmallest partition: {len(group)} episodes, {target_to - target_from} frames total")
    print(f"opening: {video_path}")

    results = get_episode_frame_counts_pyav(video_path, group, camera)
    print("\nper-episode check:")
    for ep_idx, (expected, found) in results.items():
        status = "OK" if expected == found else "MISMATCH"
        print(f"  episode {ep_idx}: expected={expected} found={found}  [{status}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. lerobot/aloha_mobile_cabinet")
    parser.add_argument(
        "--verify-read",
        action="store_true",
        help="Actually open the smallest file group's video (via torchcodec) "
             "and confirm a real frame count against meta.episodes, instead "
             "of only counting metadata.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera to use for --verify-read. Defaults to the first entry "
             "in meta.video_keys.",
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id)
    episodes_df = meta.episodes.to_pandas()
    video_keys = list(meta.video_keys)

    partitions = build_file_group_partitions(episodes_df, video_keys)

    n_episodes = len(episodes_df)
    n_cameras = len(video_keys)
    naive_opens = n_episodes * n_cameras
    grouped_opens = len(partitions) * n_cameras

    print(f"repo:              {args.repo_id}")
    print(f"episodes:          {n_episodes}")
    print(f"cameras:           {n_cameras}")
    print(f"naive by-episode:  {n_episodes} tasks, {naive_opens} video opens")
    print(f"file-group:        {len(partitions)} tasks, {grouped_opens} video opens")
    if grouped_opens:
        print(f"reduction:         {naive_opens / grouped_opens:.1f}x")

    if args.verify_read:
        camera = args.camera or video_keys[0]
        verify_read(meta, episodes_df, partitions, camera)


if __name__ == "__main__":
    main()