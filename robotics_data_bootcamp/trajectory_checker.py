#!/usr/bin/env python3
"""
trajectory_checker.py — Day 4 mini project for robot-data-forge.

Checks joint-space action-following error and suspected arm-flips across a
range of episodes from a LeRobot v3.0 dataset, RAM-safe (ds.hf_dataset only,
no video decode anywhere except --plot). Exits non-zero if anything is
flagged, so this wires into CI the same way sync_checker.py (Day 3) does.

Usage:
    python trajectory_checker.py --repo-id lerobot/aloha_mobile_cabinet --episodes 0-69
    python trajectory_checker.py --repo-id lerobot/aloha_mobile_cabinet --episode 0 --plot
"""
import argparse
import sys

import numpy as np
import pandas as pd

LEFT_RIGHT_SWAP = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6]


def parse_episode_range(spec: str) -> list:
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def get_episode_trajectory(ds, episode_idx: int):
    """
    RAM-safe: tabular columns only via ds.hf_dataset, no video decode.
    Filters by the episode_index COLUMN VALUE (not by position), so this is
    correct whether `ds` is the full dataset or was constructed with
    episodes=[...] -- see the Debugging Challenge for why that matters.
    """
    df = ds.hf_dataset.to_pandas()
    ep_rows = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    if ep_rows.empty:
        raise ValueError(f"episode {episode_idx} not found in this dataset object")
    state = np.stack(ep_rows["observation.state"].to_numpy())
    action = np.stack(ep_rows["action"].to_numpy())
    return state, action


def action_following_error(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """action[t] is an ABSOLUTE commanded joint target (ALOHA leader-follower
    teleop), not a delta -- see the Coding Exercise for why that's true here
    and why it might not be true on a different repo_id."""
    predicted_next = action[:-1]
    actual_next = state[1:]
    return np.linalg.norm(actual_next - predicted_next, axis=-1)


def flag_action_error_outliers(ds, episodes_indices, z_thresh: float= 2.0):
    means = {}
    for ep in episodes_indices:
        state, action = get_episode_trajectory(ds, ep)
        means[ep] = action_following_error(state, action).mean()
    s = pd.Series(means)
    z = (s - s.median()) / s.std()
    return s[z > z_thresh].index.tolist()


def detect_arm_flip(ds, episode_indices, ratio_thresh: float = 4.0):
    per_ep_means = {ep: get_episode_trajectory(ds, ep)[0].mean(axis=0) for ep in episode_indices}
    ref = np.median(np.stack(list(per_ep_means.values())), axis=0)
    flagged = []
    for ep, m in per_ep_means.items():
        err_as_is = np.mean((m - ref) ** 2)
        err_unswapped = np.mean((m[LEFT_RIGHT_SWAP] - ref) ** 2)
        if err_unswapped < err_as_is / ratio_thresh:
            flagged.append(ep)
    return flagged


def plot_episode_joints(ds, episode_idx: int, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state, action = get_episode_trajectory(ds, episode_idx)

    # 1. Create a grid of 2 rows and 7 columns
    fig, axes = plt.subplots(2, 7, figsize=(20, 6), sharex=True)

    # Find list of motors dynamically
    motor_names = ds.meta.features["observation.state"]["names"]["motors"]

    for j in range(len(motor_names)):
        ax = axes[j // 7, j % 7]    # Grab the j-th box perfectly
        # Plot the lines
        ax.plot(state[:, j], label="state")
        ax.plot(action[:, j], "--", label="action", alpha=0.7)
        # Set the text title for this specific box
        ax.set_title(motor_names[j], fontsize=9)
    
    # Put a legend only on the very first plot
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episodes", help="range like 0-65")
    group.add_argument("--episode", type=int, help="single episode, use with --plot")
    parser.add_argument("--plot", action="store_true", help="save a joint-trajectory PNG for --episode")
    parser.add_argument("--z-thresh", type=float, default=2.0)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id)

    if args.plot:
        if args.episode is None:
            parser.error("--plot requires --episode")
        out_path = f"episode{args.episode}__joints.png"
        plot_episode_joints(ds, args.episode, out_path)
        print(f"Saved {out_path}    (left arm top row, righ arm bottom row)")
        return
    
    episodes = parse_episode_range(args.episodes) if args.episodes else [args.episode]
    action_flags = flag_action_error_outliers(ds, episodes, z_thresh=args.z_thresh)
    flip_flags = detect_arm_flip(ds, episodes)
    all_flagged = sorted(set(action_flags) | set(flip_flags))

    print(f"Checked {len(episodes)} episodes.")
    print(f"  action-following outliers flagged: {len(action_flags)}"
          + (f"   ({', '.join(map(str, action_flags))})" if action_flags else ""))
    print(f"  suspected arm-flip flagged: {len(flip_flags)}"
          + (f"          ({', '.join(map(str, flip_flags))})" if flip_flags else ""))

    if all_flagged:
        pct = 100 * len(all_flagged) / len(episodes)
        print(f"FAIL — {len(all_flagged)}/{len(episodes)} episodes flagged ({pct:.1f}%)")
        sys.exit(1)
    else:
        print("PASS - no episodes flagged")
        sys.exit(0)


if __name__ == "__main__":
    main()