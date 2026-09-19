
#!/usr/bin/env python3
"""
sensor_inventory.py — Day 2 Mini Project (Week 1)
 
Scans N episodes of a LeRobot v3.0 dataset and reports which modalities are
present, absent, or degenerate (present but suspiciously low variance across
the episode). Writes coverage_report.csv and prints a summary.
 
Usage:
    python sensor_inventory.py --repo-id lerobot/aloha_mobile_cabinet --episodes 20
"""
import argparse

import numpy as np
import pandas as pd

MODALITY_PREFIXES = ("observation.images.")
MODALITY_EXACT = ("observation.state", "action")

def first_frame_of_episode(ds, meta, episode_idx:int) -> dict:
    """
    v3.0-correct episode access: jump straight to an episode's first frame
    via metadata, instead of assuming ds[i] walks episodes - it walks the
    global frame table.
    """
    from_idx = int(meta.episodes["dataset_from_index"][episode_idx])
    return ds[from_idx]


def _is_modality_key(key: str) -> bool:
    return key.startswith(MODALITY_PREFIXES) or key in (MODALITY_EXACT)


def sensor_coverage_matrix(ds, meta, n_episodes: int, samples_per_episode: int = 10, variance_threshold: float = 1e-6) -> pd.DataFrame:
    """
    Episodes as rows, modality as keys as columns, values in {present, degenerate}.
    'Degenerate' = key exists but variance across sampled frames is -0 (frozen
    camera, dead sensor) - presence alone doesn't catch this.
    """
    rows = []
    for ep_idx in range(n_episodes):
        from_idx = int(meta.episodes["dataset_from_index"][ep_idx])
        to_idx   = int(meta.episodes["dataset_to_index"][ep_idx])
        first = first_frame_of_episode(ds, meta, ep_idx)
        row = {"episode": ep_idx}
        n_samples = min(samples_per_episode, to_idx-from_idx)
        sample_idxs = np.linspace(from_idx, to_idx-1, num=n_samples, dtype=int)
        for key in first:
            if not _is_modality_key(key):
                continue
            values = np.stack([np.asarray(ds[i][key]).ravel() for i in sample_idxs])
            variance = values.astype(float).var()
            row[key] = "degenerate" if variance < variance_threshold else "present"
        rows.append(row)
    return pd.DataFrame(rows).set_index("episode")


def summarized(df: pd.DataFrame, meta, repo_id:str) -> str:
    lines = [
        f"Modality coverage - {repo_id} "
        f"({meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps})"
    ]
    for col in df.columns:
        present_pct = (df[col] == "present").mean() * 100
        degenerate_pct = (df[col] == "degenerate").mean() * 100
        lines.append(f"     {col:38s} present {present_pct:5.1f}%   degenerate  {degenerate_pct: 5.1f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default="coverage_report.csv")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id)
    ds = LeRobotDataset(args.repo_id)

    n = min(args.episodes, meta.total_episodes)
    df = sensor_coverage_matrix(ds, meta, n)
    df.to_csv(args.out)
    print(summarized(df, meta, args.repo_id))
    print(f"\n{args.out} written ({len(df)} rows)")


if __name__ == "__main__":
    main()
