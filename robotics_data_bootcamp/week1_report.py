#!/usr/bin/env python3
import argparse
import sys

sys.path.insert(0, ".")
from episode_card import describe_episode                 # Day 1 (after refactor -- Appendix B)
from sensor_inventory import sensor_coverage_matrix        # Day 2 -- ADAPT if your signature differs
from sync_checker import detect_dropped_frames, detect_duplicate_frames, detect_drift  # Day 3 -- ADAPT
from trajectory_checker import get_episode_trajectory, action_following_error          # Day 4 -- ADAPT

from health_pipeline import EpisodeResult, build_health_table, generate_report


def parse_episode_range(spec: str, total_episodes: int) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), min(int(hi), total_episodes - 1) + 1))
    return [int(spec)]


def run_pipeline(ds, episode_ids: list[int]) -> list[EpisodeResult]:
    results = []

    # Compute sensor coverage ONCE outside the loop -- same lesson as last
    # session's sensor_inventory.py performance fix.
    coverage_df = sensor_coverage_matrix(ds, n_episodes=len(episode_ids))
    coverage_by_ep = {
        row["episode_index"]: row["coverage_fraction"]
        for row in coverage_df.to_dict("records")
    }

    for ep_id in episode_ids:
        try:
            desc = describe_episode(ds, ep_id)
            schema_ok = (desc["state_dim"] == 14 and desc["action_dim"] == 14
                         and len(desc["camera_keys"]) == 3)
        except Exception:
            schema_ok = False

        sensor_coverage = coverage_by_ep.get(ep_id, 0.0)

        try:
            from_idx = ds.meta.episodes["dataset_from_index"][ep_id]
            to_idx = ds.meta.episodes["dataset_to_index"][ep_id]
            ep_table = ds.hf_dataset[from_idx:to_idx]
            dropped = detect_dropped_frames(ep_table["timestamp"], expected_dt=1.0 / ds.meta.fps)
            duped = detect_duplicate_frames(ep_table["frame_index"])
            sync_ok = (not dropped) and (not duped)
        except Exception:
            sync_ok = False

        try:
            states, actions = get_episode_trajectory(ds, ep_id)
            errors = action_following_error(states, actions)
            mean_error = float(errors.mean())
        except Exception:
            mean_error = None   # trajectory_checker couldn't score this one

        results.append(EpisodeResult(
            episode_id=ep_id, schema_ok=schema_ok, sensor_coverage=sensor_coverage,
            sync_ok=sync_ok, action_error=mean_error,
        ))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episodes", required=True, help="e.g. 0-49 or a single index")
    p.add_argument("--out", default="week1_report.md")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id)   # unfiltered -- keep global indexing intact
    episode_ids = parse_episode_range(args.episodes, ds.meta.total_episodes)

    results = run_pipeline(ds, episode_ids)
    rows = build_health_table(results)   # gated scorer is the default
    report = generate_report(rows, repo_id=args.repo_id, ep_range=args.episodes)

    with open(args.out, "w") as f:
        f.write(report)
    print(report)

    excluded_or_flagged = sum(1 for r in rows if r["decision"] in ("excluded", "flagged"))
    flag_rate = excluded_or_flagged / len(rows)
    print(f"\n{excluded_or_flagged}/{len(rows)} episodes flagged or excluded ({flag_rate:.0%})")
    if flag_rate > 0.15:
        print("FAIL — flag rate exceeds 15% threshold")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()