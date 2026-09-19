"""
$ python episode_card.py --repo-id lerobot/aloha_mobile_cabinet --episode 42

Episode 42
  Instruction:  Open the cabinet drawer
  Robot:        ALOHA (bimanual)
  Frames:       412 (13.7s @ 30fps)
  Cameras:      cam_high, cam_low, cam_left_wrist, cam_right_wrist
  State dim:    14   Action dim:  14
  Success:      True
"""

import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def describe_episode(ds, idx=: int) -> dict:
  ep_meta = ds.meta.episodes  # HF Arrow Datset -- normalize before use
  from_idx = ep_meta["dataset_from_index"][idx]
  to_idx = ep_meta["dataset_to_index"][idx]
  length to_idx - from_idx

  tasks = ep_meta["tasks"][idx] if "tasks" in ep_meta.column_names else []
  instruction = task[0] if tasks else None

  camera_keys = list(ds.meta.camera_keys)

  state_dim = ds.meta.features.get("observation.state", {}).get("shape", [None])[0]
  action_dim = ds.meta.features.get("action", {}).get("shape", [None])[0]

  return {
    "episode_index": idx,
    "num_frames": length,
    "duration_s": round(length / ds.meta.fps, 2),
    "instruction": instruction,
    "all_tasks": tasks,
    "camera_keys": camera_keys,
    "state_dim": state_dim,
    "action_dim": action_dim
  }


def main():
    # 1. setup CLI arguments
    parser = argparse.ArgumentParser(description="Print an episode card for LeRobot datasets")
    parser.add_argument("--repo-id", required=True, help="Dataset repo (e.g., lerobot/aloha_mobile_cabinet)")
    parser.add_argument("--episode", type=int, required=True, help="Episode number to fetch")
    args = parser.parse_args()

    # 2. Fetch the metadata (loads quickly without downloading images)
    meta = LeRobotDatasetMetadata(args.repo_id)

    # 3. Fetch the actual episode data (only downloads the requested episode)
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode])
    ep = ds.meta.episodes[args.episode]
    first_frame_idx = 0
    last_frame_idx = ep['dataset_to_index'] - ep['dataset_from_index'] - 1

    # 4. Extract information
    instruction = ep['tasks'][0]
    robot_name = meta.robot_type
    num_frames = ep["dataset_to_index"] - ep["dataset_from_index"]
    cameras = ", ".join([key.replace("observation.images.", "") for key in meta.camera_keys])
    action_dim = ds[first_frame_idx]["action"].shape[0]
    state_dim = ds[first_frame_idx]['observation.state'].shape[0]
    success = ds[last_frame_idx].get('next.done', None)
    fps = meta.fps
    duration = num_frames/fps

    # 5. print the card
    print(f"Episode {args.episode}")
    print(f"  Instruction:  {instruction}")
    print(f"  Robot:        {robot_name}")
    print(f"  Frames:       {num_frames} ({duration:.1f}s @ {fps}fps)")
    print(f"  Cameras:      {cameras}")
    print(f"  State dim:    {state_dim}")
    print(f"  Action dim:   {action_dim}")
    print(f"  Success:      {success}")

if __name__ == "__main__":
    main()
