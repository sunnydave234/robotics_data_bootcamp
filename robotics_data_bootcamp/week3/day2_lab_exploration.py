"""
day2_lab_exploration.py -- Week 3 Day 2, Labs 1-2 + Debugging Challenge
==========================================================================
Run this top to bottom in Colab, one section at a time -- it's meant to be
read and stepped through, not executed as a silent batch job. Each section
has a "PREDICT FIRST" comment: stop and guess before you run that part.

Lab 2 uses PyAV, not ray.data.read_videos() and not torchcodec.
read_videos() depends on decord, whose compiled binary needs a specific
FFmpeg version (libavformat.61) that a Homebrew FFmpeg upgrade removed on
this machine -- blocked. torchcodec worked as a fix and was confirmed
correct end to end, but Anyscale's own published reference implementation
for this exact job (LeRobot v3.0 + Ray Data) uses PyAV specifically, so
this version matches it: open the file once, seek to the first episode's
from_timestamp, decode forward continuously across the whole file group.

Requires: lerobot, pandas, av (PyAV -- confirm it's importable; if not,
`pip install av`)
Needs partition_utils.py in the same folder (or on your Python path).
"""
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from partition_utils import build_file_group_partitions, get_episode_frame_counts_pyav


# ======================================================================
# LAB 1a -- rebuild Day 1's partition count with real, reusable code
# ======================================================================
meta = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet")

# meta.episodes is an Arrow-backed HF Dataset, not a pandas DataFrame.
# Week 2 rule: convert before you do anything else with it.
episodes_df = meta.episodes.to_pandas()
print("meta.episodes type:", type(meta.episodes))
print("episodes_df type:  ", type(episodes_df))

video_keys = list(meta.video_keys)
print("video_keys:", video_keys)

partitions = build_file_group_partitions(episodes_df, video_keys)
print(f"\n{len(episodes_df)} episodes -> {len(partitions)} partitions")
for p in partitions:
    print(f"  rows {p[0]}-{p[1]}  (size {p[1] - p[0]})")

# PREDICT FIRST: you already know the count is 4 from Day 1. What you
# haven't checked is whether the row-range VALUES line up with the episode
# rollover points you already found (66, 79, 82). Do they?


# ======================================================================
# LAB 1b -- confirm from_timestamp / to_timestamp actually exist
# ======================================================================
# PREDICT FIRST: chunk_index/file_index showed up cleanly on Day 1. Do you
# expect these two columns to exist as well on YOUR installed version?
ts_cols = [c for c in episodes_df.columns if "from_timestamp" in c or "to_timestamp" in c]
print("\ntimestamp columns found:", ts_cols)


# ======================================================================
# LAB 2 -- open the smallest file group for real (PyAV version)
# ======================================================================
sizes = [to_idx - from_idx for from_idx, to_idx in partitions]
smallest_i = sizes.index(min(sizes))
target_from, target_to = partitions[smallest_i]
print(f"\nsmallest partition: rows {target_from}-{target_to} ({sizes[smallest_i]} frames)")

group = episodes_df[
    (episodes_df["dataset_from_index"] >= target_from)
    & (episodes_df["dataset_to_index"] <= target_to)
].reset_index(drop=True)
print(group[["episode_index", "length", "dataset_from_index", "dataset_to_index"]])

cam = video_keys[0]
row0 = group.iloc[0]
chunk_i, file_i = row0[f"videos/{cam}/chunk_index"], row0[f"videos/{cam}/file_index"]
video_path = meta.root / f"videos/{cam}/chunk-{chunk_i:03d}/file-{file_i:03d}.mp4"
print("video file:", video_path, "exists:", video_path.exists())

import av

# PREDICT FIRST: is `av` (PyAV) importable, or does this need `pip install av`?
# If you saw an "av/.dylibs/libavdevice..." warning in Day 2's earlier
# runs, that's a strong hint it's already there.
with av.open(str(video_path)) as container:
    stream = container.streams.video[0]
    print("stream time_base:", stream.time_base)
    # PREDICT FIRST: does stream.frames match the frame total you printed
    # for this partition above, or is it 0 (some containers don't store
    # a frame count in the header, and PyAV won't decode the whole file
    # just to answer this)?
    print("stream.frames (container header, may be 0/unknown):", stream.frames)

# Slice ONE episode's frames out by seeking to its from_timestamp and
# decoding forward -- PyAV has no built-in "give me frames in [lo, hi)"
# method the way torchcodec did, so this is spelled out by hand once here
# before get_episode_frame_counts_pyav() generalizes it for the whole group.
ep = group.iloc[0]
lo, hi = ep[f"videos/{cam}/from_timestamp"], ep[f"videos/{cam}/to_timestamp"]
print(f"\nepisode {ep['episode_index']}: expect length={ep['length']}, window [{lo}, {hi})")

with av.open(str(video_path)) as container:
    stream = container.streams.video[0]
    offset = int(lo / float(stream.time_base))
    # PREDICT FIRST: does seek() land EXACTLY on timestamp `lo`, or on the
    # nearest keyframe before it (meaning you'll decode a few frames you
    # have to skip before you're actually inside this episode's window)?
    container.seek(offset, backward=True, any_frame=False, stream=stream)

    n_found = 0
    for frame in container.decode(stream):
        t = frame.time
        if t < lo:
            continue  # still in the pre-episode keyframe rewind
        if t >= hi:
            break
        n_found += 1

print("frames found:", n_found)

# Now check EVERY episode in the group at once, with ONE open+seek+decode
# pass instead of one per episode -- this is the Coding Exercise function
# from partition_utils.py, and it also matches Anyscale's reference
# implementation: open the file once, seek once, stream forward.
results = get_episode_frame_counts_pyav(video_path, group, cam)
print("\nper-episode check:")
for ep_idx, (expected, found) in results.items():
    status = "OK" if expected == found else "MISMATCH"
    print(f"  episode {ep_idx}: expected={expected} found={found}  [{status}]")


# ======================================================================
# DEBUGGING CHALLENGE -- break the contiguity assertion on purpose
# ======================================================================
# Two fake episodes that share a file-group key (same chunk_index/file_index)
# but are NOT adjacent in dataset_from_index/dataset_to_index. This should
# never happen on a freshly-recorded dataset, but is exactly what a non-
# trivial edit/merge could produce.
toy = pd.DataFrame([
    {"episode_index": 5,  "dataset_from_index": 500,  "dataset_to_index": 600,
     "videos/cam/chunk_index": 0, "videos/cam/file_index": 0},
    {"episode_index": 60, "dataset_from_index": 6000, "dataset_to_index": 6100,
     "videos/cam/chunk_index": 0, "videos/cam/file_index": 0},
])

try:
    build_file_group_partitions(toy, ["cam"])
    print("\nno error raised -- that's WRONG, this should have failed")
except AssertionError as e:
    print("\nassertion fired as expected:")
    print(" ", e)