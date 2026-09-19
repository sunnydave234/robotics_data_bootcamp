"""
partition_utils.py -- Week 3 Day 2
=====================================
Two functions:
  1. build_file_group_partitions() -- given a LeRobot v3.0 dataset's episode
     metadata, figure out which episodes share a physical video file, so you
     can read that file ONCE instead of once per episode.
  2. get_episode_frame_counts() -- given a Ray Dataset of decoded video
     frames (one row per frame) and the episodes in one file group, check
     that the number of frames you can find for each episode matches what
     the dataset's own metadata says that episode should have.
"""
from typing import Iterable


def build_file_group_partitions(episodes_df, video_keys: Iterable[str]) -> list[tuple[int, int]]:
    """episodes_df: meta.episodes.to_pandas() for a v3.0 LeRobotDataset --
    NOT the raw Arrow-backed meta.episodes object; convert first.
    video_keys: e.g. ['observation.images.cam_high', ...].

    Returns: a list of (dataset_from_index, dataset_to_index) row ranges.
    Each range covers one GROUP of consecutive episodes that all share the
    exact same video file, for every camera in video_keys at once.

    Raises AssertionError if two episodes share a file-group key but are NOT
    contiguous in dataset_from_index/dataset_to_index -- see the Debugging
    Challenge for when that can legitimately happen.
    """
    # Build the list of column names we need to check, per camera.
    # For 3 cameras this becomes something like:
    #   ["videos/cam_high/chunk_index", "videos/cam_high/file_index",
    #    "videos/cam_left_wrist/chunk_index", "videos/cam_left_wrist/file_index",
    #    "videos/cam_right_wrist/chunk_index", "videos/cam_right_wrist/file_index"]
    key_cols = []
    for vk in video_keys:
        key_cols.append(f"videos/{vk}/chunk_index")
        key_cols.append(f"videos/{vk}/file_index")

    partitions: list[tuple[int, int]] = []
    current_key = None
    current_from = None
    current_to = None

    # episodes_df.to_dict("records") turns the DataFrame into a plain list of
    # dicts, one per row, in the ORIGINAL row order. That order matters here:
    # we're relying on episodes being visited in dataset order so "did the key
    # change since the last row" is a meaningful question.
    for row in episodes_df.to_dict("records"):
        # tuple(...) makes this row's (chunk_index, file_index) pairs, across
        # every camera, into ONE hashable value we can compare with ==.
        # Two rows have the "same file group" only if EVERY camera's
        # (chunk_index, file_index) matches -- that's why we bundle all
        # cameras' values into a single tuple instead of checking one camera.
        key = tuple(row[c] for c in key_cols)
        from_idx, to_idx = row["dataset_from_index"], row["dataset_to_index"]

        if key == current_key:
            # Same file group as the row before it. Before extending the
            # range, insist that this episode picks up exactly where the
            # last one left off. If it doesn't, something about this
            # dataset's structure violates the assumption the whole
            # partitioning technique depends on -- fail loudly here instead
            # of silently producing a wrong range.
            assert from_idx == current_to, (
                f"episode {row['episode_index']} shares a file group with the "
                f"previous episode but isn't contiguous: previous range ends "
                f"at {current_to}, this one starts at {from_idx}."
            )
            current_to = to_idx
        else:
            # The key changed -- we've moved into a new video file. Close
            # out the previous partition (if there was one) and start a new
            # running range at this row.
            if current_key is not None:
                partitions.append((current_from, current_to))
            current_key, current_from, current_to = key, from_idx, to_idx

    # The loop above only closes a partition when it sees the NEXT key
    # change. The very last partition never gets a "next" row to trigger
    # that, so it has to be appended once after the loop ends.
    if current_key is not None:
        partitions.append((current_from, current_to))

    return partitions


def get_episode_frame_counts(ds_frames, group_df, camera: str) -> dict:
    """ds_frames: a Ray Dataset from ray.data.read_videos(..., include_timestamps=True)
    for ONE camera's video file. group_df: the slice of episodes_df belonging
    to that same file group.

    Returns {episode_index: (expected_length, frames_found)} so you can see
    a mismatch per-episode instead of only as one aggregate pass/fail.
    """
    results = {}
    for ep in group_df.to_dict("records"):
        lo = ep[f"videos/{camera}/from_timestamp"]
        hi = ep[f"videos/{camera}/to_timestamp"]

        # lo=lo, hi=hi inside the lambda's signature is deliberate, not
        # decoration -- see the explanation of "late binding" below.
        n_found = ds_frames.filter(
            lambda r, lo=lo, hi=hi: lo <= r["frame_timestamp"] < hi
        ).count()

        results[ep["episode_index"]] = (ep["length"], n_found)
    return results


def get_episode_frame_counts_torchcodec(decoder, group_df, camera: str) -> dict:
    """VERIFIED WORKING (Day 2) -- fixed the decord/libavformat dylib
    mismatch and confirmed correct end to end (all episodes OK on a real
    run). Superseded below by get_episode_frame_counts_pyav(), not
    because this is wrong, but because Anyscale's own published reference
    implementation for this exact job (LeRobot v3.0 + Ray Data) uses
    PyAV, not torchcodec -- kept here as real, working, historical code,
    same reason the blocked decord version above is kept. decoder: a
    torchcodec.decoders.VideoDecoder already opened on this file group's
    video for `camera`.

    Returns {episode_index: (expected_length, frames_found)}.
    """
    results = {}
    for ep in group_df.to_dict("records"):
        lo = ep[f"videos/{camera}/from_timestamp"]
        hi = ep[f"videos/{camera}/to_timestamp"]
        # get_frames_played_in_range does the [lo, hi) slicing itself --
        # no manual filter() needed.
        frames = decoder.get_frames_played_in_range(lo, hi)
        n_found = frames.data.shape[0]
        results[ep["episode_index"]] = (ep["length"], n_found)
    return results


def get_episode_frame_counts_pyav(video_path, group_df, camera: str) -> dict:
    """Current recommended version. Reads frames with PyAV, matching the
    approach Anyscale's own published reference implementation uses for
    this exact job: open the file ONCE, seek to the first episode's
    from_timestamp, then decode forward continuously across the whole
    file group -- not one seek per episode. Every camera in a group
    shares the same physical file across the episodes in that group, so
    one sequential pass buckets every frame into the right episode as it
    streams past each [lo, hi) boundary.

    video_path: path to this file group's video for `camera` (NOT an
    already-open decoder -- this function owns the open/seek/decode).

    Returns {episode_index: (expected_length, frames_found)}.
    """
    import av

    episodes = group_df.to_dict("records")
    results = {ep["episode_index"]: (ep["length"], 0) for ep in episodes}
    windows = [
        (
            ep[f"videos/{camera}/from_timestamp"],
            ep[f"videos/{camera}/to_timestamp"],
            ep["episode_index"],
        )
        for ep in episodes
    ]

    first_lo = windows[0][0]

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        # Seeking lands on the nearest KEYFRAME at or before first_lo, not
        # first_lo exactly -- this is why we still filter by [lo, hi)
        # below instead of trusting the seek to be precise.
        offset = int(first_lo / float(stream.time_base))
        container.seek(offset, backward=True, any_frame=False, stream=stream)

        window_i = 0
        for frame in container.decode(stream):
            t = frame.time  # presentation timestamp in seconds

            # Advance past any windows this frame is already beyond --
            # handles the (expected) case of several episodes streaming
            # by in one continuous pass.
            while window_i < len(windows) and t >= windows[window_i][1]:
                window_i += 1
            if window_i >= len(windows):
                break  # past the last episode in this file group -- done

            lo, hi, ep_idx = windows[window_i]
            if lo <= t < hi:
                expected, found = results[ep_idx]
                results[ep_idx] = (expected, found + 1)
            # else: t < lo -- a pre-target frame from the keyframe rewind,
            # not part of any episode in this group. Skip it.

    return results