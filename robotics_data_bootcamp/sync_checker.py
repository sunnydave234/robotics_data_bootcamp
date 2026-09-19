#!/usr/bin/env python3
"""
sync_checker.py — tiered sync-integrity report for a LeRobotDataset.

The whole design follows one rule from this week's lessons: spend cheap
checks everywhere, expensive checks only where they earn their cost.

  Tier 1 (tabular)  — dropped frames, duplicate frame_index.
                      Reads only ds.hf_dataset (parquet). No video decode.
                      Cheap enough to run on every requested episode.

  Tier 2 (visual)   — held/duplicated camera frames.
                      Needs ds[i] to decode real pixels. Expensive, so it
                      only runs on a small sample: episodes tabular checks
                      already flagged, plus a few random ones (because
                      held frames are invisible to timestamps entirely --
                      a clean-looking episode can still hide them).

The report explicitly separates "checked, and clean" from "never checked"
for every episode -- those are different claims, and blurring them is the
easiest way to ship a report that looks thorough but isn't.

Usage:
    python sync_checker.py --repo-id lerobot/aloha_mobile_cabinet \
        --episodes 0-49 --visual-sample 10
"""

import argparse
import json
import random
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Detectors. Each one is a pure function: hand it arrays, get back flagged
# indices. Keeping them pure (no dataset object required) is what makes them
# unit-testable on a synthetic array you construct by hand -- the only way
# to know a detector catches what you think it catches, per this week's
# closing lesson, instead of just running clean on real data by accident.
# ---------------------------------------------------------------------------

def detect_dropped_frames(timestamps, expected_dt, tolerance=1.5):
    """
    Pure timestamp math, no images touched -- this is nearly free.
    A dropped frame means real time jumped: the gap to the next row is
    much bigger than the nominal 1/fps interval. Flag any gap that
    exceeds `tolerance` times what's expected.
    """
    deltas = np.diff(timestamps)
    return np.where(deltas > tolerance * expected_dt)[0]


def detect_duplicate_frames(frame_index_col):
    """
    frame_index is bookkeeping: it should be a clean 0..N-1 sequence
    within an episode, each value appearing exactly once. A repeated
    value means the dataset's OWN metadata is inconsistent -- a
    build-time bug, not a camera behaving normally. Still pure tabular
    math, no video decode.
    """
    vals, counts = np.unique(frame_index_col, return_counts=True)
    return vals[counts > 1]


def held_frame_runs(ds, from_idx, to_idx, camera_key, atol=1e-4):
    """
    Streaming pairwise pixel comparison across one episode.
    Holds at most one previous decoded frame in memory at a time, so
    peak memory is "two frames," never "one whole episode" (this is the
    fix for the Lab 1 memory crash, applied here to images instead of
    timestamps). Still pays one video decode per frame -- unavoidable,
    since you're comparing actual pixels -- which is exactly why this
    only ever runs on a small sample of episodes, never all of them.
    """
    run_lengths, current = [], 1
    prev = ds[from_idx][camera_key].numpy()
    for i in range(from_idx + 1, to_idx):
        curr = ds[i][camera_key].numpy()
        if np.allclose(curr, prev, atol=atol):
            current += 1                    # same picture as before -> streak continues
        else:
            run_lengths.append(current)     # streak just ended -> record its length
            current = 1                     # start a new streak at this frame
        prev = curr
    run_lengths.append(current)             # final streak never got "closed" by a mismatch
    return run_lengths


def detect_held_frames(ds, from_idx, to_idx, camera_key, max_run=3, atol=1e-4):
    """
    Some holding is normal and NOT a bug -- e.g. a 15Hz camera stretched
    onto a 30Hz declared clock will always show runs of ~2. Only flag
    runs LONGER than max_run, so routine resampling doesn't look like a
    stall.
    """
    runs = held_frame_runs(ds, from_idx, to_idx, camera_key, atol=atol)
    return [r for r in runs if r > max_run]


# ---------------------------------------------------------------------------
# Small helper: turn "0-49" into a list of episode indices, and refuse to
# silently run past the dataset's real episode count on a typo.
# ---------------------------------------------------------------------------

def parse_episode_range(spec, total_episodes):
    if "-" in spec:
        start, end = spec.split("-")
        start, end = int(start), int(end)
    else:
        start = end = int(spec)
    end = min(end, total_episodes - 1)
    return list(range(start, end + 1))


# ---------------------------------------------------------------------------
# The tiered check itself
# ---------------------------------------------------------------------------

def run_sync_check(repo_id, episode_spec, visual_sample_size, seed=0):
    # We need the FULL LeRobotDataset, not just LeRobotDatasetMetadata --
    # tier 1 needs ds.hf_dataset (tabular) and tier 2 needs ds[i] (pixels).
    # Loading it does not by itself decode any video; decode only happens
    # per-frame, later, when we actually index into a camera key.
    ds = LeRobotDataset(repo_id)
    fps = ds.meta.fps
    expected_dt = 1.0 / fps
    total_episodes = ds.meta.total_episodes

    episodes = parse_episode_range(episode_spec, total_episodes)
    # Only checking the first camera by default -- checking every camera
    # view would multiply the (already expensive) visual-check cost by the
    # number of cameras. This is a real, honest limitation, not hidden:
    # it's called out again in the printed report below.
    camera_key = ds.meta.camera_keys[0]

    # One explicit status record per episode, built up front, so that at
    # the end every episode has an answer to "was this checked, and what
    # did we find" -- rather than that information only existing implicitly
    # in which lists an episode number happens to appear in.
    report = {
        ep: {"dropped": False, "duplicate": False, "visual_checked": False, "held": False}
        for ep in episodes
    }

    # ----- Tier 1: tabular checks -- every requested episode, no video decode -----
    dropped_eps = []
    duplicate_eps = []

    for ep in episodes:
        from_idx = ds.meta.episodes["dataset_from_index"][ep]
        to_idx = ds.meta.episodes["dataset_to_index"][ep]

        # Slicing ds.hf_dataset reads the parquet table directly and never
        # touches the MP4s -- this is the Lab 1 fix (pull the cheap channel
        # you actually need, instead of ds[i]'s full image+state+action bundle).
        rows = ds.hf_dataset[from_idx:to_idx]
        timestamps = np.asarray(rows["timestamp"])
        frame_index_col = np.asarray(rows["frame_index"])

        if len(detect_dropped_frames(timestamps, expected_dt)) > 0:
            dropped_eps.append(ep)
            report[ep]["dropped"] = True

        if len(detect_duplicate_frames(frame_index_col)) > 0:
            duplicate_eps.append(ep)
            report[ep]["duplicate"] = True

    # ----- Pick which episodes earn the expensive visual pass -----
    # Priority 1: anything tabular checks already flagged -- top suspects.
    # Priority 2: fill remaining slots randomly. Held frames are invisible
    # to timestamps entirely, so an episode with a perfectly clean tabular
    # result could still be full of duplicated pixels -- random coverage
    # is how you catch that class of problem instead of only ever
    # re-confirming what tier 1 already found.
    suspects = sorted(set(dropped_eps) | set(duplicate_eps))
    remaining_pool = [ep for ep in episodes if ep not in suspects]

    rng = random.Random(seed)
    rng.shuffle(remaining_pool)

    visual_sample = suspects[:visual_sample_size]
    slots_left = visual_sample_size - len(visual_sample)
    if slots_left > 0:
        visual_sample += remaining_pool[:slots_left]
    visual_sample = sorted(visual_sample)

    # ----- Tier 2: visual checks -- sampled episodes only, video decode required -----
    held_eps = []
    for ep in visual_sample:
        from_idx = ds.meta.episodes["dataset_from_index"][ep]
        to_idx = ds.meta.episodes["dataset_to_index"][ep]

        # Mark "checked" BEFORE looking at the result, so that even if this
        # episode's decode crashes partway through, the report still
        # honestly shows it was attempted rather than silently looking
        # like it was skipped.
        report[ep]["visual_checked"] = True

        flagged_runs = detect_held_frames(ds, from_idx, to_idx, camera_key)
        if len(flagged_runs) > 0:
            held_eps.append(ep)
            report[ep]["held"] = True

    return {
        "episodes": episodes,
        "camera_key": camera_key,
        "dropped_eps": dropped_eps,
        "duplicate_eps": duplicate_eps,
        "visual_sample": visual_sample,
        "held_eps": held_eps,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Report printing. The one rule this has to follow: never let "not flagged"
# and "not checked" look the same on the page.
# ---------------------------------------------------------------------------

def print_report(result, episodes):
    n = len(episodes)
    n_visual = len(result["visual_sample"])

    print(f"Tabular checks — {n}/{n} episodes (hf_dataset only, no video decode)")
    dropped_note = f"        (episodes {', '.join(map(str, result['dropped_eps']))})" if result["dropped_eps"] else ""
    dup_note = f"      (episodes {', '.join(map(str, result['duplicate_eps']))})" if result["duplicate_eps"] else ""
    print(f"  dropped_frames flagged: {len(result['dropped_eps'])}{dropped_note}")
    print(f"  duplicate_frames flagged: {len(result['duplicate_eps'])}{dup_note}")
    print()

    print(f"Visual checks — {n_visual}/{n} episodes sampled (video decode required)")
    held_note = f"   (episodes {', '.join(map(str, result['held_eps']))})" if result["held_eps"] else ""
    print(f"  held_frame_runs > 3 flagged: {len(result['held_eps'])}{held_note}")
    print(f"  (camera checked: {result['camera_key']} only -- other camera views were not decoded)")
    print()

    # The line the whole exercise is really about: which episodes never
    # got a visual pass at all, printed explicitly instead of left implicit.
    not_visually_checked = [ep for ep in episodes if ep not in result["visual_sample"]]
    coverage_list = ", ".join(map(str, not_visually_checked)) if not_visually_checked else "none"
    print(f"Not visually checked ({len(not_visually_checked)} episodes): {coverage_list}")
    print("  ('not flagged' and 'not checked' are different claims -- these have")
    print("   no held-frame information either way)")
    print()

    n_tabular_flagged = len(set(result["dropped_eps"]) | set(result["duplicate_eps"]))
    n_visual_flagged = len(result["held_eps"])
    tabular_pct = (n_tabular_flagged / n * 100) if n else 0.0
    visual_pct = (n_visual_flagged / n_visual * 100) if n_visual else 0.0

    hard_failure = n_tabular_flagged > 0 or n_visual_flagged > 0
    status = "FAIL" if hard_failure else "PASS"

    print(f"{status} — {n_tabular_flagged}/{n} flagged by tabular checks ({tabular_pct:.0f}%); "
          f"{n_visual_flagged}/{n_visual} sampled flagged by visual checks ({visual_pct:.0f}%)")

    return hard_failure


def main():
    parser = argparse.ArgumentParser(description="Tiered sync-integrity checker for a LeRobotDataset.")
    parser.add_argument("--repo-id", required=True, help="e.g. lerobot/aloha_mobile_cabinet")
    parser.add_argument("--episodes", default="0-9", help="episode range, e.g. '0-49'")
    parser.add_argument("--visual-sample", type=int, default=10,
                         help="how many episodes get the expensive pixel-level check")
    parser.add_argument("--seed", type=int, default=0,
                         help="random seed for sampling the non-suspect episodes")
    parser.add_argument("--report-json", default=None,
                         help="optional path to dump the full per-episode status as JSON, "
                              "for a pipeline to consume instead of parsing printed text")
    args = parser.parse_args()

    result = run_sync_check(args.repo_id, args.episodes, args.visual_sample, seed=args.seed)
    hard_failure = print_report(result, result["episodes"])

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(result["report"], f, indent=2)
        print(f"\nFull per-episode status written to {args.report_json}")

    # Non-zero exit code lets a training pipeline / CI system gate on this
    # automatically -- refuse to proceed to training -- without a human
    # needing to read the printed report at all.
    sys.exit(1 if hard_failure else 0)


if __name__ == "__main__":
    main()