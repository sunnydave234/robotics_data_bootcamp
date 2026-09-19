#!/usr/bin/env python3
"""
rosbag2_inspect.py -- read a rosbag2 bag directory (sqlite3 or mcap storage)
without a ROS 2 install, and report per-connection topic/type/count/duration.

Usage:
    python rosbag2_inspect.py --path my_bag_real
    python rosbag2_inspect.py --path my_bag_real --convert-to-mcap-report
"""
import argparse
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# Known custom types this tool can register before opening a bag. A real
# version of this would load definitions from a registry/config rather than
# hardcoding one message -- this is the minimum to make today's bag readable.
JOINT_STATE_MSG="""
int32 frame_index
float32[14] state
float32[14] action
"""

MSGTYPE = "lerobot_msg/msg/JointState"

def build_typestore():
    typestore = get_typestore(Stores.LATEST)
    typestore.register(get_types_from_msg(JOINT_STATE_MSG, MSGTYPE))
    return typestore


def inspect_bag(reader: AnyReader) -> list[dict]:
    rows = []
    for connection in reader.connections:
        timestamps = [ts for _, ts, _ in reader.messages(connections=[connection])]
        is_typed = "/" in connection.msgtype and connection.msgtype not in ("unknown", "")
        rows.append({
            "topic": connection.topic,
            "msgtype": connection.msgtype,
            "typed": is_typed,
            "message_count": len(timestamps),
            "start_ns": min(timestamps) if timetamps else None,
            "end_ns": max(timestamps) if timestamps else None,
        })
    return rows


def print_report(rows: list[dict]) -> None:
    print(f"{'topic':<24s} {'type':<34s} {'count':>8s} {'duration_s':>12s}")
    for r in rows:
        duration_s = (
            (r["end_ns"] - r["start_ns"]) / 1e9
            if r["start_ns"] is not None and r["end_ns"] is not None
            else 0.0
        )
        type_label = r["msgtype"] if r["typed"] else f'{r["msgtype"]} (custom/untyped)'
        print(f"{r['topic']:<24s} {type_label:<34s} {r['message_count']:>8d} {duration_s:>12.2f}")


def convert_to_mcap_report(rows: list[dict]) -> None:
    print("\n--- MCAP re-encoding plan ---")
    for r in rows:
        schema_encoding, message_encoding = ("ros2msg", "cdr") if r["typed"] else ("jsonschema", "json")
        print(
            f"topic={r['topic']!r} schema_name={r['msgtype']!r} "
            f"schema_encoding={schema_encoding!r} message_encoding={message_encoding!r} "
            f"messages={r['message_count']}"
        )


def open_and_inspect(path: Path) -> list[dict] | None:
    """Isolates the part that can fail before there's even a reader to
    inspect. An unreadable bag is a real, reportable outcome, not a crash --
    same triage instinct as Week 1's health scorer: flag it, don't blow up."""
    try:
        with AnyReader([path], default_typestore=build_typestore()) as reader:
            return inspect_bag(reader)
    except Exception as exc:  # rosbags raises several distinct error types
        print(f"UNREADABLE BAG: {path} -- {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, type=Path)
    p.add_argument("--convert-to-mcap-report", action="store_true")
    args = p.parse_args()

    rows = open_and_inspect(args.path)
    if rows is None:
        raise SystemExit(1)  # CI-friendly: unreadable bag is a hard failure

    print_report(rows)
    if args.convert_to_mcap_report:
        convert_to_mcap_report(rows)