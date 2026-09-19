#!/usr/bin/env python3
"""
mcap_inspect.py -- report topics, schemas, counts, duration and inter-message
gaps for any MCAP file.

Tiering habit from Week 1's sync_checker.py:
  cheap pass  -> reader.get_summary()  (reads the summary section only)
  full pass   -> iter_messages()       (touches every message; needed for gaps)

Exit code 1 if any topic has a flagged gap => CI-usable.
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import median

NS_PER_S = 1_000_000_000


# --------------------------------------------------------------------------
# analysis core -- no mcap dependency
# --------------------------------------------------------------------------
def gap_report(log_times_ns, gap_factor=1.5):
    """Inter-message gap stats + flags. Week 1's dropped-frame logic, one
    layer down: container messages instead of dataset rows.

    Threshold is RELATIVE TO THE MEDIAN, never an equality test -- float32
    timestamps upstream mean consecutive gaps never repeat exactly.
    """
    lt = sorted(int(t) for t in log_times_ns)
    if len(lt) < 2:
        return {"count": len(lt), "median_gap_ns": None, "max_gap_ns": None,
                "min_gap_ns": None, "threshold_ns": None, "flagged": [],
                "non_monotonic": 0}
    raw = [int(t) for t in log_times_ns]
    non_monotonic = sum(1 for a, b in zip(raw, raw[1:]) if b < a)
    gaps = [b - a for a, b in zip(lt, lt[1:])]
    med = float(median(gaps))
    threshold = med * gap_factor
    flagged = [{"after_message_index": i, "gap_ns": g,
                "gap_over_median": (g / med) if med else None}
               for i, g in enumerate(gaps) if med > 0 and g > threshold]
    return {"count": len(lt), "median_gap_ns": med, "max_gap_ns": max(gaps),
            "min_gap_ns": min(gaps), "threshold_ns": threshold,
            "implied_hz": (NS_PER_S / med) if med else None,
            "flagged": flagged, "non_monotonic": non_monotonic}


def build_topic_stats(records, gap_factor=1.5):
    """records: iterable of (topic, schema_name, encoding, log_time_ns)."""
    by_topic: dict[str, dict] = {}
    for topic, schema_name, encoding, log_time in records:
        t = by_topic.setdefault(topic, {"topic": topic, "schema": schema_name,
                                        "encoding": encoding, "log_times": []})
        t["log_times"].append(int(log_time))
    out = []
    for topic, t in sorted(by_topic.items()):
        lts = t.pop("log_times")
        rpt = gap_report(lts, gap_factor=gap_factor)
        t.update(rpt)
        t["start_ns"] = min(lts) if lts else None
        t["end_ns"] = max(lts) if lts else None
        t["duration_s"] = ((t["end_ns"] - t["start_ns"]) / NS_PER_S) if lts else None
        out.append(t)
    return out


def format_report(stats, path):
    lines = [f"MCAP: {path}", f"  topics: {len(stats)}"]
    for t in stats:
        lines.append(f"  {t['topic']}  schema={t['schema']} encoding={t['encoding']}")
        detail = f"    messages: {t['count']}"
        if t.get("duration_s") is not None:
            detail += f"   duration: {t['duration_s']:.3f}s"
        if t.get("implied_hz"):
            detail += f"   implied: {t['implied_hz']:.2f} Hz"
        lines.append(detail)
        if t["median_gap_ns"]:
            lines.append(
                f"    gap ns  median={t['median_gap_ns']:.0f}"
                f"  min={t['min_gap_ns']}  max={t['max_gap_ns']}"
                f"  threshold={t['threshold_ns']:.0f}")
        if t["non_monotonic"]:
            lines.append(f"    WARN non-monotonic log_time transitions: {t['non_monotonic']}")
        if t["flagged"]:
            lines.append(f"    FLAG {len(t['flagged'])} gap(s) over threshold:")
            for f in t["flagged"][:10]:
                lines.append(f"      after msg {f['after_message_index']}: "
                             f"{f['gap_ns']} ns ({f['gap_over_median']:.2f}x median)")
            if len(t["flagged"]) > 10:
                lines.append(f"      ... {len(t['flagged']) - 10} more")
        else:
            lines.append("    OK no gaps over threshold")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# mcap I/O
# --------------------------------------------------------------------------
def cheap_summary(path):
    """Summary-section-only read. Returns None if the file has no summary
    (e.g. writer.finish() was never called)."""
    from mcap.reader import make_reader
    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    if summary is None or summary.statistics is None:
        return None
    st = summary.statistics
    return {
        "message_count": st.message_count,
        "schema_count": st.schema_count,
        "channel_count": st.channel_count,
        "chunk_count": st.chunk_count,
        "start_ns": st.message_start_time,
        "end_ns": st.message_end_time,
        "duration_s": (st.message_end_time - st.message_start_time) / NS_PER_S,
        "channels": {c.id: c.topic for c in summary.channels.values()},
        "per_channel": dict(st.channel_message_counts),
    }


def iter_records(path):
    from mcap.reader import make_reader
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            yield (channel.topic,
                   schema.name if schema else "<none>",
                   channel.message_encoding,
                   message.log_time)


def main(argv=None):
    p = argparse.ArgumentParser(description="inspect an MCAP file")
    p.add_argument("--path", required=True)
    p.add_argument("--gap-factor", type=float, default=1.5)
    p.add_argument("--json", action="store_true")
    p.add_argument("--summary-only", action="store_true",
                   help="cheap pass: read the summary section, skip messages")
    args = p.parse_args(argv)

    cheap = cheap_summary(args.path)
    if args.summary_only:
        if cheap is None:
            print("no summary section (was writer.finish() called?)", file=sys.stderr)
            return 2
        print(json.dumps(cheap, indent=2) if args.json else cheap)
        return 0

    stats = build_topic_stats(iter_records(args.path), gap_factor=args.gap_factor)
    if args.json:
        print(json.dumps({"path": args.path, "summary": cheap, "topics": stats}, indent=2))
    else:
        if cheap:
            print(f"  [summary] {cheap['message_count']} messages, "
                  f"{cheap['chunk_count']} chunk(s), {cheap['duration_s']:.3f}s")
        print(format_report(stats, args.path))

    return 1 if any(t["flagged"] or t["non_monotonic"] for t in stats) else 0


if __name__ == "__main__":
    sys.exit(main())