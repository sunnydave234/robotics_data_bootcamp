from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import statistics

DEFAULT_WEIGHTS = {"schema": 0.25, "sensors": 0.25, "sync": 0.25, "action": 0.25}


@dataclass
class EpisodeResult:
    """episode_id is the GLOBAL episode_index value -- never a loop position."""
    episode_id: int
    schema_ok: bool
    sensor_coverage: float
    sync_ok: bool
    action_error: Optional[float]
    action_error_ceiling: float = 0.15


def episode_health_score_naive(ep: EpisodeResult, weights: dict = None) -> tuple[float, dict]:
    weights = weights or DEFAULT_WEIGHTS
    action_component = 0.0
    if ep.action_error is not None:
        action_component = max(0.0, 1.0 - min(ep.action_error / ep.action_error_ceiling, 1.0))
    components = {
        "schema": 1.0 if ep.schema_ok else 0.0,
        "sensors": ep.sensor_coverage,
        "sync": 1.0 if ep.sync_ok else 0.0,
        "action": action_component,
    }
    score = sum(weights[k] * v for k, v in components.items())
    return score, components


def episode_health_score_gated(ep: EpisodeResult, weights: dict = None) -> tuple[float, dict, str]:
    weights = weights or DEFAULT_WEIGHTS
    if not ep.schema_ok:
        return 0.0, {"schema": 0.0, "sensors": None, "sync": None, "action": None}, "excluded"

    decision = "insufficient_data" if ep.action_error is None else "scored"

    action_component = None
    if ep.action_error is not None:
        action_component = max(0.0, 1.0 - min(ep.action_error / ep.action_error_ceiling, 1.0))

    have = {"schema": 1.0, "sensors": ep.sensor_coverage, "sync": 1.0 if ep.sync_ok else 0.0}
    if action_component is not None:
        have["action"] = action_component
    active_weights = {k: weights[k] for k in have}
    total_w = sum(active_weights.values())
    score = sum(active_weights[k] * have[k] for k in have) / total_w

    flagged = (not ep.sync_ok) or (ep.sensor_coverage < 1.0) or (
        ep.action_error is not None and score < 0.8
    )
    if decision == "scored" and flagged:
        decision = "flagged"
    elif decision == "scored":
        decision = "included"

    return score, {"schema": 1.0, "sensors": ep.sensor_coverage,
                    "sync": 1.0 if ep.sync_ok else 0.0, "action": action_component}, decision


def build_health_table(results: list[EpisodeResult], scorer=episode_health_score_gated):
    rows = []
    for ep in results:
        out = scorer(ep)
        if len(out) == 3:
            score, components, decision = out
        else:
            score, components = out
            decision = "excluded" if not ep.schema_ok else ("flagged" if score < 0.8 else "included")
        rows.append({
            "episode_id": ep.episode_id, "score": round(score, 4), "decision": decision,
            "schema_ok": ep.schema_ok, "sensor_coverage": ep.sensor_coverage,
            "sync_ok": ep.sync_ok, "action_error": ep.action_error,
        })
    return rows


def generate_report(rows: list[dict], repo_id: str, ep_range: str) -> str:
    n = len(rows)
    scored_rows = [r for r in rows if r["decision"] != "excluded"]
    mean_score = statistics.mean(r["score"] for r in scored_rows) if scored_rows else 0.0

    issues = []
    schema_fail_n = sum(1 for r in rows if not r["schema_ok"])
    degenerate_n = sum(1 for r in rows if r["sensor_coverage"] < 1.0)
    sync_fail_n = sum(1 for r in rows if not r["sync_ok"])
    action_outlier_n = sum(1 for r in rows if r["decision"] == "flagged"
                            and r["action_error"] is not None
                            and r["score"] < 0.8
                            and r["sync_ok"] and r["sensor_coverage"] >= 1.0)
    insufficient_n = sum(1 for r in rows if r["decision"] == "insufficient_data")

    if schema_fail_n:
        issues.append((schema_fail_n, f"Schema failures — {schema_fail_n} episodes (excluded)"))
    if degenerate_n:
        issues.append((degenerate_n, f"Degenerate/missing sensor coverage — {degenerate_n} episodes"))
    if sync_fail_n:
        issues.append((sync_fail_n, f"Sync drift/drop/duplicate flagged — {sync_fail_n} episodes"))
    if action_outlier_n:
        issues.append((action_outlier_n, f"High action-following error — {action_outlier_n} episodes"))
    if insufficient_n:
        issues.append((insufficient_n, f"Insufficient data to score (trajectory check skipped) — {insufficient_n} episodes"))
    issues.sort(key=lambda x: -x[0])

    worst5 = sorted(scored_rows, key=lambda r: r["score"])[:5]

    lines = [f"# Week 1 Dataset Health Report — {repo_id}\n",
             f"Episodes checked: {n}  (range {ep_range})\n",
             f"Mean health score (excludes hard-excluded episodes): {mean_score:.2f}\n",
             "\n## Top issues found\n"]
    for i, (count, desc) in enumerate(issues, 1):
        lines.append(f"{i}. {desc}\n")
    lines.append("\n## Worst 5 scored episodes\n")
    for r in worst5:
        lines.append(f"- episode {r['episode_id']}: score={r['score']:.2f}, decision={r['decision']}\n")
    excluded = [r["episode_id"] for r in rows if r["decision"] == "excluded"]
    if excluded:
        lines.append("\n## Hard-excluded episodes (schema failure — not eligible for training regardless of other scores)\n")
        lines.append(f"{excluded}\n")
    return "".join(lines)