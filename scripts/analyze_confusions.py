#!/usr/bin/env python3
"""
Analyze cross-class detection overlaps from a SAM3 detector JSONL output file.

This script reads the detections JSONL written by FilterSAM3Detector and reports
which prompt pairs overlap, how often, and what to do about it.

If the JSONL was produced with ``confusion_detection_enabled`` active (i.e. with
multiple text prompts), per-frame ``meta.confusions`` records are used directly.
Otherwise the script re-computes IoU from raw ``meta.detections`` — so it works
on any historical JSONL.

Usage
-----
    python scripts/analyze_confusions.py detections.jsonl
    python scripts/analyze_confusions.py detections.jsonl --iou-threshold 0.7
    python scripts/analyze_confusions.py detections.jsonl --format json --output report.json
    python scripts/analyze_confusions.py detections.jsonl --min-frames 5

Exit code: 0 if no pairs exceed the configured thresholds; 1 otherwise.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from filter_sam3_detector.utils.bbox import to_xyxy
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# IoU (self-contained — no filter dependency)
# ---------------------------------------------------------------------------


def _compute_iou(box_a: list, box_b: list) -> float:
    """AABB IoU for [x1, y1, x2, y2] boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _get_class(det: dict) -> str:
    return det.get("class") or det.get("class_name") or det.get("label") or "unknown"


def _detection_strength(det: dict) -> float:
    """Prefer ``score`` if present, else ``confidence``, else 0.0.

    Uses explicit ``None`` checks so ``0.0`` is not treated as missing (unlike ``or``).
    Matches ``ConfusionDetector._detection_strength``.
    """
    s = det.get("score")
    if s is not None:
        return float(s)
    c = det.get("confidence")
    if c is not None:
        return float(c)
    return 0.0


# ---------------------------------------------------------------------------
# Per-frame confusion extraction
# ---------------------------------------------------------------------------


def _confusions_from_record(record: dict, iou_threshold: float) -> list[dict]:
    """Return confusion events for one JSONL record.

    Uses pre-computed ``meta.confusions`` when available (and IoU gate matches);
    otherwise re-computes from ``meta.detections``.
    """
    data = record.get("data", record)
    meta = data.get("meta", {})
    frame_id = data.get("id") or meta.get("id") or meta.get("frame_id")

    # Try pre-computed confusions first
    pre = meta.get("confusions")
    if isinstance(pre, list) and pre:
        # Re-filter by the requested iou_threshold (may differ from stored threshold)
        filtered = [
            c for c in pre if isinstance(c, dict) and c.get("iou", 0.0) >= iou_threshold
        ]
        for c in filtered:
            c.setdefault("frame_id", frame_id)
        return filtered

    # Re-compute from raw detections
    from filter_sam3_detector.utils.detections import extract_items

    dets = extract_items(data)

    if not dets:
        return []

    by_class: dict[str, list] = defaultdict(list)
    for d in dets:
        if isinstance(d, dict):
            by_class[_get_class(d)].append(d)

    if len(by_class) < 2:
        return []

    confusions = []
    for class_a, class_b in combinations(by_class.keys(), 2):
        for det_a in by_class[class_a]:
            box_a = to_xyxy(det_a)
            if box_a is None:
                continue
            score_a = _detection_strength(det_a)
            for det_b in by_class[class_b]:
                box_b = to_xyxy(det_b)
                if box_b is None:
                    continue
                score_b = _detection_strength(det_b)
                iou = _compute_iou(box_a, box_b)
                if iou >= iou_threshold:
                    confusions.append(
                        {
                            "frame_id": frame_id,
                            "prompt_a": class_a,
                            "prompt_b": class_b,
                            "iou": round(iou, 4),
                            "score_a": round(score_a, 4),
                            "score_b": round(score_b, 4),
                        }
                    )

    return confusions


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_stats(jsonl_path: Path, iou_threshold: float) -> dict:
    """Stream JSONL and aggregate per-pair confusion statistics."""
    total_frames = 0
    frames_with_confusion = 0

    # pair_key -> stats
    pair_stats: dict[tuple, dict] = {}

    with open(jsonl_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_frames += 1
            confusions = _confusions_from_record(record, iou_threshold)
            if not confusions:
                continue

            frames_with_confusion += 1
            data = record.get("data", record)
            frame_id = data.get("id") or data.get("meta", {}).get("id")

            # One entry per confusion *event* (overlapping pair of boxes). A single frame can
            # emit multiple events for the same prompt pair, so avg_iou / avg_score_* are means
            # over events — unlike frames_with_confusion below, which is de-duped per frame.
            for c in confusions:
                pa, pb = c.get("prompt_a", "?"), c.get("prompt_b", "?")
                key = (pa, pb)
                if key not in pair_stats:
                    pair_stats[key] = {
                        "prompt_a": pa,
                        "prompt_b": pb,
                        "frames_with_confusion": 0,
                        "iou_values": [],
                        "score_a_values": [],
                        "score_b_values": [],
                        "example_frame_ids": [],
                    }
                stats = pair_stats[key]
                stats["iou_values"].append(c.get("iou", 0.0))
                stats["score_a_values"].append(c.get("score_a", 0.0))
                stats["score_b_values"].append(c.get("score_b", 0.0))
                if (
                    len(stats["example_frame_ids"]) < 5
                    and frame_id not in stats["example_frame_ids"]
                ):
                    stats["example_frame_ids"].append(frame_id)
            # Count frame once per pair
            seen_pairs: set = set()
            for c in confusions:
                pair_key = (c.get("prompt_a", "?"), c.get("prompt_b", "?"))
                if pair_key not in seen_pairs:
                    pair_stats[pair_key]["frames_with_confusion"] += 1
                    seen_pairs.add(pair_key)

    # Compute derived stats
    results = []
    for key, stats in pair_stats.items():
        n = stats["frames_with_confusion"]
        rate = n / total_frames if total_frames > 0 else 0.0
        iou_vals = stats["iou_values"]
        sa_vals = stats["score_a_values"]
        sb_vals = stats["score_b_values"]
        results.append(
            {
                "prompt_a": stats["prompt_a"],
                "prompt_b": stats["prompt_b"],
                "frames_with_confusion": n,
                "confusion_rate": round(rate, 4),
                "avg_iou": round(sum(iou_vals) / len(iou_vals), 4) if iou_vals else 0.0,
                "max_iou": round(max(iou_vals), 4) if iou_vals else 0.0,
                "avg_score_a": round(sum(sa_vals) / len(sa_vals), 4)
                if sa_vals
                else 0.0,
                "avg_score_b": round(sum(sb_vals) / len(sb_vals), 4)
                if sb_vals
                else 0.0,
                "example_frame_ids": stats["example_frame_ids"],
                "suggestion": _suggest(
                    stats["prompt_a"],
                    stats["prompt_b"],
                    rate,
                    sum(iou_vals) / len(iou_vals) if iou_vals else 0.0,
                    sum(sa_vals) / len(sa_vals) if sa_vals else 0.0,
                    sum(sb_vals) / len(sb_vals) if sb_vals else 0.0,
                ),
            }
        )

    # Sort by confusion_rate descending
    results.sort(key=lambda x: x["confusion_rate"], reverse=True)

    return {
        "total_frames": total_frames,
        "frames_with_confusion": frames_with_confusion,
        "confusion_rate": round(frames_with_confusion / total_frames, 4)
        if total_frames > 0
        else 0.0,
        "iou_threshold": iou_threshold,
        "prompt_pairs": results,
    }


# ---------------------------------------------------------------------------
# Resolution suggestions
# ---------------------------------------------------------------------------


def _suggest(
    pa: str, pb: str, rate: float, avg_iou: float, avg_sa: float, avg_sb: float
) -> str:
    # Higher average score for a prompt ⇒ that prompt "wins" the overlap comparison.
    winner, loser = (pa, pb) if avg_sa >= avg_sb else (pb, pa)
    loser_score = min(avg_sa, avg_sb)
    winner_score = max(avg_sa, avg_sb)

    if avg_iou > 0.85 and rate > 0.5:
        return (
            f"Near-identical regions (avg_iou={avg_iou:.2f}, rate={rate:.1%}). "
            f"Merge '{pa}' and '{pb}' into one prompt (e.g. '{pa} or {pb}'), "
            f"or drop '{loser}' (avg_score={loser_score:.3f} < '{winner}' avg_score={winner_score:.3f})."
        )
    if avg_iou > 0.70 and rate > 0.3:
        return (
            f"Frequent overlap (avg_iou={avg_iou:.2f}, rate={rate:.1%}). "
            f"Add negative reference examples to distinguish '{pa}' from '{pb}', "
            f"or raise confidence_threshold for '{loser}' (avg_score={loser_score:.3f})."
        )
    if rate > 0.1:
        return (
            f"Moderate overlap rate ({rate:.1%}, avg_iou={avg_iou:.2f}). "
            f"Consider raising confidence_threshold for '{loser}' (avg_score={loser_score:.3f}) "
            "to reduce duplicate detections."
        )
    return (
        f"Low overlap rate ({rate:.1%}, avg_iou={avg_iou:.2f}). "
        "Acceptable — monitor or raise FILTER_CONFUSION_IOU_THRESHOLD if this is expected."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _recommendations(stats: dict) -> list[str]:
    recs = []
    pairs = stats["prompt_pairs"]
    bad = [p for p in pairs if p["confusion_rate"] > 0.1]
    if bad:
        recs.append(
            f"{len(bad)} prompt pair(s) exceed 10% confusion rate — review before production use."
        )
    if stats["iou_threshold"] > 0.7:
        recs.append(
            "Re-run with --iou-threshold 0.7 (or lower) to include partial cross-class overlaps."
        )
    recs.append(
        "Default filter gate is 0.95; lower FILTER_CONFUSION_IOU_THRESHOLD only if you "
        "intentionally want to merge looser cross-class boxes."
    )
    if stats["frames_with_confusion"] > 0:
        recs.append(
            "Set FILTER_REMOVE_OVERLAP=true to automatically keep the highest-confidence "
            "class per overlapping pair at shutdown."
        )
    return recs


def format_text(stats: dict, jsonl_path: str) -> str:
    lines = []
    lines.append("=== Confusion Analysis Report ===")
    lines.append(f"File   : {jsonl_path}")
    n, total = stats["frames_with_confusion"], stats["total_frames"]
    rate_pct = stats["confusion_rate"] * 100
    lines.append(f"Frames : {total} analyzed  |  {n} with confusion ({rate_pct:.1f}%)")
    lines.append(f"IoU threshold: {stats['iou_threshold']}")

    if not stats["prompt_pairs"]:
        lines.append("\nNo cross-class overlaps detected.")
    else:
        lines.append("")
        for pair in stats["prompt_pairs"]:
            pa, pb = pair["prompt_a"], pair["prompt_b"]
            lines.append(f"{pa}  vs  {pb}")
            conf_rate_pct = pair["confusion_rate"] * 100
            lines.append(
                f"  Confusion rate : {conf_rate_pct:.1f}%  "
                f"({pair['frames_with_confusion']} / {total} frames)"
            )
            lines.append(
                f"  Avg IoU        : {pair['avg_iou']:.2f}   |  Max IoU: {pair['max_iou']:.2f}"
            )
            examples = ", ".join(str(f) for f in pair["example_frame_ids"])
            lines.append(f"  Example frames : {examples}")
            # Wrap suggestion at 80 chars
            suggestion = pair["suggestion"]
            lines.append(f"  Suggestion     : {suggestion[:77]}")
            if len(suggestion) > 77:
                for chunk in [
                    suggestion[i : i + 77] for i in range(77, len(suggestion), 77)
                ]:
                    lines.append(f"                   {chunk}")
            lines.append("")

    recs = _recommendations(stats)
    if recs:
        lines.append("=== Recommendations ===")
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cross-class detection overlaps from a SAM3 JSONL output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("jsonl", type=Path, help="Path to detections.jsonl")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.95,
        help="IoU threshold for overlap detection (default: 0.95)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Only report pairs appearing in >= N frames (default: 1)",
    )

    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: file not found: {args.jsonl}", file=sys.stderr)
        return 2

    stats = compute_stats(args.jsonl, args.iou_threshold)

    # Filter by min-frames
    stats["prompt_pairs"] = [
        p
        for p in stats["prompt_pairs"]
        if p["frames_with_confusion"] >= args.min_frames
    ]

    if args.format == "json":
        output = json.dumps(stats, indent=2)
    else:
        output = format_text(stats, str(args.jsonl))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Report written to: {args.output}")
    else:
        print(output)

    # Exit 1 if any pair exceeds 10% confusion rate
    bad = [p for p in stats["prompt_pairs"] if p["confusion_rate"] > 0.1]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
