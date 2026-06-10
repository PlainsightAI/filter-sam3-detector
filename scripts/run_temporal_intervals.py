#!/usr/bin/env python3
"""
Standalone script to run temporal interval detection on a video file.

This script demonstrates the temporal interval detection system using SAM3
for object detection and EMA-based temporal smoothing.

EFFICIENT MULTI-PROMPT DETECTION:
This script uses the `text_prompts` feature for efficient parallel detection
of multiple classes. Instead of running the full SAM3 pipeline N times for
N prompts (O(N) image encodings), it:
  1. Encodes the image ONCE (expensive backbone pass)
  2. For each prompt, encodes text and runs grounding (reusing cached image features)

This provides significant speedup when detecting multiple classes.

Usage:
    python scripts/run_temporal_intervals.py VIDEO_PATH [OPTIONS]

Examples:
    # Basic usage with default prompts
    python scripts/run_temporal_intervals.py video.mp4

    # Custom prompts (all detected efficiently in single pass per frame)
    python scripts/run_temporal_intervals.py video.mp4 --prompts "person,car,dog"

    # Adjust detection parameters
    python scripts/run_temporal_intervals.py video.mp4 --half-life 10 --threshold 0.5

    # Process every 4th frame for faster processing
    python scripts/run_temporal_intervals.py video.mp4 --sample-every 4

    # Output to JSON file
    python scripts/run_temporal_intervals.py video.mp4 --output results.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2


def run_temporal_detection(
    video_path: Path,
    prompts: list[str],
    half_life: float = 5.0,
    presence_threshold: float = 0.4,
    confidence_threshold: float = 0.3,
    max_frames: int | None = None,
    sample_every: int = 1,
    output_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run temporal interval detection on a video file.

    Args:
        video_path: Path to video file
        prompts: List of text prompts for SAM3 detection
        half_life: EMA half-life in frames (how quickly signal responds)
        presence_threshold: Threshold for presence detection (0-1)
        confidence_threshold: Minimum confidence for SAM3 detections
        max_frames: Maximum frames to process (None = all)
        sample_every: Process every Nth frame
        output_path: Optional path to write JSON output
        verbose: Print progress updates

    Returns:
        Dict with intervals, metadata, and detection statistics
    """
    # Import here to avoid import errors if dependencies missing
    from openfilter.filter_runtime.frame import Frame

    from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig
    from filter_sam3_detector.temporal_intervals import TemporalIntervalFilter

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if verbose:
        print(f"Video: {video_path}")
        print(
            f"Frames: {total_frames}, FPS: {fps:.1f}, Duration: {total_frames / fps:.1f}s"
        )
        print(f"Prompts: {prompts}")
        print(f"EMA half_life: {half_life}, threshold: {presence_threshold}")
        print()

    # Initialize SAM3 detector with MULTIPLE PROMPTS
    # This uses the new text_prompts feature for efficient parallel detection:
    # - Image features are encoded ONCE
    # - Each prompt runs text encoding + grounding (reusing cached image features)
    # - Much faster than running the full pipeline N times for N prompts
    detector_config = FilterSAM3DetectorConfig()
    detector_config["model_size"] = "large"
    detector_config["text_prompts"] = (
        prompts  # Use text_prompts (list) for parallel detection
    )
    detector_config["confidence_threshold"] = confidence_threshold
    detector_config["output_label"] = "sam3_detections"

    detector = FilterSAM3Detector(detector_config)
    detector.setup(detector_config)

    # Initialize temporal interval filter
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_output = f.name

    interval_config = TemporalIntervalFilter.normalize_config(
        {
            "half_life": half_life,
            "presence_threshold": presence_threshold,
            "detection_key": "detections",
            "label_field": "label",  # Use 'label' field which is set by multi-prompt detection
            "score_field": "score",
            "output_json_path": temp_output,
            "emit_on_complete": True,
        }
    )

    interval_filter = TemporalIntervalFilter(interval_config)
    interval_filter.setup(interval_config)

    # Process video
    frames_processed = 0
    detection_counts = {prompt: 0 for prompt in prompts}
    frame_id = 0

    while True:
        ret, image = cap.read()
        if not ret:
            break

        frame_id += 1

        # Sample every Nth frame
        if frame_id % sample_every != 0:
            continue

        if max_frames and frames_processed >= max_frames:
            break

        frames_processed += 1

        # Run detector ONCE for all prompts (efficient multi-prompt detection)
        # The detector processes all prompts in a single call, reusing cached image features
        detector_frame = Frame(
            image=image.copy(),
            data={"meta": {"id": frame_id, "ts": frame_id / fps}},
            format="BGR",
        )

        detector_output = detector.process({"video": detector_frame})
        detected_frame = detector_output.get("video", detector_frame)

        # All detections from all prompts are returned in the canonical detections schema
        dets_payload = detected_frame.data.get("detections", {})
        all_detections = (
            dets_payload.get("items", [])
            if isinstance(dets_payload, dict)
            else (dets_payload if isinstance(dets_payload, list) else [])
        )
        if not all_detections:
            all_detections = detected_frame.data.get("meta", {}).get(
                "sam3_detections", []
            )

        # Count detections by class (set by multi-prompt detection)
        for det in all_detections:
            label = det.get("label", det.get("class", "unknown"))
            if label in detection_counts:
                detection_counts[label] += 1

        # Feed to temporal filter
        combined_frame = Frame(
            image=image,
            data={
                "detections": {"items": all_detections},
                "meta": {"id": frame_id, "ts": frame_id / fps},
            },
            format="BGR",
        )
        interval_filter.process({"video": combined_frame})

        # Progress
        if verbose and frames_processed % 10 == 0:
            print(f"  Processed {frames_processed} frames...")

    cap.release()

    # Finalize
    interval_filter.shutdown()
    detector.shutdown()

    # Read results
    with open(temp_output) as f:
        output = json.load(f)

    Path(temp_output).unlink()

    results = {
        "intervals": output["intervals"],
        "metadata": output["metadata"],
        "video_info": {
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": total_frames / fps,
        },
        "processing_info": {
            "frames_processed": frames_processed,
            "sample_every": sample_every,
            "prompts": prompts,
        },
        "detection_counts": detection_counts,
    }

    # Write output if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults written to: {output_path}")

    return results


def print_results(results: dict):
    """Print results in a readable format."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    vi = results["video_info"]
    pi = results["processing_info"]
    print(f"Video: {vi['path']}")
    print(
        f"Duration: {vi['duration_seconds']:.1f}s ({vi['total_frames']} frames @ {vi['fps']:.1f} fps)"
    )
    print(f"Processed: {pi['frames_processed']} frames (every {pi['sample_every']})")

    print("\nDetection counts per label:")
    for label, count in results["detection_counts"].items():
        print(f"  {label}: {count}")

    intervals = results["intervals"]
    print(f"\nTemporal intervals ({len(intervals)} total):")
    for interval in sorted(intervals, key=lambda x: (x["label"], x["start_frame"])):
        duration = interval["end_frame"] - interval["start_frame"]
        status = "PRESENT" if interval["present"] else "absent"
        print(
            f"  [{interval['start_frame']:4d} - {interval['end_frame']:4d}] "
            f"{interval['label']:15s} {status:7s} (conf: {interval['confidence']:.3f}, "
            f"duration: {duration} frames)"
        )

    print("\nMetadata:")
    for key, value in results["metadata"].items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Run temporal interval detection on a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to video file",
    )
    parser.add_argument(
        "--prompts",
        "-p",
        type=str,
        default="person,hand,object",
        help="Comma-separated list of detection prompts (default: person,hand,object)",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=5.0,
        help="EMA half-life in frames (default: 5.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Presence threshold 0-1 (default: 0.4)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="SAM3 confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    prompts = [p.strip() for p in args.prompts.split(",")]

    try:
        results = run_temporal_detection(
            video_path=args.video,
            prompts=prompts,
            half_life=args.half_life,
            presence_threshold=args.threshold,
            confidence_threshold=args.confidence,
            max_frames=args.max_frames,
            sample_every=args.sample_every,
            output_path=args.output,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print_results(results)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
