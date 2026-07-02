#!/usr/bin/env python3
"""
Example: Detect objects in video using SAM3 with natural language prompts.

This demonstrates a pipeline for object detection in video:
- VideoIn: Stream video frames with optional resizing
- FilterSAM3Detector: Detect objects using text prompts, save JSONL + frames

The detector uses its built-in output_path (JSONL) and frames_output_dir
capabilities. Recorder and ImageOut are openfilter sink filters that expect
file:// outputs, not ZMQ, so they cannot be chained via ZMQ in a pipeline.

Usage:
    python detect_objects_video.py \\
        --video input.mp4 \\
        --prompt "small transparent cup" \\
        --output-dir ./results \\
        --confidence 0.2 \\
        --resize 480

    # Multiple prompts (detect several object types at once)
    python detect_objects_video.py \\
        --video input.mp4 \\
        --prompt "cup" "bowl" "plate" \\
        --output-dir ./results

    # Multiple prompts (repeated --prompt flags)
    python detect_objects_video.py \\
        --video input.mp4 \\
        --prompt "cup" --prompt "bowl" --prompt "plate" \\
        --output-dir ./results

    # Multiple videos
    python detect_objects_video.py \\
        --video video1.mp4 video2.mp4 video3.mp4 \\
        --prompt "person" \\
        --output-dir ./detections

    # With exemplar images instead of text
    python detect_objects_video.py \\
        --video input.mp4 \\
        --exemplars ./cup_examples/ \\
        --output-dir ./results
"""

import argparse
from pathlib import Path

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from filter_sam3_detector import FilterSAM3Detector


def main():
    parser = argparse.ArgumentParser(description="Detect objects in video using SAM3")
    parser.add_argument("--video", nargs="+", required=True, help="Input video file(s)")
    parser.add_argument(
        "--prompt",
        nargs="+",
        action="extend",
        help="Text prompt(s) for detection; repeatable "
        "(e.g., --prompt 'cup' 'bowl' or --prompt 'cup' --prompt 'bowl')",
    )
    parser.add_argument("--exemplars", help="Directory with exemplar images")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--resize", type=int, help="Resize max dimension (e.g., 480 for 480p)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=1, help="Process every Nth frame"
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu, mps")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save annotated frames with bboxes drawn",
    )
    args = parser.parse_args()

    if not args.prompt and not args.exemplars:
        parser.error("Must provide either --prompt or --exemplars")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Build video sources with topics
    video_sources = []
    for i, video_path in enumerate(args.video):
        source = f"file://{Path(video_path).absolute()}"
        if args.resize:
            source += f"!maxsize={args.resize}x{args.resize}"
        video_sources.append(f"{source};video{i}")

    # Define pipeline: VideoIn → FilterSAM3Detector
    # The detector writes JSONL and frames directly via output_path / frames_output_dir.
    # No Recorder or ImageOut needed (those are sink filters that expect file:// outputs,
    # not ZMQ addresses, and cannot be chained in a ZMQ pipeline).
    frames_dir = str(output_dir / "frames")
    jsonl_path = str(output_dir / "detections.jsonl")

    filters = [
        # Input: Stream video frames
        (
            VideoIn,
            {
                "sources": ",".join(video_sources),
                "outputs": ["tcp://127.0.0.1:5555"],
            },
        ),
        # Detect objects with SAM3 and write results to disk.
        # text_prompts accepts a list (single- or multi-element); normalize_config
        # treats list inputs as raw prompts, so no per-length dispatch is needed.
        (
            FilterSAM3Detector,
            {
                "sources": "tcp://127.0.0.1:5555",
                **({"text_prompts": args.prompt} if args.prompt else {}),
                "exemplars_path": args.exemplars,
                "confidence_threshold": args.confidence,
                "device": args.device,
                "output_label": "detections",
                "output_path": jsonl_path,
                "frames_output_dir": frames_dir,
                **({"save_annotated_frames": True} if args.visualize else {}),
            },
        ),
    ]

    print(f"Processing {len(args.video)} video(s)...")
    print(f"Prompt: {', '.join(args.prompt) if args.prompt else 'exemplars'}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {args.confidence}")

    # Run the pipeline using Filter.run_multi
    Filter.run_multi(filters)

    print(f"\nDone! Results saved to {output_dir}")
    print("  - detections.jsonl: Frame-by-frame detections")
    print("  - frames/: Saved frames")


if __name__ == "__main__":
    main()
