#!/usr/bin/env python3
"""
Example: Detect objects in video using SAM3 with natural language prompts.

This demonstrates a complete pipeline for object detection in video:
- VideoIn: Stream video frames with optional resizing
- FilterSAM3Detector: Detect objects using text prompts
- Recorder: Export detections to JSONL/COCO format
- ImageOut: Save annotated frames with detections

Usage:
    python detect_objects_video.py \
        --video input.mp4 \
        --prompt "small transparent cup" \
        --output-dir ./results \
        --confidence 0.2 \
        --resize 480

    # Multiple videos
    python detect_objects_video.py \
        --video video1.mp4 video2.mp4 video3.mp4 \
        --prompt "person" \
        --output-dir ./detections

    # With exemplar images instead of text
    python detect_objects_video.py \
        --video input.mp4 \
        --exemplars ./cup_examples/ \
        --output-dir ./results
"""

import argparse
from pathlib import Path

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.recorder import Recorder
from openfilter.filter_runtime.filters.image_out import ImageOut
from filter_sam3_detector import FilterSAM3Detector


def main():
    parser = argparse.ArgumentParser(description="Detect objects in video using SAM3")
    parser.add_argument("--video", nargs="+", required=True, help="Input video file(s)")
    parser.add_argument("--prompt", help="Text prompt for detection (e.g., 'person', 'car')")
    parser.add_argument("--exemplars", help="Directory with exemplar images")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--resize", type=int, help="Resize max dimension (e.g., 480 for 480p)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu, mps")
    parser.add_argument("--visualize", action="store_true", help="Draw bboxes on output frames")
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

    # Define pipeline filters
    filters = [
        # Input: Stream video frames
        (VideoIn, {
            "sources": ",".join(video_sources),
            "outputs": ["tcp://127.0.0.1:5555"],
        }),

        # Detect objects with SAM3
        (FilterSAM3Detector, {
            "sources": "tcp://127.0.0.1:5555",
            "outputs": ["tcp://127.0.0.1:5556"],
            "text_prompt": args.prompt,
            "exemplars_path": args.exemplars,
            "confidence_threshold": args.confidence,
            "device": args.device,
            "visualize": args.visualize,
            "output_label": "detections",
        }),

        # Export detections to JSONL/COCO
        (Recorder, {
            "sources": "tcp://127.0.0.1:5556",
            "outputs": ["tcp://127.0.0.1:5557"],
            "path": str(output_dir / "detections.jsonl"),
            "format": "jsonl",
        }),

        # Save frames with detections
        (ImageOut, {
            "sources": "tcp://127.0.0.1:5557",
            "path": str(output_dir / "frames" / "%05d.jpg"),
            "only_with_detections": True,
        }),
    ]

    print(f"Processing {len(args.video)} video(s)...")
    print(f"Prompt: {args.prompt or 'exemplars'}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {args.confidence}")

    # Run the pipeline using Filter.run_multi
    Filter.run_multi(filters)

    print(f"\nDone! Results saved to {output_dir}")
    print(f"  - detections.jsonl: Frame-by-frame detections")
    print(f"  - frames/: Annotated frames")


if __name__ == "__main__":
    main()
