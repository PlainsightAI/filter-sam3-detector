#!/usr/bin/env python

"""
Example script for running object detection with FilterSAM3Detector (text prompts).

Pipeline: VideoIn -> SAM3Detector -> Webvis.

Required in .env:
    VIDEO_PATH: Path to the input video file
    FILTER_TEXT_PROMPT: Text prompt (e.g., "person", "car")

Optional:
    FILTER_DEVICE, FILTER_CONFIDENCE_THRESHOLD, FILTER_MAX_DETECTIONS, FILTER_VISUALIZE, FILTER_OUTPUT_DIR
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openfilter.filter_runtime.filter import Filter
from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis


if __name__ == '__main__':
    video_path = os.getenv('VIDEO_PATH', '')
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'

    if not video_path:
        print("Error: VIDEO_PATH is required. Set it in .env")
        exit(1)
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        exit(1)

    print(f"Using VideoIn: {video_path}")
    print(f"Text prompt: {os.getenv('FILTER_TEXT_PROMPT', 'NOT SET')}")
    print(f"Device: {os.getenv('FILTER_DEVICE', 'cuda')}")
    print(f"Visualize: {visualize}")
    print(f"Output: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    filters = [
        (
            VideoIn,
            dict(
                sources=f"file://{video_path}!sync",
                outputs="tcp://*:5550",
            ),
        ),
        (
            FilterSAM3Detector,
            FilterSAM3DetectorConfig(
                id="filter_sam3_detector",
                sources="tcp://localhost:5550",
                outputs="tcp://*:5552",
                output_label="detections",
                output_path=str(output_path / "detections.jsonl"),
                frames_output_dir=str(output_path / "frames"),
            ),
        ),
        (
            Webvis,
            dict(sources="tcp://localhost:5552", port=9000),
        ),
    ]

    print(f"\nStarting pipeline... Results in {output_path} (detections.jsonl, frames/)")
    Filter.run_multi(filters)
