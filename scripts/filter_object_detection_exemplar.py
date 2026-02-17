#!/usr/bin/env python

"""
Example script for running object detection with FilterSAM3Detector using
**reference boxes** (positive/negative bboxes on the original image). Exemplar = ref boxes only (no exemplars_path).

Same pipeline as filter_object_detection.py (VideoIn -> SAM3Detector -> Webvis),
but configures positive_boxes and negative_boxes so SAM3 uses geometric prompts:
boxes in [x, y, w, h] pixel coordinates on the frame.

Required environment variables in .env:
    VIDEO_PATH: Path to the input video file
    FILTER_TEXT_PROMPT: Text prompt for detection (e.g., "person", "car"). Required when not using ref boxes; optional when using FILTER_POSITIVE_BOXES/FILTER_NEGATIVE_BOXES.

Optional:
    FILTER_POSITIVE_BOXES: JSON array of [x, y, w, h] boxes (positive prompts), e.g. '[[480,290,110,360]]'
    FILTER_NEGATIVE_BOXES: JSON array of [x, y, w, h] boxes (negative prompts), e.g. '[[100,100,50,200]]'
    FILTER_DEVICE: cuda, cpu, mps - default: cuda
    FILTER_CONFIDENCE_THRESHOLD: 0.0-1.0 - default: 0.5
    FILTER_MAX_DETECTIONS: Max detections per frame - default: 100
    FILTER_VISUALIZE: true/false - default: false
    FILTER_VIZ_TOPIC: when set (e.g. viz), main=original+meta, this topic=drawn frame+meta - default: unset
    FILTER_OUTPUT_DIR: Output directory - default: ./output

Example .env:
    VIDEO_PATH=/path/to/video.mp4
    FILTER_TEXT_PROMPT=person
    FILTER_POSITIVE_BOXES='[[480,290,110,360]]'
    FILTER_NEGATIVE_BOXES='[[100,100,50,200]]'
    FILTER_DEVICE=cuda
    FILTER_VISUALIZE=true
    FILTER_OUTPUT_DIR=./results_exemplar
"""

import os
import json
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


def _parse_boxes_env(env_value: str):
    """Parse FILTER_POSITIVE_BOXES or FILTER_NEGATIVE_BOXES from JSON to list of [x,y,w,h], or None if empty."""
    if not env_value or not env_value.strip():
        return None
    try:
        raw = json.loads(env_value.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            out.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
    return out if out else None


if __name__ == '__main__':
    video_path = os.getenv("VIDEO_PATH", "")
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'

    positive_boxes = _parse_boxes_env(os.getenv('FILTER_POSITIVE_BOXES', ''))
    negative_boxes = _parse_boxes_env(os.getenv('FILTER_NEGATIVE_BOXES', ''))

    print(f"Using VideoIn with path: {video_path} (loop)")
    print(f"Text prompt: {os.getenv('FILTER_TEXT_PROMPT', 'NOT SET')}")
    print(f"Positive boxes: {positive_boxes if positive_boxes else '(none)'}")
    print(f"Negative boxes: {negative_boxes if negative_boxes else '(none)'}")
    print(f"Device: {os.getenv('FILTER_DEVICE', 'cuda')}")
    print(f"Confidence threshold: {os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.5')}")
    print(f"Max detections: {os.getenv('FILTER_MAX_DETECTIONS', '100')}")
    print(f"Visualize: {visualize}")
    print(f"Output directory: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    detector_config = FilterSAM3DetectorConfig(
        id="filter_sam3_detector",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        output_label="detections",
        output_path=str(output_path / "detections.jsonl"),
        frames_output_dir=str(output_path / "frames"),
        positive_boxes=positive_boxes or [],
        negative_boxes=negative_boxes or [],
    )

    filters = [
        (
            VideoIn,
            dict(
                sources=f"file://{video_path}!sync!resize=960x540",
                outputs="tcp://*:5550",
            ),
        ),
        (FilterSAM3Detector, detector_config),
        (
            Webvis, dict(sources="tcp://localhost:5552",
            port=9000,
        )),
    ]

    mode = "reference-boxes" if (positive_boxes or negative_boxes) else "text-prompt only"
    print(f"\nStarting pipeline ({mode})...")
    print(f"Results will be saved to: {output_path}")
    print(f"  - detections.jsonl, frames/")
    Filter.run_multi(filters)
