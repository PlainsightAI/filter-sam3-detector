#!/usr/bin/env python

"""
Example script for running object detection with FilterSAM3Detector using
**exemplar (reference) images** in addition to a text prompt.

Same pipeline as filter_object_detection.py (VideoIn -> SAM3Detector -> Webvis),
but configures ref_images and ref_images_negative so SAM3 can use visual examples
to improve detection. Reference images are pasted beside the frame and used as
positive/negative visual prompts.

Required environment variables in .env:
    VIDEO_PATH: Path to the input video file
    FILTER_TEXT_PROMPT: Text prompt for detection (e.g., "avocado", "ripe peach")
    FILTER_REF_IMAGES: Path(s) to positive reference images. Can be:
        - A single directory path (all .jpg/.png inside are used)
        - Comma-separated file or directory paths

Optional:
    FILTER_REF_IMAGES_NEGATIVE: Path(s) to negative reference images (what to exclude)
    FILTER_DEVICE: cuda, cpu, mps - default: cuda
    FILTER_CONFIDENCE_THRESHOLD: 0.0-1.0 - default: 0.5
    FILTER_MAX_DETECTIONS: Max detections per frame - default: 100
    FILTER_VISUALIZE: true/false - default: false
    FILTER_OUTPUT_DIR: Output directory - default: ./output

Example .env:
    VIDEO_PATH=/path/to/video.mp4
    FILTER_TEXT_PROMPT=avocado in salad
    FILTER_REF_IMAGES=/path/to/ref_images/
    FILTER_REF_IMAGES_NEGATIVE=/path/to/ref_images_negative/
    FILTER_DEVICE=cuda
    FILTER_VISUALIZE=true
    FILTER_OUTPUT_DIR=./results_exemplar
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


def _parse_ref_paths(env_value: str):
    """Parse FILTER_REF_IMAGES or FILTER_REF_IMAGES_NEGATIVE into a list of paths, or None if empty."""
    if not env_value or not env_value.strip():
        return None
    paths = [p.strip() for p in env_value.split(",") if p.strip()]
    return paths if paths else None


if __name__ == '__main__':
    video_path = os.getenv("VIDEO_PATH", "")
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'

    ref_images = _parse_ref_paths(os.getenv('FILTER_REF_IMAGES', ''))
    ref_images_negative = _parse_ref_paths(os.getenv('FILTER_REF_IMAGES_NEGATIVE', ''))

    if not video_path:
        print("Error: VIDEO_PATH is required")
        exit(1)
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        exit(1)

    print(f"Using VideoIn with path: {video_path} (loop)")
    print(f"Text prompt: {os.getenv('FILTER_TEXT_PROMPT', 'NOT SET')}")
    print(f"Ref images (positive): {ref_images}")
    neg_str = ref_images_negative if ref_images_negative else "(none) — FILTER_REF_IMAGES_NEGATIVE not used"
    print(f"Ref images (negative): {neg_str}")
    print(f"Device: {os.getenv('FILTER_DEVICE', 'cuda')}")
    print(f"Confidence threshold: {os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.5')}")
    print(f"Max detections: {os.getenv('FILTER_MAX_DETECTIONS', '100')}")
    print(f"Visualize: {visualize}")
    print(f"Output directory: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Detector config with exemplar refs explicitly set (normalize_config can still override from env)
    detector_config = FilterSAM3DetectorConfig(
        id="filter_sam3_detector",
        sources="tcp://localhost:5550",
        outputs="tcp://*:5552",
        output_label="detections",
        output_path=str(output_path / "detections.jsonl"),
        frames_output_dir=str(output_path / "frames"),
        ref_images=ref_images,
    )

    filters = [
        (
            VideoIn,
            dict(
                sources=f"file://{video_path}!loop",
                outputs="tcp://*:5550",
            ),
        ),
        (FilterSAM3Detector, detector_config),
        (Webvis, dict(sources="tcp://localhost:5552")),
    ]

    print("\nStarting pipeline (exemplar mode)...")
    print(f"Results will be saved to: {output_path}")
    print(f"  - detections.jsonl, frames/")
    Filter.run_multi(filters)
