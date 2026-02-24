#!/usr/bin/env python

"""
Example script for running object detection with FilterSAM3Detector using
**reference boxes** (positive/negative bboxes) and/or **reference images** (REF_IMGS).

Same pipeline as filter_object_detection.py (VideoIn -> SAM3Detector -> Webvis).
- Ref boxes: FILTER_POSITIVE_BOXES / FILTER_NEGATIVE_BOXES (bboxes [x, y, w, h] on the frame).
- Ref images: FILTER_REF_IMAGES / FILTER_REF_IMAGES_NEGATIVE (images pasted on composite). When ref boxes are set, ref images are disabled.

Required in .env:
    VIDEO_PATH: Path to the input video file
    At least one of: FILTER_TEXT_PROMPT, ref boxes (FILTER_POSITIVE_BOXES / FILTER_NEGATIVE_BOXES),
    or ref images (FILTER_REF_IMAGES and/or FILTER_REF_IMAGES_NEGATIVE). FILTER_TEXT_PROMPT is optional
    when using REF_IMGS only (detection is driven by the reference images).

Optional (ref boxes):
    FILTER_POSITIVE_BOXES: JSON array of [x, y, w, h] boxes, e.g. '[[480,290,110,360]]'
    FILTER_NEGATIVE_BOXES: JSON array of [x, y, w, h] boxes, e.g. '[[100,100,50,200]]'

Optional (ref images; ignored if ref boxes are set):
    FILTER_REF_IMAGES: Comma-separated paths or dir (positive ref images)
    FILTER_REF_IMAGES_NEGATIVE: Comma-separated paths or dir (negative ref images)
    FILTER_COMPOSITE_TOPIC: e.g. composite — publish composite image for inspection

Optional (common):
    FILTER_DEVICE, FILTER_CONFIDENCE_THRESHOLD, FILTER_MAX_DETECTIONS, FILTER_VISUALIZE, FILTER_VIZ_TOPIC, FILTER_OUTPUT_DIR
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


def _parse_ref_images_env(env_value: str):
    """Parse FILTER_REF_IMAGES or FILTER_REF_IMAGES_NEGATIVE: comma-separated paths -> list, or None if empty."""
    if not env_value or not env_value.strip():
        return None
    parts = [p.strip() for p in env_value.strip().split(",") if p.strip()]
    return parts if parts else None


if __name__ == '__main__':
    video_path = os.getenv("VIDEO_PATH", "")
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'

    ref_images = _parse_ref_images_env(os.getenv('FILTER_REF_IMAGES', ''))
    ref_images_negative = _parse_ref_images_env(os.getenv('FILTER_REF_IMAGES_NEGATIVE', ''))
    composite_topic = (os.getenv('FILTER_COMPOSITE_TOPIC') or '').strip()

    print(f"Using VideoIn with path: {video_path}")
    print(f"Text prompt: {os.getenv('FILTER_TEXT_PROMPT') or '(none; ref images/boxes can be used without)'}")
    print(f"Positive boxes: {os.getenv('FILTER_POSITIVE_BOXES') or '(none)'}")
    print(f"Negative boxes: {os.getenv('FILTER_NEGATIVE_BOXES') or '(none)'}")
    print(f"Ref images (FILTER_REF_IMAGES): {ref_images if ref_images else '(none)'}")
    print(f"Ref images negative: {ref_images_negative if ref_images_negative else '(none)'}")
    if composite_topic:
        print(f"Composite topic: {composite_topic}")
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
        ref_images=ref_images,
        ref_images_negative=ref_images_negative,
        composite_topic=composite_topic or None,
    )

    filters = [
        (
            VideoIn,
            dict(
                sources=f"file://{video_path}!sync",
                outputs="tcp://*:5550",
            ),
        ),
        (FilterSAM3Detector, detector_config),
        (
            Webvis, dict(sources="tcp://localhost:5552",
            port=9000,
        )),
    ]

    if os.getenv('FILTER_POSITIVE_BOXES') or os.getenv('FILTER_NEGATIVE_BOXES'):
        mode = "reference-boxes"
    elif ref_images or ref_images_negative:
        mode = "reference-images (REF_IMGS)"
    else:
        mode = "text-prompt only"
    print(f"\nStarting pipeline ({mode})...")
    print(f"Results will be saved to: {output_path}")
    print(f"  - detections.jsonl, frames/")
    Filter.run_multi(filters)
