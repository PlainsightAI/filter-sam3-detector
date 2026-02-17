#!/usr/bin/env python

"""
Example script for running object detection using FilterSAM3Detector.

This script demonstrates how to use the FilterSAM3Detector with text prompts
in a complete pipeline with video input and output.

The script uses OpenFilter's standard environment variable pattern:
- FILTER_SOURCES: Input source (automatically handled by Filter.normalize_config)
- FILTER_OUTPUTS: Output destination (automatically handled by Filter.normalize_config)
- Filter-specific variables: FILTER_TEXT_PROMPT, FILTER_DEVICE, etc.

Required environment variables in .env file (one of):
    VIDEO_PATH: Path to the input video file
    IMAGE_PATH: Path to input image file or directory of images (uses ImageIn instead of VideoIn)
    FILTER_TEXT_PROMPT: Text prompt for detection (e.g., "person", "car")

Optional environment variables:
    FILTER_DEVICE: Device to use (cuda, cpu, mps) - default: cuda
    FILTER_CONFIDENCE_THRESHOLD: Confidence threshold (0.0-1.0) - default: 0.5
    FILTER_MAX_DETECTIONS: Maximum detections per frame - default: 100
    FILTER_VISUALIZE: Draw detections on frames (true/false) - default: false
    FILTER_OUTPUT_DIR: Output directory - default: ./output
    FILTER_VIDEO_LOOP: true to loop video so frames are still available after model load (~14s) - default: false

Example .env file content:
    VIDEO_PATH=/path/to/your/video.mp4
    FILTER_TEXT_PROMPT=person
    FILTER_DEVICE=cuda
    FILTER_CONFIDENCE_THRESHOLD=0.5
    FILTER_VISUALIZE=true
    FILTER_OUTPUT_DIR=./results
"""

import os
from pathlib import Path

# Try to load .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass

from openfilter.filter_runtime.filter import Filter
from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.image_in import ImageIn
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.recorder import Recorder


if __name__ == '__main__':
    # Get video or image path from environment variable
    video_path = os.getenv('VIDEO_PATH', '')
    image_path = os.getenv('IMAGE_PATH', '')
    
    # Get optional configuration (will be read by normalize_config from env vars)
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'
    resize = os.getenv('FILTER_RESIZE', '')
    
    # Require either VIDEO_PATH or IMAGE_PATH
    if not video_path and not image_path:
        print("Error: Either VIDEO_PATH or IMAGE_PATH environment variable is required")
        print("Please set the path to your input video or image (file/dir) in the .env file")
        exit(1)
    
    if image_path:
        if not Path(image_path).exists():
            print(f"Error: Image path not found: {image_path}")
            exit(1)
        input_source = (ImageIn, dict(
            sources=f'file://{Path(image_path).absolute()}!maxfps=1!no-loop;main',
            outputs='tcp://*:5550',
            poll_interval=0,
        ))
        print(f"Using ImageIn with path: {image_path} (no loop, no polling)")
    else:
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            exit(1)
        video_loop = os.getenv('FILTER_VIDEO_LOOP', 'false').lower() == 'true'
        video_source = f'file://{Path(video_path).absolute()}'
        if resize:
            video_source += f'!maxsize={resize}x{resize}'
        video_source += '!loop;main' if video_loop else '!no-loop;main'
        loop_str = "loop" if video_loop else "no loop"
        input_source = (VideoIn, dict(
            sources=video_source,
            outputs='tcp://*:5550',
        ))
        print(f"Using VideoIn with path: {video_path} ({loop_str}, sync)")
    print(f"Text prompt: {os.getenv('FILTER_TEXT_PROMPT', 'NOT SET')}")
    print(f"Device: {os.getenv('FILTER_DEVICE', 'cuda')}")
    print(f"Confidence threshold: {os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.5')}")
    print(f"Max detections: {os.getenv('FILTER_MAX_DETECTIONS', '100')}")
    print(f"Visualize: {visualize}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Build filters pipeline
    # OpenFilter will automatically read FILTER_SOURCES and FILTER_OUTPUTS from env vars
    filters = [
        # Input: Stream video or image(s)
        input_source,
        
        # Detect objects with SAM3
        # Filter-specific config will be read from FILTER_* env vars by normalize_config
        (FilterSAM3Detector, FilterSAM3DetectorConfig(
            id="filter_sam3_detector",
            sources="tcp://localhost:5550",  # Can be overridden by FILTER_SOURCES env var
            outputs="tcp://*:5552",  # Can be overridden by FILTER_OUTPUTS env var
            output_label="detections",
            output_path=str(output_path / "detections.jsonl"),  # Save annotations directly
            frames_output_dir=str(output_path / "frames"),  # Save frames with detections
        )),
        
        # Record filter output to confirm detections
        (Recorder, dict(
            id="rec",
            sources="tcp://localhost:5552",
            outputs=f"file://{output_path / 'detection_out_dev.json'}",
            rules="+",
            flush=True,
        )),
        
        # Add Webvis for visualization
        (Webvis, dict(
            sources="tcp://localhost:5552",
            port=9000,
        )),
    ]
    
    print("Results will be shown in web interface")
    
    print(f"\nStarting pipeline...")
    print(f"Results will be saved to: {output_path}")
    print(f"  - detections.jsonl: Frame-by-frame detections (from FilterSAM3Detector)")
    print(f"  - detection_out_dev.json: Filter output confirmation (from Recorder)")
    print(f"  - frames/: Frames with detections (saved automatically)")
    print("\nNote: Filter configuration is read from FILTER_* environment variables")
    print("      (FILTER_TEXT_PROMPT, FILTER_DEVICE, FILTER_CONFIDENCE_THRESHOLD, etc.)")
    
    # Run the pipeline
    Filter.run_multi(filters)
