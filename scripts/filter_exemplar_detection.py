#!/usr/bin/env python

"""
Example script for running object detection using FilterSAM3Detector with exemplar images.

This script demonstrates how to use the FilterSAM3Detector with exemplar images (few-shot learning)
in a complete pipeline with video input and output.

The script uses OpenFilter's standard environment variable pattern:
- FILTER_SOURCES: Input source (automatically handled by Filter.normalize_config)
- FILTER_OUTPUTS: Output destination (automatically handled by Filter.normalize_config)
- Filter-specific variables: FILTER_EXEMPLARS_PATH, FILTER_DEVICE, etc.

Required environment variables in .env file:
    FILTER_EXEMPLARS_PATH: Path to directory with exemplar images
    VIDEO_PATH: Path to the input video file

Optional environment variables:
    FILTER_DEVICE: Device to use (cuda, cpu, mps) - default: cuda
    FILTER_CONFIDENCE_THRESHOLD: Confidence threshold (0.0-1.0) - default: 0.3 (recommended for exemplars)
    FILTER_MAX_DETECTIONS: Maximum detections per frame - default: 100
    FILTER_VISUALIZE: Draw detections on frames (true/false) - default: false
    FILTER_OUTPUT_DIR: Output directory - default: ./output

Example .env file content:
    VIDEO_PATH=/path/to/your/video.mp4
    FILTER_EXEMPLARS_PATH=./exemplars/
    FILTER_DEVICE=cuda
    FILTER_CONFIDENCE_THRESHOLD=0.3
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
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.image_out import ImageOut


if __name__ == '__main__':
    # Get video path from environment variable
    video_path = os.getenv('VIDEO_PATH', '')
    
    # Get optional configuration (will be read by normalize_config from env vars)
    output_dir = os.getenv('FILTER_OUTPUT_DIR', './output')
    visualize = os.getenv('FILTER_VISUALIZE', 'false').lower() == 'true'
    resize = os.getenv('FILTER_RESIZE', '')
    
    # Validate required variables
    if not video_path:
        print("Error: VIDEO_PATH environment variable is required")
        print("Please set the path to your input video in the .env file")
        exit(1)
    
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        exit(1)
    
    # Build video source with options
    video_source = f'file://{Path(video_path).absolute()}'
    if resize:
        video_source += f'!maxsize={resize}x{resize}'
    video_source += '!no-loop;main'  # Process video once, no loop, topic: main, sync
    
    print(f"Using VideoIn with path: {video_path} (no loop, sync)")
    print(f"Exemplars directory: {os.getenv('FILTER_EXEMPLARS_PATH', 'NOT SET')}")
    print(f"Device: {os.getenv('FILTER_DEVICE', 'cuda')}")
    print(f"Confidence threshold: {os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.3')}")
    print(f"Max detections: {os.getenv('FILTER_MAX_DETECTIONS', '100')}")
    print(f"Visualize: {visualize}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Build filters pipeline
    # OpenFilter will automatically read FILTER_SOURCES and FILTER_OUTPUTS from env vars
    filters = [
        # Input: Stream video frames
        (VideoIn, dict(
            sources=video_source,
            outputs='tcp://*:5550',
        )),
        
        # Detect objects with SAM3 using exemplars
        # Filter-specific config will be read from FILTER_* env vars by normalize_config
        (FilterSAM3Detector, FilterSAM3DetectorConfig(
            id="filter_sam3_detector",
            sources="tcp://localhost:5550",  # Can be overridden by FILTER_SOURCES env var
            outputs="tcp://*:5552",  # Can be overridden by FILTER_OUTPUTS env var
            output_label="detections",
            output_path=str(output_path / "detections.jsonl"),  # Save annotations directly
        )),
    ]
    
    # Add ImageOut if visualization is enabled
    if visualize:
        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)
        filters.append(
            (ImageOut, dict(
                sources="tcp://localhost:5552",
                path=str(frames_dir / "%05d.jpg"),
                only_with_detections=True,
                outputs="tcp://*:5554",  # Add output to satisfy ImageOut requirement
            ))
        )
        print(f"Annotated frames will be saved to: {frames_dir}")
    else:
        # Add Webvis for visualization without saving frames
        filters.append(
            (Webvis, dict(
                sources="tcp://localhost:5552",
            ))
        )
        print("Results will be shown in web interface (frames not saved)")
    
    print(f"\nStarting pipeline...")
    print(f"Results will be saved to: {output_path}")
    print(f"  - detections.jsonl: Frame-by-frame detections")
    if visualize:
        print(f"  - frames/: Annotated frames")
    print("\nNote: Filter configuration is read from FILTER_* environment variables")
    print("      (FILTER_EXEMPLARS_PATH, FILTER_DEVICE, FILTER_CONFIDENCE_THRESHOLD, etc.)")
    
    # Run the pipeline
    Filter.run_multi(filters)
