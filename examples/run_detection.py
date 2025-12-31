#!/usr/bin/python3
"""Run SAM3 detection on video."""
import argparse
from pathlib import Path
import json
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description="Run SAM3 detection on video")
parser.add_argument("--video", "-v", type=str, 
                    default=str(Path.home() / "Downloads/2025-11-21_processed.mp4"),
                    help="Path to input video file")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output directory (default: next to video file with timestamp)")
parser.add_argument("--prompt", "-p", type=str, default="small transparent cup",
                    help="Text prompt for detection")
parser.add_argument("--skip-frames", "-s", type=int, default=0,
                    help="Number of frames to skip between processing (0 = process all frames)")

args = parser.parse_args()

# Config
video_path = Path(args.video)
video_file = Path(video_path)
video_name = video_file.stem

# Generate unique timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Default output: next to video file or in current directory
if args.output:
    output_dir = Path(args.output)
else:
    # Save next to video file with timestamp
    output_dir = video_file.parent / f"{video_name}_detections_{timestamp}"

prompt = args.prompt
skip_frames = args.skip_frames

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

print(f"Video: {video_path}")
print(f"Output: {output_dir}")
print(f"Prompt: {prompt}")
print(f"Skip frames: {skip_frames}")
if skip_frames > 0:
    processing_rate = 1.0 / (skip_frames + 1)
    print(f"Processing rate: {processing_rate:.1%} of frames (1 every {skip_frames + 1} frames)")

# Setup filters
from openfilter.filter_runtime.filters.video_in import VideoIn
from filter_sam3_detector.filter import FilterSAM3Detector
import cv2

video_in = VideoIn({
    "sources": f"file://{video_path.absolute()}!sync!resize=1280x720",
    "outputs": ["frames"],
})
video_in.setup(video_in.config)

# Output label for detections (used as key in frame.data['meta'])
output_label = "dressing_cup"

detector = FilterSAM3Detector({
    "text_prompt": prompt,
    "confidence_threshold": 0.2,
    "device": "cuda",
    "visualize": True,
    "output_label": output_label,
})
detector.setup(detector.config)

# Process frames
detections_filename = f"{video_name}_detections_{timestamp}.jsonl"
detections_file = open(output_dir / detections_filename, "w")
frame_count = 0  # Total frames read from video
valid_frames_seen = 0  # Valid frames seen (used for skip logic)
processed_count = 0  # Frames actually processed (after skipping)
saved_count = 0
invalid_frames = 0  # Frames skipped due to corruption/invalid data

try:
    while True:
        # Get frame
        frames_dict = video_in.process({})
        if not frames_dict or frames_dict is None:
            break

        if callable(frames_dict):
            frames_dict = frames_dict()

        if not frames_dict:
            break

        frame_count += 1

        # Validate frame before processing (check for corrupted frames from HEVC errors)
        frame_valid = False
        for topic, frame in frames_dict.items():
            if frame and hasattr(frame, 'rw_bgr') and frame.rw_bgr and hasattr(frame.rw_bgr, 'image'):
                img = frame.rw_bgr.image
                if img is not None and len(img.shape) == 3 and img.shape[0] > 0 and img.shape[1] > 0:
                    # Check if image has valid pixel values (not all zeros or corrupted)
                    if img.size > 0 and img.max() > 0:
                        frame_valid = True
                        break
        
        if not frame_valid:
            # Skip corrupted/invalid frames (common with HEVC decoding errors)
            # These don't count towards skip logic - we'll process the next valid frame
            invalid_frames += 1
            continue

        # Count this as a valid frame seen
        valid_frames_seen += 1

        # Skip frames logic: if skip_frames > 0, only process every (skip_frames + 1) VALID frames
        # This ensures invalid frames don't affect the skip count
        # skip_frames = 2 means process valid frame 1, 4, 7, 10... (process 1, skip 2)
        # skip_frames = 3 means process valid frame 1, 5, 9, 13... (process 1, skip 3)
        # skip_frames = 19 means process valid frame 1, 21, 41, 61... (process 1, skip 19)
        #   For a 20 fps video: skip_frames=19 processes 1 valid frame per second
        #   For a 30 fps video: skip_frames=29 processes 1 valid frame per second
        if skip_frames > 0:
            # Use valid_frames_seen for skip logic (only valid frames count)
            if (valid_frames_seen - 1) % (skip_frames + 1) != 0:
                continue

        processed_count += 1

        # Detect
        detected = detector.process(frames_dict)

        if not detected:
            continue

        # Check for detections
        for topic, frame in detected.items():
            dets = frame.data.get('meta', {}).get(output_label, [])

            # Save frame if it has detections (or optionally save all frames)
            if dets:
                # Save frame with video name, frame number, and timestamp
                img_filename = f"{video_name}_frame_{frame_count:06d}_{timestamp}.jpg"
                img_path = frames_dir / img_filename
                cv2.imwrite(str(img_path), frame.rw_bgr.image)

                # Save detection
                h, w = frame.rw_bgr.image.shape[:2]
                rec = {
                    "frame": frame_count,
                    "width": w,
                    "height": h,
                    "detections": [{"box": d['box'], "score": float(d['score'])} for d in dets]
                }
                detections_file.write(json.dumps(rec) + "\n")
                detections_file.flush()
                saved_count += 1
            # Uncomment below to save ALL frames (even without detections)
            # else:
            #     img_filename = f"{video_name}_frame_{frame_count:06d}_no_det_{timestamp}.jpg"
            #     img_path = frames_dir / img_filename
            #     cv2.imwrite(str(img_path), frame.rw_bgr.image)

        if frame_count % 100 == 0:
            print(f"Read {frame_count} frames, processed {processed_count}, saved {saved_count} with detections" + 
                  (f", skipped {invalid_frames} invalid" if invalid_frames > 0 else ""))

finally:
    detections_file.close()
    video_in.shutdown()
    detector.shutdown()

print(f"\nDone! Read {frame_count} frames, processed {processed_count}, saved {saved_count} with detections")
if invalid_frames > 0:
    print(f"Warning: Skipped {invalid_frames} invalid/corrupted frames (likely due to HEVC decoding errors)")
print(f"Results in: {output_dir}")

# Convert to COCO format
if saved_count > 0:
    print(f"\nConverting detections to COCO format...")
    try:
        # Import the conversion function
        import sys
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        
        from jsonl_to_coco import detections_jsonl_to_coco
        
        jsonl_path = output_dir / detections_filename
        coco_output = output_dir / f"{video_name}_detections_{timestamp}_coco.json"
        
        detections_jsonl_to_coco(
            str(jsonl_path),
            str(coco_output),
            category_name=output_label,
            min_score=0.0
        )
        print(f"COCO format saved to: {coco_output}")
    except Exception as e:
        print(f"Warning: Failed to convert to COCO format: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No detections found, skipping COCO conversion")
