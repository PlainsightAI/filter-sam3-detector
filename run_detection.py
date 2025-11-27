#!/usr/bin/env python3
"""Run SAM3 detection on video."""
import sys
from pathlib import Path
import json

# Config
video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/2025-11-21_processed.mp4"
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads/detection_results_2025-11-21"
prompt = sys.argv[3] if len(sys.argv) > 3 else "small transparent cup"

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

print(f"Video: {video_path}")
print(f"Output: {output_dir}")
print(f"Prompt: {prompt}")

# Setup filters
from openfilter.filter_runtime.filters.video_in import VideoIn
from filter_sam3_detector.filter import FilterSAM3Detector
import cv2

video_in = VideoIn({
    "sources": f"file://{video_path.absolute()}!maxsize=480x480",
    "outputs": ["frames"],
})
video_in.setup(video_in.config)

detector = FilterSAM3Detector({
    "text_prompt": prompt,
    "confidence_threshold": 0.2,
    "device": "cuda",
    "visualize": True,
    "output_label": "cups",
})
detector.setup(detector.config)

# Process frames
detections_file = open(output_dir / "detections.jsonl", "w")
frame_count = 0
saved_count = 0

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

        # Detect
        detected = detector.process(frames_dict)

        if not detected:
            continue

        # Check for detections
        for topic, frame in detected.items():
            dets = frame.data.get('meta', {}).get('cups', [])

            if dets:
                # Save frame
                img_path = frames_dir / f"frame_{frame_count:06d}.jpg"
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

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, saved {saved_count} with detections")

finally:
    detections_file.close()
    video_in.shutdown()
    detector.shutdown()

print(f"\nDone! Processed {frame_count} frames, saved {saved_count} with detections")
print(f"Results in: {output_dir}")
