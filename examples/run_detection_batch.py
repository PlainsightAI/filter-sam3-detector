#!/usr/bin/python3
"""Run SAM3 detection on all videos in a directory."""
import sys
from pathlib import Path
import json

# Config
if len(sys.argv) < 2:
    print("Usage: python3 run_detection_batch.py <video_directory> [output_base_dir] [prompt]")
    print("Example: python3 run_detection_batch.py /home/leandrobmarinho/datasets/2025-11-12")
    sys.exit(1)

video_dir = Path(sys.argv[1])
if not video_dir.exists() or not video_dir.is_dir():
    print(f"Error: {video_dir} is not a valid directory")
    sys.exit(1)

output_base_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else video_dir
prompt = sys.argv[3] if len(sys.argv) > 3 else "small transparent cup"

# Video extensions to process
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v'}

# Find all video files
video_files = [f for f in video_dir.iterdir() 
               if f.is_file() and f.suffix.lower() in video_extensions]

if not video_files:
    print(f"No video files found in {video_dir}")
    sys.exit(1)

print(f"Found {len(video_files)} video file(s) in {video_dir}")
print(f"Output base directory: {output_base_dir}")
print(f"Prompt: {prompt}")
print()

# Setup filters (will be reused for all videos)
from openfilter.filter_runtime.filters.video_in import VideoIn
from filter_sam3_detector.filter import FilterSAM3Detector
import cv2

detector = FilterSAM3Detector({
    "text_prompt": prompt,
    "confidence_threshold": 0.2,
    "device": "cuda",
    "visualize": True,
    "output_label": "cups",
})
detector.setup(detector.config)

# Process each video
for idx, video_path in enumerate(video_files, 1):
    print(f"\n{'='*60}")
    print(f"Processing video {idx}/{len(video_files)}: {video_path.name}")
    print(f"{'='*60}")
    
    # Output directory for this video
    output_dir = output_base_dir / f"{video_path.stem}_detections"
    output_dir.mkdir(exist_ok=True, parents=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    try:
        # Setup video input for this video
        video_in = VideoIn({
            "sources": f"file://{video_path.absolute()}!maxsize=1920x1080!sync",
            "outputs": ["frames"],
        })
        video_in.setup(video_in.config)
        
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
                    
                    # Save frame if it has detections (or optionally save all frames)
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
                    print(f"  Processed {frame_count} frames, saved {saved_count} with detections")
        
        finally:
            detections_file.close()
            video_in.shutdown()
        
        print(f"\n✓ Completed: {video_path.name}")
        print(f"  Processed {frame_count} frames, saved {saved_count} with detections")
        print(f"  Results in: {output_dir}")
        print(f"  Moving to next video...")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user. Stopping batch processing.")
        print(f"  Processed {idx}/{len(video_files)} videos before interruption.")
        raise
    except Exception as e:
        print(f"\n✗ Error processing {video_path.name}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        continue

# Shutdown detector
detector.shutdown()

print(f"\n{'='*60}")
print(f"All videos processed!")
print(f"{'='*60}")

