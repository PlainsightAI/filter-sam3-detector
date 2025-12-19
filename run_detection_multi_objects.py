#!/usr/bin/python3
"""
Run SAM3 detection on video with multiple objects (like detect_image.ipynb).

Each object is processed separately to get labeled bounding boxes.
Frames with detections are saved with bounding boxes drawn.

Usage:
    python run_detection_multi_objects.py --video video.mp4 --objects avocado tomato --confidence 0.2 --min-score 0.3
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import cv2
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SAM3 detection on video with multiple labeled objects",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Detect avocado and tomato
    python run_detection_multi_objects.py -v video.mp4 -o avocado tomato -c 0.2 -m 0.3
    
    # Single object
    python run_detection_multi_objects.py -v video.mp4 -o cup -c 0.2 -m 0.3
    
    # With skip frames
    python run_detection_multi_objects.py -v video.mp4 -o avocado tomato -c 0.2 -m 0.3 -s 19
    """
)
parser.add_argument("--video", "-v", type=str, required=True,
                    help="Path to input video file")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output directory (default: next to video file with timestamp)")
parser.add_argument("--objects", "-obj", type=str, nargs="+", required=True,
                    help="Objects to detect (e.g., avocado tomato cup)")
parser.add_argument("--confidence", "-c", type=float, default=0.2,
                    help="Confidence threshold for SAM3 detector (default: 0.2)")
parser.add_argument("--min-score", "-m", type=float, default=0.7,
                    help="Minimum score to keep in final results (default: 0.7)")
parser.add_argument("--skip-frames", "-s", type=int, default=0,
                    help="Number of frames to skip between processing (0 = process all frames)")

args = parser.parse_args()

# Config
video_path = Path(args.video)
if not video_path.exists():
    print(f"Error: Video file not found: {video_path}")
    exit(1)

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

objects_to_detect = args.objects
confidence_threshold = args.confidence
min_score_filter = args.min_score
skip_frames = args.skip_frames

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

print(f"Video: {video_path}")
print(f"Output: {output_dir}")
print(f"Objects to detect: {objects_to_detect}")
print(f"Confidence threshold: {confidence_threshold}")
print(f"Min score filter: {min_score_filter}")
print(f"Skip frames: {skip_frames}")

# Try to get video info for time estimation
try:
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
         '-show_entries', 'stream=nb_frames,r_frame_rate', 
         '-of', 'default=noprint_wrappers=1', str(video_path)],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        nb_frames = None
        fps = 24.0  # default
        for line in lines:
            if 'nb_frames=' in line:
                nb_frames = int(line.split('=')[1])
            elif 'r_frame_rate=' in line:
                rate_parts = line.split('=')[1].split('/')
                if len(rate_parts) == 2:
                    fps = float(rate_parts[0]) / float(rate_parts[1])
        
        if nb_frames:
            # Calculate frames to process
            if skip_frames > 0:
                frames_to_process = nb_frames // (skip_frames + 1)
            else:
                frames_to_process = nb_frames
            
            detections_count = frames_to_process * len(objects_to_detect)
            # Estimate: 1-3 seconds per detection (conservative: 2 seconds)
            estimated_seconds = detections_count * 2
            estimated_minutes = estimated_seconds / 60
            
            print(f"Video info: {nb_frames} frames @ {fps:.1f} fps")
            print(f"Frames to process: {frames_to_process} ({detections_count} detections)")
            print(f"Estimated time: ~{estimated_minutes:.1f} minutes ({estimated_seconds/60:.1f} min - {detections_count*3/60:.1f} min range)")
except:
    pass

if skip_frames > 0:
    processing_rate = 1.0 / (skip_frames + 1)
    print(f"Processing rate: {processing_rate:.1%} of frames (1 every {skip_frames + 1} frames)")
print()

# Setup filters
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filter import Frame
from filter_sam3_detector.filter import FilterSAM3Detector

# Configure VideoIn with sync to ensure all frames are processed in order
video_in = VideoIn({
    "sources": f"file://{video_path.absolute()}!sync!resize=1280x720",
    "outputs": ["frames"],
})
video_in.setup(video_in.config)
print("VideoIn configured with sync mode - all frames will be processed in order")

# Initialize detector (will update prompt for each object)
detector = FilterSAM3Detector({
    "text_prompt": objects_to_detect[0],
    "confidence_threshold": confidence_threshold,
    "device": "cuda",
    "visualize": False,
    "output_label": "detections",
})
detector.setup(detector.config)

# Color map will be defined inline where needed (BGR for OpenCV)

# Process frames
detections_filename = f"{video_name}_detections_{timestamp}.jsonl"
detections_file = open(output_dir / detections_filename, "w")
frame_count = 0  # Total frames read from video
valid_frames_seen = 0  # Valid frames seen (used for skip logic)
processed_count = 0  # Frames actually processed (after skipping)
saved_count = 0
invalid_frames = 0  # Frames skipped due to corruption/invalid data

# Video writer setup (will be initialized after we know the frame dimensions)
video_writer = None
video_output_path = None

try:
    print("Starting frame processing (sync mode - processing all frames in order)...")
    while True:
        # Get frame (sync mode ensures frames are processed in order)
        frames_dict = video_in.process({})
        if not frames_dict or frames_dict is None:
            print("End of video reached")
            break

        if callable(frames_dict):
            frames_dict = frames_dict()

        if not frames_dict:
            print("No more frames available")
            break

        frame_count += 1

        # Validate frame before processing (check for corrupted frames from HEVC errors)
        frame_valid = False
        frame_img = None
        for topic, frame in frames_dict.items():
            if frame and hasattr(frame, 'rw_bgr') and frame.rw_bgr and hasattr(frame.rw_bgr, 'image'):
                img = frame.rw_bgr.image
                if img is not None and len(img.shape) == 3 and img.shape[0] > 0 and img.shape[1] > 0:
                    # Check if image has valid pixel values (not all zeros or corrupted)
                    if img.size > 0 and img.max() > 0:
                        frame_valid = True
                        frame_img = img.copy()
                        break

        if not frame_valid:
            # Skip corrupted/invalid frames (common with HEVC decoding errors)
            # These don't count towards skip logic - we'll process the next valid frame
            invalid_frames += 1
            continue

        # Count this as a valid frame seen
        valid_frames_seen += 1

        # Skip frames logic: if skip_frames > 0, only process every (skip_frames + 1) VALID frames
        if skip_frames > 0:
            # Use valid_frames_seen for skip logic (only valid frames count)
            if (valid_frames_seen - 1) % (skip_frames + 1) != 0:
                continue

        processed_count += 1

        # Process each object type separately to get labeled bounding boxes
        all_detections = []  # List of detections with labels
        
        for obj_label in objects_to_detect:
            # Update detector prompt for this object
            detector.cfg.text_prompt = obj_label
            detector.text_prompt = obj_label
            
            # Create Frame object for this detection
            frame = Frame(image=frame_img, format="BGR")
            frames_dict_det = {"frames": frame}
            
            # Run detection (this can take a few seconds per frame)
            if processed_count == 1 and frame_count <= 5:
                print(f"  Processing frame {frame_count}, detecting: {obj_label}...")
            detected = detector.process(frames_dict_det)
            
            # Extract detections and label them
            if detected:
                for topic, frame in detected.items():
                    dets = frame.data.get('meta', {}).get('detections', [])
                    for det in dets:
                        # Add label to each detection
                        labeled_det = det.copy()
                        labeled_det['label'] = obj_label
                        all_detections.append(labeled_det)
                    break

        # Filter detections by minimum score
        filtered_detections = [det for det in all_detections if det.get('score', 0.0) >= min_score_filter]
        all_detections = filtered_detections

        # Check for detections
        if all_detections:
            # Draw bounding boxes directly on original frame (preserves quality)
            img_with_boxes = frame_img.copy()
            
            # Color map for bounding boxes (BGR for OpenCV)
            label_colors_bgr = {
                "avocado": (0, 255, 0),      # Green
                "tomato": (0, 0, 255),      # Red
                "cucumber": (255, 255, 0),   # Cyan
                "cup": (255, 0, 0),         # Blue
                "dressing_cup": (255, 0, 0), # Blue
            }
            default_color_bgr = (0, 255, 255)  # Yellow for unknown
            
            # Draw bounding boxes with labels
            for det in all_detections:
                label = det.get('label', 'unknown')
                box = det.get('box', [])
                score = det.get('score', 0.0)
                
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    color = label_colors_bgr.get(label.lower(), default_color_bgr)
                    
                    # Draw rectangle (thicker line for visibility)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
                    
                    # Add label with object type and score (black text, white background)
                    label_text = f"{label}: {score:.2f}"
                    
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw white background rectangle for text
                    cv2.rectangle(img_with_boxes, 
                                 (x1, y1 - text_height - 10),
                                 (x1 + text_width + 6, y1),
                                 (255, 255, 255), -1)
                    
                    # Draw black text
                    cv2.putText(img_with_boxes, label_text, (x1 + 3, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save frame with original quality (high quality JPEG)
            img_filename = f"{video_name}_frame_{frame_count:06d}_{timestamp}.jpg"
            img_path = frames_dir / img_filename
            cv2.imwrite(str(img_path), img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Initialize video writer on first frame with detections
            if video_writer is None:
                h, w = img_with_boxes.shape[:2]
                video_output_path = output_dir / f"{video_name}_detections_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0  # Default FPS (adjust if needed)
                video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (w, h))
                print(f"Creating output video: {video_output_path}")
            
            # Write frame to video
            if video_writer is not None:
                video_writer.write(img_with_boxes)
            
            # Save detection
            h, w = frame_img.shape[:2]
            rec = {
                "frame": frame_count,
                "width": w,
                "height": h,
                "objects_detected": objects_to_detect,
                "detections": [
                    {
                        "label": d.get('label', 'unknown'),
                        "box": d['box'],
                        "score": float(d['score'])
                    }
                    for d in all_detections
                ]
            }
            detections_file.write(json.dumps(rec) + "\n")
            detections_file.flush()
            saved_count += 1

        # Progress messages
        if frame_count % 10 == 0 or saved_count == 1:
            # Show progress every 10 frames or on first detection
            print(f"Frame {frame_count}: processed {processed_count}, saved {saved_count} with detections" + 
                  (f", skipped {invalid_frames} invalid" if invalid_frames > 0 else ""))
        elif frame_count % 100 == 0:
            # More detailed progress every 100 frames
            print(f"Progress: Read {frame_count} frames, processed {processed_count}, saved {saved_count} with detections" + 
                  (f", skipped {invalid_frames} invalid" if invalid_frames > 0 else ""))

finally:
    detections_file.close()
    if video_writer is not None:
        video_writer.release()
        if video_output_path and video_output_path.exists():
            print(f"Video saved: {video_output_path}")
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
        
        # Use first object as category name (or generic name)
        category_name = objects_to_detect[0] if len(objects_to_detect) == 1 else "objects"
        
        detections_jsonl_to_coco(
            str(jsonl_path),
            str(coco_output),
            category_name=category_name,
            min_score=min_score_filter
        )
        print(f"COCO format saved to: {coco_output}")
    except Exception as e:
        print(f"Warning: Failed to convert to COCO format: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No detections found, skipping COCO conversion")

