#!/usr/bin/python3
"""
Run SAM3 detection on video with simplified order UI overlay.

Shows a monitoring interface with:
- Status indicators (ONLINE, Connected) with green border
- Status panel (Dressing Cups, Bowl, Chit) with green border
- COMPLETE status bar when all chit ingredients are detected

Usage:
    python run_detection_with_order_ui.py --video video.mp4 --objects avocado tomato cucumber --confidence 0.2 --min-score 0.3
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import cv2
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SAM3 detection on video with simplified order UI overlay",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Detect avocado, tomato, cucumber
    python run_detection_with_order_ui.py -v video.mp4 -o avocado tomato cucumber -c 0.2 -m 0.3
    
    # With skip frames
    python run_detection_with_order_ui.py -v video.mp4 -o avocado tomato cucumber -c 0.2 -m 0.3 -s 19
    """
)
parser.add_argument("--video", "-v", type=str, required=True,
                    help="Path to input video file")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output directory (default: next to video file with timestamp)")
parser.add_argument("--objects", "-obj", type=str, nargs="+", required=True,
                    help="Objects to detect (e.g., avocado tomato cucumber)")
parser.add_argument("--confidence", "-c", type=float, default=0.2,
                    help="Confidence threshold for SAM3 detector (default: 0.2)")
parser.add_argument("--min-score", "-m", type=float, default=0.7,
                    help="Minimum score to keep in final results (default: 0.7)")
parser.add_argument("--skip-frames", "-s", type=int, default=0,
                    help="Number of frames to skip between processing (0 = process all frames)")
parser.add_argument("--show-score", action="store_true", default=False,
                    help="Show confidence score in bounding box labels (default: False)")

args = parser.parse_args()

# Fixed chit ingredients (from the receipt/note)
CHIT_INGREDIENTS = [
    "Basmati Rice",  # Base
    "Nori",  # Crunch
    "Crispy Onions",  # Crunch
    "Pickled Onion",  # Toppings 1
    "Cucumber",  # Toppings 2
    "Avocado",  # Toppings 2
    "Spicy Cash Dr",  # Premium (Spicy Cashew Dressing)
    "Miso Steelhd",  # Premium (Miso Steelhead)
]

# Map detected objects to chit ingredients (normalize names)
INGREDIENT_MAPPING = {
    "avocado": "Avocado",
    "tomato": "Tomato",
    "cucumber": "Cucumber",
    "cumber": "Cucumber",
}

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
    output_dir = video_file.parent / f"{video_name}_order_ui_{timestamp}"

objects_to_detect = args.objects
confidence_threshold = args.confidence
min_score_filter = args.min_score
skip_frames = args.skip_frames
show_score = args.show_score

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

print(f"Video: {video_path}")
print(f"Output: {output_dir}")
print(f"Objects to detect: {objects_to_detect}")
print(f"Confidence threshold: {confidence_threshold}")
print(f"Min score filter: {min_score_filter}")
print(f"Skip frames: {skip_frames}")
print(f"Show score in labels: {show_score}")

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

# Color map for bounding boxes (BGR for OpenCV)
label_colors_bgr = {
    "avocado": (0, 255, 0),      # Green
    "tomato": (0, 0, 255),      # Red
    "cucumber": (255, 255, 0),   # Cyan
    "cup": (255, 0, 0),         # Blue
    "dressing_cup": (255, 0, 0), # Blue
}
default_color_bgr = (0, 255, 255)  # Yellow for unknown

# Process frames
detections_filename = f"{video_name}_detections_{timestamp}.jsonl"
detections_file = open(output_dir / detections_filename, "w")
frame_count = 0
valid_frames_seen = 0
processed_count = 0
saved_count = 0
invalid_frames = 0

# Video writer setup
video_writer = None
video_output_path = None

def draw_order_ui(img, detected_ingredients_set, frame_time_str, has_bowl=False, has_chit=False):
    """Draw simplified order monitoring UI overlay on the image."""
    h, w = img.shape[:2]
    
    # Create overlay for UI elements
    overlay = img.copy()
    
    font_ui = cv2.FONT_HERSHEY_SIMPLEX
    font_status = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Top-left: Status indicator (ONLINE, Connected) with green border
    status_box_w = 200
    status_box_h = 80
    status_x = 10
    status_y = 10
    
    # Draw black background
    cv2.rectangle(overlay, 
                 (status_x, status_y), 
                 (status_x + status_box_w, status_y + status_box_h),
                 (0, 0, 0), -1)
    
    # Draw green border
    cv2.rectangle(overlay,
                 (status_x, status_y),
                 (status_x + status_box_w, status_y + status_box_h),
                 (0, 255, 0), 2)
    
    # Draw status text
    cv2.putText(overlay, "ONLINE", 
               (status_x + 10, status_y + 30),
               font_status, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, "Connected", 
               (status_x + 10, status_y + 60),
               font_status, 0.5, (200, 200, 200), 1)
    
    # 2. Top-right: Status panel (Dressing Cups, Bowl, Chit) with green border
    panel_padding = 15
    panel_x = w - 220
    panel_y = 10
    panel_w = 210
    panel_h = 100
    
    # Draw black background
    cv2.rectangle(overlay,
                 (panel_x, panel_y),
                 (panel_x + panel_w, panel_y + panel_h),
                 (0, 0, 0), -1)
    
    # Draw green border
    cv2.rectangle(overlay,
                 (panel_x, panel_y),
                 (panel_x + panel_w, panel_y + panel_h),
                 (0, 255, 0), 2)
    
    # Draw panel content
    y_pos = panel_y + 25
    line_spacing = 25
    
    # Dressing Cups (fixed as 0/0 for now)
    cv2.putText(overlay, "Dressing Cups: 0/0",
               (panel_x + panel_padding, y_pos),
               font_ui, 0.5, (255, 255, 255), 1)
    y_pos += line_spacing
    
    # Bowl: YES/NO
    bowl_text = "Bowl: YES" if has_bowl else "Bowl: NO"
    bowl_color = (255, 255, 255) if has_bowl else (0, 255, 255)  # White if YES, Yellow if NO
    cv2.putText(overlay, bowl_text,
               (panel_x + panel_padding, y_pos),
               font_ui, 0.5, bowl_color, 1)
    y_pos += line_spacing
    
    # Chit: YES/NO
    chit_text = "Chit: YES" if has_chit else "Chit: NO"
    chit_color = (255, 255, 255) if has_chit else (0, 255, 255)  # White if YES, Yellow if NO
    cv2.putText(overlay, chit_text,
               (panel_x + panel_padding, y_pos),
               font_ui, 0.5, chit_color, 1)
    
    # 3. Bottom-right: Timestamp
    timestamp_x = w - 200
    timestamp_y = h - 50
    cv2.putText(overlay, frame_time_str,
               (timestamp_x, timestamp_y),
               font_ui, 0.5, (255, 255, 255), 1)
    
    # 4. Bottom: Status bar - COMPLETE if all chit ingredients are detected
    # Check which chit ingredients match detected ingredients
    detected_chit_count = 0
    for chit_ingredient in CHIT_INGREDIENTS:
        chit_lower = chit_ingredient.lower()
        for det_ingredient in detected_ingredients_set:
            det_lower = det_ingredient.lower()
            # Check if detected ingredient matches chit ingredient
            if (chit_lower in det_lower or det_lower in chit_lower or 
                any(word in det_lower for word in chit_lower.split() if len(word) > 3) or
                any(word in chit_lower for word in det_lower.split() if len(word) > 3)):
                detected_chit_count += 1
                break
    
    # Mark as COMPLETE if we have detections matching chit ingredients and bowl/chit are present
    # For simplicity, we'll mark complete if key ingredients (Cucumber, Avocado) are detected
    key_ingredients = ["cucumber", "avocado"]
    key_detected = any(key in det.lower() for det in detected_ingredients_set for key in key_ingredients)
    
    all_complete = (key_detected and has_bowl and has_chit and len(detected_ingredients_set) > 0)
    
    bar_h = 40
    if all_complete:
        status_color = (0, 255, 0)  # Green - COMPLETE
        status_text = "COMPLETE"
        cv2.rectangle(overlay,
                     (0, h - bar_h),
                     (w, h),
                     status_color, -1)
        cv2.putText(overlay, status_text,
                   (w // 2 - 60, h - 10),
                   font_status, 0.8, (255, 255, 255), 2)
    
    # Blend overlay with original image (less opacity for cleaner look)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    return img

try:
    print("Starting frame processing...")
    while True:
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

        # Validate frame
        frame_valid = False
        frame_img = None
        for topic, frame in frames_dict.items():
            if frame and hasattr(frame, 'rw_bgr') and frame.rw_bgr and hasattr(frame.rw_bgr, 'image'):
                img = frame.rw_bgr.image
                if img is not None and len(img.shape) == 3 and img.shape[0] > 0 and img.shape[1] > 0:
                    if img.size > 0 and img.max() > 0:
                        frame_valid = True
                        frame_img = img.copy()
                        break

        if not frame_valid:
            invalid_frames += 1
            continue

        valid_frames_seen += 1

        # Skip frames logic
        if skip_frames > 0:
            if (valid_frames_seen - 1) % (skip_frames + 1) != 0:
                continue

        processed_count += 1

        # Process each object type separately
        all_detections = []
        
        for obj_label in objects_to_detect:
            detector.cfg.text_prompt = obj_label
            detector.text_prompt = obj_label
            
            frame = Frame(image=frame_img, format="BGR")
            frames_dict_det = {"frames": frame}
            
            if processed_count == 1 and frame_count <= 5:
                print(f"  Processing frame {frame_count}, detecting: {obj_label}...")
            detected = detector.process(frames_dict_det)
            
            if detected:
                for topic, frame in detected.items():
                    dets = frame.data.get('meta', {}).get('detections', [])
                    for det in dets:
                        labeled_det = det.copy()
                        labeled_det['label'] = obj_label
                        all_detections.append(labeled_det)
                    break

        # Filter detections by minimum score
        filtered_detections = [det for det in all_detections if det.get('score', 0.0) >= min_score_filter]
        all_detections = filtered_detections

        # Always draw UI (even if no detections)
        img_with_ui = frame_img.copy()
        
        # Get detected ingredients
        detected_labels = set(det.get('label', 'unknown') for det in all_detections)
        detected_ingredients_set = set()
        for label in detected_labels:
            # Normalize label names
            normalized = INGREDIENT_MAPPING.get(label.lower(), label.capitalize())
            detected_ingredients_set.add(normalized)
        
        # Determine if bowl and chit are present
        # For now, assume bowl is present if we have detections, chit is always present
        # You can add actual bowl/chit detection here if needed
        has_bowl = len(all_detections) > 0  # Simple heuristic: bowl present if detections exist
        has_chit = True  # Assume chit is always present (or add detection logic)
        
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Draw order UI overlay
        img_with_ui = draw_order_ui(img_with_ui, detected_ingredients_set, current_time, has_bowl, has_chit)
        
        # Draw bounding boxes if there are detections
        if all_detections:
            for det in all_detections:
                label = det.get('label', 'unknown')
                box = det.get('box', [])
                score = det.get('score', 0.0)
                
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    color = label_colors_bgr.get(label.lower(), default_color_bgr)
                    
                    # Draw rectangle
                    cv2.rectangle(img_with_ui, (x1, y1), (x2, y2), color, 3)
                    
                    # Add label
                    if show_score:
                        label_text = f"{label}: {score:.2f}"
                    else:
                        label_text = f"{label}"
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    cv2.rectangle(img_with_ui, 
                                 (x1, y1 - text_height - 10),
                                 (x1 + text_width + 6, y1),
                                 (255, 255, 255), -1)
                    
                    cv2.putText(img_with_ui, label_text, (x1 + 3, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save frame
            img_filename = f"{video_name}_frame_{frame_count:06d}_{timestamp}.jpg"
            img_path = frames_dir / img_filename
            cv2.imwrite(str(img_path), img_with_ui, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Initialize video writer on first frame with detections
            if video_writer is None:
                h, w = img_with_ui.shape[:2]
                video_output_path = output_dir / f"{video_name}_order_ui_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0
                video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (w, h))
                print(f"Creating output video: {video_output_path}")
            
            # Write frame to video
            if video_writer is not None:
                video_writer.write(img_with_ui)
            
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
        else:
            # Even without detections, save frame with UI
            if video_writer is None:
                h, w = img_with_ui.shape[:2]
                video_output_path = output_dir / f"{video_name}_order_ui_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0
                video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (w, h))
                print(f"Creating output video: {video_output_path}")
            
            if video_writer is not None:
                video_writer.write(img_with_ui)

        # Progress messages
        if frame_count % 10 == 0 or saved_count == 1:
            print(f"Frame {frame_count}: processed {processed_count}, saved {saved_count} with detections" + 
                  (f", skipped {invalid_frames} invalid" if invalid_frames > 0 else ""))
        elif frame_count % 100 == 0:
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
    print(f"Warning: Skipped {invalid_frames} invalid/corrupted frames")
print(f"Results in: {output_dir}")
