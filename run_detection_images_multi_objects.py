#!/usr/bin/python3
"""
Run SAM3 detection on all images in a folder with multiple object classes.

Each object is processed separately to get labeled bounding boxes.
Images with detections are saved with bounding boxes drawn.
Dataset is saved in JSONL format and automatically converted to COCO format with all classes.

Usage:
    python run_detection_images_multi_objects.py --images /path/to/images --objects avocado tomato cup --confidence 0.2 --min-score 0.3
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import cv2
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SAM3 detection on images with multiple labeled objects",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Detect avocado and tomato
    python run_detection_images_multi_objects.py -i imgs_salad/ -o avocado tomato -c 0.2 -m 0.3
    
    # Single object
    python run_detection_images_multi_objects.py -i imgs_salad/ -o cup -c 0.2 -m 0.3
    
    # With custom output directory
    python run_detection_images_multi_objects.py -i imgs_salad/ -o avocado tomato -c 0.2 -m 0.3 --output output/
    """
)
parser.add_argument("--images", "-i", type=str, required=True,
                    help="Path to input images folder")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output directory (default: <images_folder>_detections_<timestamp>)")
parser.add_argument("--objects", "-obj", type=str, nargs="+", required=True,
                    help="Objects to detect (e.g., avocado tomato cup)")
parser.add_argument("--confidence", "-c", type=float, default=0.2,
                    help="Confidence threshold for SAM3 detector (default: 0.2)")
parser.add_argument("--min-score", "-m", type=float, default=0.7,
                    help="Minimum score to keep in final results (default: 0.7)")
parser.add_argument("--show-score", action="store_true", default=False,
                    help="Show confidence score in bounding box labels (default: False)")

args = parser.parse_args()

# Config
images_folder = Path(args.images)
if not images_folder.exists() or not images_folder.is_dir():
    print(f"Error: {images_folder} is not a valid directory")
    exit(1)

# Generate unique timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Default output: next to images folder or in current directory
if args.output:
    output_dir = Path(args.output)
else:
    output_dir = images_folder.parent / f"{images_folder.name}_detections_{timestamp}"

objects_to_detect = args.objects
confidence_threshold = args.confidence
min_score_filter = args.min_score
show_score = args.show_score

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

print(f"Images folder: {images_folder}")
print(f"Output: {output_dir}")
print(f"Objects to detect: {objects_to_detect}")
print(f"Confidence threshold: {confidence_threshold}")
print(f"Min score filter: {min_score_filter}")
print(f"Show score in labels: {show_score}")
print()

# Image extensions to process
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Find all image files recursively in subfolders
image_files = []
for ext in image_extensions:
    image_files.extend(images_folder.rglob(f"*{ext}"))
    image_files.extend(images_folder.rglob(f"*{ext.upper()}"))
image_files = sorted([f for f in image_files if f.is_file()])

if not image_files:
    print(f"No image files found in {images_folder} (including subfolders)")
    exit(1)

print(f"Found {len(image_files)} image file(s)")
print()

# Setup filters
from openfilter.filter_runtime.filter import Frame
from filter_sam3_detector.filter import FilterSAM3Detector

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

# Process images
detections_filename = f"detections_{timestamp}.jsonl"
detections_file = open(output_dir / detections_filename, "w")
processed_count = 0
saved_count = 0
error_count = 0

try:
    for idx, image_path in enumerate(image_files, 1):
        try:
            if idx % 10 == 0 or idx == 1 or idx == len(image_files):
                print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
            
            # Load image
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                print(f"  Warning: Could not load {image_path.name}, skipping")
                error_count += 1
                continue
            
            processed_count += 1
            
            # Process each object type separately to get labeled bounding boxes
            all_detections = []  # List of detections with labels
            
            for obj_label in objects_to_detect:
                # Update detector prompt for this object
                detector.cfg.text_prompt = obj_label
                detector.text_prompt = obj_label
                
                # Create Frame object for this detection
                frame = Frame(image=img_bgr, format="BGR")
                frames_dict_det = {"frames": frame}
                
                # Run detection
                if idx <= 3:
                    print(f"  Detecting: {obj_label}...")
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
                # Draw bounding boxes directly on original image
                img_with_boxes = img_bgr.copy()
                
                # Get unique labels from detections
                detected_labels = set(det.get('label', 'unknown') for det in all_detections)
                detected_ingredients = [obj for obj in objects_to_detect if obj in detected_labels]
                
                # Draw ingredients list in top right corner
                h, w = img_with_boxes.shape[:2]
                ingredients_text = "Ingredients:"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                line_spacing = 25
                
                # Calculate text dimensions for ingredients list
                ingredients_lines = [ingredients_text] + detected_ingredients
                max_text_width = 0
                text_height = 0
                baseline = 0
                
                for line in ingredients_lines:
                    (text_width, line_height), line_baseline = cv2.getTextSize(
                        line, font, font_scale, thickness
                    )
                    max_text_width = max(max_text_width, text_width)
                    text_height = line_height
                    baseline = line_baseline
                
                num_ingredients = len(detected_ingredients)
                total_text_height = text_height + baseline + (num_ingredients * line_spacing)
                
                # Position in top right with padding
                padding = 10
                box_x = w - max_text_width - padding * 2
                box_y = padding
                box_width = max_text_width + padding * 2
                box_height = total_text_height + padding * 2
                
                # Draw semi-transparent background rectangle
                overlay = img_with_boxes.copy()
                cv2.rectangle(overlay, 
                             (box_x, box_y), 
                             (box_x + box_width, box_y + box_height),
                             (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.8, img_with_boxes, 0.2, 0, img_with_boxes)
                
                # Draw border around the box
                cv2.rectangle(img_with_boxes,
                             (box_x, box_y),
                             (box_x + box_width, box_y + box_height),
                             (0, 0, 0), 2)
                
                # Draw ingredients text
                y_offset = box_y + padding + text_height + baseline
                for i, line in enumerate(ingredients_lines):
                    if i == 0:
                        cv2.putText(img_with_boxes, line,
                                   (box_x + padding, y_offset),
                                   font, font_scale, (0, 0, 0), thickness)
                        y_offset += line_spacing
                    else:
                        cv2.putText(img_with_boxes, line,
                                   (box_x + padding, y_offset),
                                   font, font_scale, (50, 50, 50), thickness - 1)
                        y_offset += line_spacing
                
                # Draw bounding boxes with labels
                for det in all_detections:
                    label = det.get('label', 'unknown')
                    box = det.get('box', [])
                    score = det.get('score', 0.0)
                    
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        color = label_colors_bgr.get(label.lower(), default_color_bgr)
                        
                        # Draw rectangle
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
                        
                        # Add label with object type and optionally score
                        if show_score:
                            label_text = f"{label}: {score:.2f}"
                        else:
                            label_text = f"{label}"
                        
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
                
                # Save image with detections
                rel_path = image_path.relative_to(images_folder)
                safe_name = str(rel_path).replace('/', '_').replace('\\', '_')
                img_filename = f"{safe_name}_det.jpg"
                img_path = frames_dir / img_filename
                cv2.imwrite(str(img_path), img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save detection
                h, w = img_bgr.shape[:2]
                rec = {
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "relative_path": str(rel_path),
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
                    
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

finally:
    detections_file.close()
    detector.shutdown()

print(f"\nDone! Processed {processed_count} images, saved {saved_count} with detections, {error_count} errors")
print(f"Results in: {output_dir}")

# Convert to COCO format
if saved_count > 0:
    print(f"\nConverting detections to COCO format with all classes...")
    try:
        # Import the conversion function
        import sys
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        
        from jsonl_to_coco import detections_jsonl_to_coco_multi_class
        
        jsonl_path = output_dir / detections_filename
        coco_output = output_dir / f"detections_{timestamp}_coco.json"
        
        detections_jsonl_to_coco_multi_class(
            str(jsonl_path),
            str(coco_output),
            objects_to_detect=objects_to_detect,
            min_score=min_score_filter
        )
        print(f"COCO format saved to: {coco_output}")
    except Exception as e:
        print(f"Warning: Failed to convert to COCO format: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No detections found, skipping COCO conversion")

