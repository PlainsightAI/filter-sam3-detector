#!/usr/bin/python3
"""Run SAM3 detection on all images in a folder."""
import sys
from pathlib import Path
import json
import cv2

# Config
if len(sys.argv) < 2:
    print("Usage: python3 run_detection_images.py <image_folder> [output_dir] [prompt]")
    print("Example: python3 run_detection_images.py /path/to/images")
    sys.exit(1)

image_folder = Path(sys.argv[1])
if not image_folder.exists() or not image_folder.is_dir():
    print(f"Error: {image_folder} is not a valid directory")
    sys.exit(1)

# Default output: next to image folder or in current directory
if len(sys.argv) > 2:
    output_dir = Path(sys.argv[2])
else:
    output_dir = image_folder.parent / f"{image_folder.name}_detections"

prompt = sys.argv[3] if len(sys.argv) > 3 else "small transparent cup"

output_dir.mkdir(exist_ok=True, parents=True)
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

# Image extensions to process
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Find all image files recursively in subfolders
image_files = []
for ext in image_extensions:
    image_files.extend(image_folder.rglob(f"*{ext}"))
    image_files.extend(image_folder.rglob(f"*{ext.upper()}"))
image_files = sorted([f for f in image_files if f.is_file()])

if not image_files:
    print(f"No image files found in {image_folder} (including subfolders)")
    sys.exit(1)

print(f"Found {len(image_files)} image file(s) in {image_folder} (including subfolders)")
print(f"Output: {output_dir}")
print(f"Prompt: {prompt}")
print()

# Setup detector
from openfilter.filter_runtime.filter import Frame
from filter_sam3_detector.filter import FilterSAM3Detector

detector = FilterSAM3Detector({
    "text_prompt": prompt,
    "confidence_threshold": 0.2,
    "device": "cuda",
    "visualize": False,  # Disabled to avoid read-only image attribute warnings
    "output_label": "cups",
})
detector.setup(detector.config)

# Process images
detections_file = open(output_dir / "detections.jsonl", "w")
processed_count = 0
saved_count = 0
error_count = 0
skipped_count = 0

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
            
            # Create Frame object
            frame = Frame(image=img_bgr, format="BGR")
            frames_dict = {"frames": frame}
            
            # Detect
            detected = detector.process(frames_dict)
            processed_count += 1
            
            # Check for detections
            has_detections = False
            if detected:
                for topic, frame in detected.items():
                    dets = frame.data.get('meta', {}).get('cups', [])
                    
                    if dets:
                        has_detections = True
                        # Save image if it has detections
                        try:
                            # Create unique filename using relative path to avoid conflicts
                            rel_path = image_path.relative_to(image_folder)
                            safe_name = str(rel_path).replace('/', '_').replace('\\', '_')
                            img_path = frames_dir / f"{safe_name}_det.jpg"
                            cv2.imwrite(str(img_path), frame.rw_bgr.image)
                            
                            # Save detection
                            h, w = frame.rw_bgr.image.shape[:2]
                            rec = {
                                "image": image_path.name,
                                "image_path": str(image_path),
                                "relative_path": str(rel_path),
                                "width": w,
                                "height": h,
                                "detections": [{"box": d['box'], "score": float(d['score'])} for d in dets]
                            }
                            detections_file.write(json.dumps(rec) + "\n")
                            detections_file.flush()
                            saved_count += 1
                        except Exception as e:
                            print(f"  Error saving results for {image_path.name}: {e}")
                            error_count += 1
                        break  # Only process first topic with detections
            
            # Skip if no detections found
            if not has_detections:
                skipped_count += 1
                    
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            error_count += 1
            continue

finally:
    detections_file.close()
    detector.shutdown()

print(f"\n{'='*60}")
print(f"Done! Summary:")
print(f"  Total images: {len(image_files)}")
print(f"  Processed: {processed_count}")
print(f"  With detections (saved): {saved_count}")
print(f"  Skipped (no detections): {skipped_count}")
print(f"  Errors: {error_count}")
print(f"  Results in: {output_dir}")
print(f"{'='*60}")

