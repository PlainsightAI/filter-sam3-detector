#!/usr/bin/env python3
"""
Convert detections.jsonl file to COCO format JSON.

Input format (detections.jsonl):
Each line contains:
{
    "frame": 11,
    "width": 1920,
    "height": 1080,
    "detections": [
        {"box": [x1, y1, x2, y2], "score": 0.95},
        ...
    ]
}

COCO format output:
Standard COCO JSON with images, annotations, and categories arrays.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def convert_box_to_coco(x1, y1, x2, y2):
    """
    Convert bounding box from [x1, y1, x2, y2] to COCO format [x, y, width, height].
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        tuple: (x, y, width, height) in COCO format
    """
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return x, y, width, height


def detections_jsonl_to_coco(jsonl_file_path, output_file_path=None, category_name="cup", min_score=0.0):
    """
    Convert detections.jsonl file to COCO format.
    
    Args:
        jsonl_file_path: Path to input JSONL file
        output_file_path: Path to output JSON file (default: same as JSONL with .json extension)
        category_name: Name of the category (default: "cup")
        min_score: Minimum score threshold for detections (default: 0.0)
    """
    # Determine output file
    if output_file_path is None:
        jsonl_path = Path(jsonl_file_path)
        output_file_path = str(jsonl_path.parent / f"{jsonl_path.stem}_coco.json")
    
    # Initialize COCO data structure
    coco_data = {
        'info': {
            'description': 'Converted from detections.jsonl',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'Detections JSONL to COCO Converter',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'licenses': [
            {
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }
        ],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': category_name,
                'supercategory': 'object'
            }
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    print(f"Reading JSONL file: {jsonl_file_path}")
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                frame = data.get('frame')
                width = data.get('width')
                height = data.get('height')
                detections = data.get('detections', [])
                
                if frame is None or width is None or height is None:
                    print(f"Warning: Skipping line {line_num} - missing frame, width, or height")
                    continue
                
                # Create filename from frame number
                filename = f"frame_{frame:06d}.jpg"
                
                # Add image info
                image_info = {
                    'id': image_id,
                    'width': width,
                    'height': height,
                    'file_name': filename,
                    'license': 1,
                    'flickr_url': '',
                    'coco_url': '',
                    'date_captured': ''
                }
                coco_data['images'].append(image_info)
                
                # Process detections
                for detection in detections:
                    box = detection.get('box')
                    score = detection.get('score', 1.0)
                    
                    if box is None or len(box) != 4:
                        print(f"Warning: Skipping invalid box in line {line_num}")
                        continue
                    
                    # Filter by score
                    if score < min_score:
                        continue
                    
                    # Convert box format from [x1, y1, x2, y2] to COCO [x, y, width, height]
                    x, y, w, h = convert_box_to_coco(box[0], box[1], box[2], box[3])
                    area = w * h
                    
                    # Create annotation
                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,  # Single category
                        'segmentation': [],
                        'area': area,
                        'bbox': [x, y, w, h],
                        'iscrowd': 0,
                        'score': score  # Add score as extra field
                    }
                    coco_data['annotations'].append(annotation)
                    annotation_id += 1
                
                image_id += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Save to JSON
    print(f"Writing COCO format to: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n✅ Conversion complete!")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")
    print(f"   Categories: {[c['name'] for c in coco_data['categories']]}")
    
    if min_score > 0.0:
        print(f"   (Filtered with min_score >= {min_score})")
    
    return output_file_path


def detections_jsonl_to_coco_multi_class(jsonl_file_path, output_file_path=None, objects_to_detect=None, min_score=0.0):
    """
    Convert detections.jsonl file to COCO format with multiple object classes.
    
    This function supports JSONL files with labeled detections (each detection has a 'label' field).
    It creates a separate category for each unique label found in the detections.
    
    Args:
        jsonl_file_path: Path to input JSONL file
        output_file_path: Path to output JSON file (default: same as JSONL with .json extension)
        objects_to_detect: List of object names to use as categories (if None, will auto-detect from labels)
        min_score: Minimum score threshold for detections (default: 0.0)
    """
    # Determine output file
    if output_file_path is None:
        jsonl_path = Path(jsonl_file_path)
        output_file_path = str(jsonl_path.parent / f"{jsonl_path.stem}_coco.json")
    
    # Initialize COCO data structure
    coco_data = {
        'info': {
            'description': 'Converted from detections.jsonl with multiple classes',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'Detections JSONL to COCO Converter (Multi-Class)',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'licenses': [
            {
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }
        ],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Build category mapping
    # If objects_to_detect is provided, use it; otherwise, we'll collect unique labels
    category_map = {}  # label -> category_id
    category_id = 1
    
    if objects_to_detect:
        # Use provided objects as categories
        for obj in objects_to_detect:
            category_map[obj] = category_id
            coco_data['categories'].append({
                'id': category_id,
                'name': obj,
                'supercategory': 'object'
            })
            category_id += 1
    
    # First pass: collect all unique labels if objects_to_detect not provided
    if not objects_to_detect:
        print("Collecting unique labels from detections...")
        unique_labels = set()
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    detections = data.get('detections', [])
                    for det in detections:
                        label = det.get('label')
                        if label:
                            unique_labels.add(label)
                except:
                    continue
        
        # Create categories from unique labels
        for label in sorted(unique_labels):
            category_map[label] = category_id
            coco_data['categories'].append({
                'id': category_id,
                'name': label,
                'supercategory': 'object'
            })
            category_id += 1
    
    print(f"Categories: {[c['name'] for c in coco_data['categories']]}")
    
    image_id = 1
    annotation_id = 1
    image_path_to_id = {}  # Map image paths to image IDs to avoid duplicates
    
    print(f"Reading JSONL file: {jsonl_file_path}")
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                width = data.get('width')
                height = data.get('height')
                detections = data.get('detections', [])
                
                if width is None or height is None:
                    print(f"Warning: Skipping line {line_num} - missing width or height")
                    continue
                
                # Determine image identifier (for videos: frame number, for images: path)
                frame = data.get('frame')
                image_path = data.get('image_path') or data.get('image')
                relative_path = data.get('relative_path')
                
                if frame is not None:
                    # Video format: use frame number
                    filename = f"frame_{frame:06d}.jpg"
                    image_key = f"frame_{frame}"
                elif image_path:
                    # Image format: use path
                    if relative_path:
                        filename = relative_path
                    else:
                        filename = Path(image_path).name
                    image_key = str(image_path)
                else:
                    # Fallback: use line number
                    filename = f"image_{line_num:06d}.jpg"
                    image_key = f"line_{line_num}"
                
                # Check if we've already added this image
                if image_key in image_path_to_id:
                    current_image_id = image_path_to_id[image_key]
                else:
                    # Add new image
                    current_image_id = image_id
                    image_path_to_id[image_key] = image_id
                    
                    image_info = {
                        'id': image_id,
                        'width': width,
                        'height': height,
                        'file_name': filename,
                        'license': 1,
                        'flickr_url': '',
                        'coco_url': '',
                        'date_captured': ''
                    }
                    coco_data['images'].append(image_info)
                    image_id += 1
                
                # Process detections
                for detection in detections:
                    box = detection.get('box')
                    score = detection.get('score', 1.0)
                    label = detection.get('label')
                    
                    if box is None or len(box) != 4:
                        print(f"Warning: Skipping invalid box in line {line_num}")
                        continue
                    
                    # Filter by score
                    if score < min_score:
                        continue
                    
                    # Get category ID for this label
                    if label and label in category_map:
                        cat_id = category_map[label]
                    elif label:
                        # Unknown label - add it as a new category
                        print(f"Warning: Unknown label '{label}' found, adding as new category")
                        cat_id = category_id
                        category_map[label] = category_id
                        coco_data['categories'].append({
                            'id': category_id,
                            'name': label,
                            'supercategory': 'object'
                        })
                        category_id += 1
                    else:
                        # No label - skip or use first category
                        print(f"Warning: Detection without label in line {line_num}, skipping")
                        continue
                    
                    # Convert box format from [x1, y1, x2, y2] to COCO [x, y, width, height]
                    x, y, w, h = convert_box_to_coco(box[0], box[1], box[2], box[3])
                    area = w * h
                    
                    # Create annotation
                    annotation = {
                        'id': annotation_id,
                        'image_id': current_image_id,
                        'category_id': cat_id,
                        'segmentation': [],
                        'area': area,
                        'bbox': [x, y, w, h],
                        'iscrowd': 0,
                        'score': score  # Add score as extra field
                    }
                    coco_data['annotations'].append(annotation)
                    annotation_id += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Save to JSON
    print(f"Writing COCO format to: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n✅ Conversion complete!")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")
    print(f"   Categories: {[c['name'] for c in coco_data['categories']]}")
    
    # Print per-category statistics
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = next(c['name'] for c in coco_data['categories'] if c['id'] == cat_id)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    if category_counts:
        print(f"\n   Per-category counts:")
        for cat_name, count in sorted(category_counts.items()):
            print(f"     {cat_name}: {count}")
    
    if min_score > 0.0:
        print(f"   (Filtered with min_score >= {min_score})")
    
    return output_file_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert detections.jsonl file to COCO format JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python jsonl_to_coco.py detections.jsonl
    python jsonl_to_coco.py detections.jsonl --output annotations.json
    python jsonl_to_coco.py detections.jsonl --category dressing_cup --min-score 0.5
        """
    )
    
    parser.add_argument(
        'jsonl_file',
        type=str,
        help='Path to input JSONL file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output JSON file (default: <jsonl_file>_coco.json)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default='cup',
        help='Category name (default: cup)'
    )
    
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        help='Minimum score threshold for detections (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.jsonl_file).exists():
        print(f"❌ Error: JSONL file not found: {args.jsonl_file}")
        return 1
    
    try:
        detections_jsonl_to_coco(args.jsonl_file, args.output, args.category, args.min_score)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
