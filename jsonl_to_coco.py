#!/usr/bin/env python3
"""Convert JSONL detections to COCO format."""
import json
import sys
from pathlib import Path
from datetime import datetime

jsonl_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

coco = {"info": {"date_created": datetime.now().isoformat()}, "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "cup"}]}

for line in open(jsonl_path):
    det = json.loads(line)
    img_id = len(coco["images"]) + 1
    frame_file = f"frame_{det['frame']:06d}.jpg"
    coco["images"].append({"id": img_id, "file_name": frame_file,
                           "width": det["width"], "height": det["height"]})
    for d in det["detections"]:
        x1, y1, x2, y2 = d["box"]
        coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id,
                                    "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1],
                                    "area": (x2-x1)*(y2-y1), "score": d["score"], "iscrowd": 0})

json.dump(coco, open(output_path, "w"), indent=2)
print(f"Wrote {len(coco['images'])} images, {len(coco['annotations'])} annotations to {output_path}")
