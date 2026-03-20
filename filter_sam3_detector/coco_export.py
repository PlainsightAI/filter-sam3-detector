"""
Convert FilterSAM3Detector detections JSONL output into COCO-style JSON.

Input: line-delimited event records written by FILTER_OUTPUT_PATH.
Output: single JSON object with top-level keys:
  - images
  - annotations
  - categories
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_bbox_xywh(det: dict[str, Any]) -> list[float] | None:
    """Extract bbox as [x, y, w, h] from supported detection shapes."""
    bbox = det.get("bbox")
    if isinstance(bbox, dict):
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("width", 0.0))
        h = float(bbox.get("height", 0.0))
        return [x, y, max(0.0, w), max(0.0, h)]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = bbox
        return [float(x), float(y), max(0.0, float(w)), max(0.0, float(h))]

    box = det.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = [float(v) for v in box]
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    rois = det.get("rois")
    if isinstance(rois, list) and rois:
        roi0 = rois[0]
        if isinstance(roi0, (list, tuple)) and len(roi0) == 4:
            x1, y1, x2, y2 = [float(v) for v in roi0]
            return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    return None


def _extract_label(det: dict[str, Any]) -> str:
    for key in ("label", "class", "class_name", "category_name"):
        val = det.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "object"


def _extract_score(det: dict[str, Any]) -> float:
    if "confidence" in det:
        return float(det["confidence"])
    if "score" in det:
        return float(det["score"])
    return 0.0


def _extract_frame_detections(meta: dict[str, Any], output_label: str) -> list[dict[str, Any]]:
    # Preferred shape produced by _add_protege_compatible_output
    detections = meta.get("detections")
    if isinstance(detections, list):
        return [d for d in detections if isinstance(d, dict)]

    # Fallback to raw detector output arrays
    for key in (output_label, "sam3_detections", "detections"):
        arr = meta.get(key)
        if isinstance(arr, list):
            return [d for d in arr if isinstance(d, dict)]
    return []


def convert_jsonl_to_coco(input_path: Path, output_path: Path, output_label: str) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []

    category_to_id: dict[str, int] = {}
    image_id_remap: dict[int, int] = {}
    next_image_id = 1
    next_ann_id = 1

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %d: %s", line_no, exc)
                continue

            if not isinstance(record, dict):
                logger.warning("Skipping non-object JSON record at line %d", line_no)
                continue

            data = record.get("data", {})
            if not isinstance(data, dict):
                data = {}
            meta = data.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            src_image_id = meta.get("frame_id", data.get("id"))
            if not isinstance(src_image_id, int):
                src_image_id = next_image_id

            if src_image_id not in image_id_remap:
                image_id_remap[src_image_id] = next_image_id
                next_image_id += 1

                file_name = (
                    meta.get("filename")
                    or meta.get("frame_filename")
                    or f"frame_{image_id_remap[src_image_id]:06d}.jpg"
                )
                width = int(meta.get("width", 0) or 0)
                height = int(meta.get("height", 0) or 0)

                images.append(
                    {
                        "id": image_id_remap[src_image_id],
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                    }
                )

            image_id = image_id_remap[src_image_id]
            frame_detections = _extract_frame_detections(meta, output_label)

            for det in frame_detections:
                label = _extract_label(det)
                if label not in category_to_id:
                    category_to_id[label] = len(category_to_id) + 1
                    categories.append({"id": category_to_id[label], "name": label})

                bbox = _extract_bbox_xywh(det)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                area = float(det.get("area", w * h))
                score = _extract_score(det)

                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": category_to_id[label],
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": int(det.get("iscrowd", 0) or 0),
                        "score": score,
                    }
                )
                next_ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
        f.write("\n")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert detections.jsonl to COCO JSON")
    parser.add_argument(
        "--input",
        default="./output/detections.jsonl",
        help="Path to detections JSONL file",
    )
    parser.add_argument(
        "--output",
        default="./output/labels_coco.json",
        help="Path to output COCO JSON file",
    )
    parser.add_argument(
        "--output-label",
        default="sam3_detections",
        help="Fallback metadata key for raw detections array",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    coco = convert_jsonl_to_coco(input_path, output_path, args.output_label)
    print(
        f"Wrote COCO JSON: {output_path} "
        f"(images={len(coco['images'])}, annotations={len(coco['annotations'])}, categories={len(coco['categories'])})"
    )


if __name__ == "__main__":
    main()
