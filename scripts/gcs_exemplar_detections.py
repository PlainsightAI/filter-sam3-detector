#!/usr/bin/env python
"""
Generate detection annotations for multilabel dataset (labels.jsonl + images).

Reads DATA_PATH (local or gs://) containing labels.jsonl and images. For each image,
runs FilterSAM3Detector only for classes with present=true, and writes one
detections.jsonl per class to OUTPUT_ROOT/<class>/ (no frames or frames_annotated).
Config: CLASS_CONFIG (text_prompt, confidence per class) and optional REF_IMAGES_BY_CLASS.

Usage:
    DATA_PATH=/path/to/data OUTPUT_ROOT=./out \\
        REF_IMAGES_ROOT=/path/to/refs python scripts/exemplar_detections.py

    MAX_IMAGES=5 DATA_PATH=/path/to/data ... python scripts/exemplar_detections.py

    DATA_PATH=gs://bucket/path/to/data ... python scripts/exemplar_detections.py
"""

import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np
from openfilter.filter_runtime.frame import Frame

from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", "")
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(_PROJECT_ROOT / "sg_samples_annotations")))
REF_IMAGES_ROOT = Path(os.getenv("REF_IMAGES_ROOT", str(_PROJECT_ROOT / "sg_samples")))
DEVICE = os.getenv("FILTER_DEVICE", "cuda")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))

# Class name -> (text_prompt, confidence_threshold). Override via CLASS_CONFIG_JSON env (JSON object) if needed.
CLASS_CONFIG = {
    "avocado": ("avocado in salad or avocado slices", 0.5),
    "roasted_chicken": ("chunks of cooked chicken (white meat)", 0.5),
    "miso_glazed_steelhead": ("cooked salmon fillet", 0.5),
    "hard_boiled_egg": ("boiled egg", 0.5),
    "caramelized_garlic_steak": ("blackened steak bites or diced steak or steak cubes", 0.5),
    "blackened_chicken": (
        "small chunks of chicken (light beige), bite-sized pieces",
        0.55,
    ),
    "roasted_tofu": ("tofu", 0.4),
    "warm_portobello_mix": ("dark mushroom mix", 0.5),
}

REF_IMAGES_DIR = REF_IMAGES_ROOT / "ref_images"
REF_IMAGES_NEGATIVE_DIR = REF_IMAGES_ROOT / "ref_images_negative"
# Class name -> (list of positive ref image paths, list of negative ref image paths). Optional.
REF_IMAGES_BY_CLASS = {
    "roasted_tofu": (
        [REF_IMAGES_DIR / "tofu_example.png"],
        [REF_IMAGES_NEGATIVE_DIR / f"carrot_{i}.png" for i in (1, 2, 3)],
    ),
    "blackened_chicken": (
        [REF_IMAGES_DIR / "blackened_chicken_example.png"],
        [REF_IMAGES_NEGATIVE_DIR / f"carrot_{i}.png" for i in (1, 2, 3)],
    ),
}


def load_labels_jsonl(data_path: str, is_gcs: bool):
    """Load labels.jsonl and return list of (image_filename, list_of_present_classes)."""
    entries = []
    if is_gcs:
        try:
            from google.cloud import storage
            from urllib.parse import urlparse
            parsed = urlparse(data_path)
            bucket_name = parsed.netloc
            prefix = (parsed.path or "").lstrip("/")
            blob_path = f"{prefix}/labels.jsonl" if prefix else "labels.jsonl"
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if not line:
                    continue
                obj = json.loads(line)
                present = [k for k, v in (obj.get("labels") or {}).items() if v.get("present")]
                if present and obj.get("image"):
                    entries.append((obj["image"], present))
        except ImportError:
            raise RuntimeError("GCS path used but google-cloud-storage not installed. pip install google-cloud-storage")
    else:
        path = Path(data_path) / "labels.jsonl"
        if not path.exists():
            path = Path(data_path) / ".." / "labels.jsonl"
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"labels.jsonl not found under {data_path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                present = [k for k, v in (obj.get("labels") or {}).items() if v.get("present")]
                if present and obj.get("image"):
                    entries.append((obj["image"], present))
    if MAX_IMAGES > 0:
        entries = entries[:MAX_IMAGES]
        print(f"Limited to first {MAX_IMAGES} images (MAX_IMAGES={MAX_IMAGES})")
    return entries


def load_image_bytes_gcs(data_path: str, image_filename: str):
    """Download image from GCS and return bytes."""
    from google.cloud import storage
    from urllib.parse import urlparse
    parsed = urlparse(data_path)
    bucket_name = parsed.netloc
    prefix = (parsed.path or "").lstrip("/")
    # Try data_path/images/<image> then data_path/<image>
    for blob_path in (f"{prefix}/images/{image_filename}", f"{prefix}/{image_filename}") if prefix else (f"images/{image_filename}", image_filename):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if blob.exists():
            return blob.download_as_bytes()
    raise FileNotFoundError(f"Image not found: {image_filename} under {data_path}")


def load_image(data_path: str, image_filename: str, is_gcs: bool):
    """Load image as BGR numpy array (cv2). Returns None if failed."""
    if is_gcs:
        raw = load_image_bytes_gcs(data_path, image_filename)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    # Local: try data_path/images/<name> then data_path/<name>
    base = Path(data_path)
    for p in (base / "images" / image_filename, base / image_filename):
        if p.exists():
            img = cv2.imread(str(p))
            return img
    return None


def _detection_to_standard(d: dict, class_name: str) -> dict:
    """One detection: only class, confidence, bbox {x,y,width,height}, rois (same as results/detections.jsonl)."""
    score = d.get("score") if d.get("score") is not None else d.get("confidence")
    box = d.get("box")
    if box and len(box) >= 4:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
    else:
        bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
    rois = d.get("rois") or ([list(d["box"])] if d.get("box") else [])
    return {"class": class_name, "confidence": score, "bbox": bbox, "rois": rois}


def normalize_detections_to_standard(detections: list, class_name: str):
    """Convert each detection to standard format: class, confidence, bbox, rois only."""
    return [_detection_to_standard(d, class_name) for d in detections]


def build_detectors():
    """Create one FilterSAM3Detector per class from CLASS_CONFIG."""
    detectors = {}
    for class_name, (prompt, conf) in CLASS_CONFIG.items():
        config_kw = dict(
            text_prompt=prompt,
            confidence_threshold=conf,
            device=DEVICE,
            output_boxes=True,
            output_scores=True,
            output_masks=False,
        )
        if class_name in REF_IMAGES_BY_CLASS:
            ref_images, ref_images_negative = REF_IMAGES_BY_CLASS[class_name]
            config_kw["ref_images"] = [str(p) for p in ref_images]
            config_kw["ref_images_negative"] = [str(p) for p in ref_images_negative]
        config = FilterSAM3DetectorConfig(**config_kw)
        config = FilterSAM3Detector.normalize_config(config)
        if class_name in REF_IMAGES_BY_CLASS:
            ref_images, ref_images_negative = REF_IMAGES_BY_CLASS[class_name]
            config["ref_images"] = [str(p) for p in ref_images]
            config["ref_images_negative"] = [str(p) for p in ref_images_negative]
        else:
            config["ref_images"] = None
            config["ref_images_negative"] = None
        det = FilterSAM3Detector(config)
        det.setup(config)
        detectors[class_name] = det
    return detectors


def main():
    if not DATA_PATH:
        print("Set DATA_PATH (local path or gs://...) to the folder containing labels.jsonl and images.")
        return 1
    is_gcs = DATA_PATH.startswith("gs://")
    print(f"Data path: {DATA_PATH} (GCS={is_gcs})")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Ref images base: {REF_IMAGES_ROOT}")

    entries = load_labels_jsonl(DATA_PATH, is_gcs)
    if not entries:
        print("No entries in labels.jsonl (or none with present classes).")
        return 0
    print(f"Loaded {len(entries)} image entries from labels.jsonl")

    detectors = build_detectors()
    # Open one jsonl file per class
    out_handles = {}
    for class_name in detectors:
        out_dir = OUTPUT_ROOT / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_handles[class_name] = open(out_dir / "detections.jsonl", "w")

    try:
        for idx, (image_filename, present_classes) in enumerate(entries):
            img = load_image(DATA_PATH, image_filename, is_gcs)
            if img is None:
                print(f"  Skip (could not load): {image_filename}")
                continue
            image_path_or_uri = f"{DATA_PATH}/{image_filename}" if is_gcs else str(Path(DATA_PATH) / image_filename)
            frame = Frame(
                image=img,
                data={"meta": {"id": idx, "path": image_path_or_uri}},
                format="BGR",
            )
            for class_name in present_classes:
                if class_name not in detectors:
                    continue
                prompt_used = CLASS_CONFIG[class_name][0]
                out = detectors[class_name].process({"main": frame})
                raw = out["main"].data.get("meta", {}).get("sam3_detections", [])
                detections = normalize_detections_to_standard(list(raw), class_name)
                score = max((d.get("confidence") for d in detections if d.get("confidence") is not None), default=None)
                h, w = img.shape[:2]
                rec = {
                    "image": image_filename,
                    "path": image_path_or_uri,
                    "class": class_name,
                    "prompt": prompt_used,
                    "score": score,
                    "width": w,
                    "height": h,
                    "detections": detections,
                }
                out_handles[class_name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Processed {idx + 1}/{len(entries)} images")
    finally:
        for f in out_handles.values():
            f.close()
        for det in detectors.values():
            det.shutdown()

    print(f"\nDone. Output: {OUTPUT_ROOT}")
    print("  One detections.jsonl per class under <class>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
