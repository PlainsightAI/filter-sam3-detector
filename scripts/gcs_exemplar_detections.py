#!/usr/bin/env python
"""
Generate detection annotations for all images in DATA_PATH.

Lists all images in DATA_PATH (and DATA_PATH/images/), runs FilterSAM3Detector for each
CLASS_CONFIG class on every image, and writes one detections.jsonl per class to
OUTPUT_ROOT/<class>/ (no frames or frames_annotated). Config: CLASS_CONFIG and optional
REF_IMAGES_BY_CLASS. Only local files are supported; gs:// is not allowed.

Usage:
    DATA_PATH=/path/to/data OUTPUT_ROOT=./out \\
        REF_IMAGES_ROOT=/path/to/refs python scripts/gcs_exemplar_detections.py

    MAX_IMAGES=5 DATA_PATH=/path/to/data ... python scripts/gcs_exemplar_detections.py
"""

import os
import json
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
from openfilter.filter_runtime.frame import Frame

from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", "")
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(_PROJECT_ROOT / "results_gcp")))
REF_IMAGES_ROOT = Path(os.getenv("REF_IMAGES_ROOT", str(Path.home() / "datasets" / "sg_samples")))
# Labels file: path relative to DATA_PATH (e.g. labels.jsonl) or absolute path
LABELS_FILENAME = os.getenv("LABELS_FILENAME", "labels.jsonl").strip()
DEVICE = os.getenv("FILTER_DEVICE", "cuda")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))

# Postprocess: area fraction, aspect ratio, NMS, top-k (env: POST_MIN_AREA_FRAC, etc.)
POST_MIN_AREA_FRAC = float(os.getenv("POST_MIN_AREA_FRAC", "0.003"))
POST_MAX_AREA_FRAC = float(os.getenv("POST_MAX_AREA_FRAC", "0.35"))
POST_MIN_AR = float(os.getenv("POST_MIN_AR", "0.25"))
POST_MAX_AR = float(os.getenv("POST_MAX_AR", "4.0"))
POST_NMS_IOU = float(os.getenv("POST_NMS_IOU", "0.5"))
POST_TOP_K = int(os.getenv("POST_TOP_K", "20"))

# Per-class postprocess overrides (e.g. min_area_frac for avocado). Keys: min_area_frac, max_area_frac, min_ar, max_ar, nms_iou, top_k.
POSTPROCESS_BY_CLASS = {
    "avocado": {"min_area_frac": 0.02},
}

# Class name -> (text_prompt, confidence_threshold). Override via CLASS_CONFIG_JSON env (JSON object) if needed.
CLASS_CONFIG = {
    "avocado": ("avocado in salad or avocado slices", 0.2),
    "roasted_chicken": ("chunks of cooked chicken (white meat)", 0.5),
    "miso_glazed_steelhead": ("cooked salmon fillet", 0.5),
    "hard_boiled_egg": ("boiled egg", 0.5),
    "caramelized_garlic_steak": ("blackened steak bites or diced steak or steak cubes", 0.2),
    "blackened_chicken": (
        "small cooked chicken chunks, light beige/tan, irregular pieces of meat",
        0.2,
    ),
    "roasted_tofu": ("tofu", 0.05),
    "warm_portobello_mix": ("small dark mushroom pieces", 0.2),
}

REF_IMAGES_DIR = REF_IMAGES_ROOT / "ref_images"
REF_IMAGES_NEGATIVE_DIR = REF_IMAGES_ROOT / "ref_images_negative"
# Class name -> (list of positive ref image paths, list of negative ref image paths). Optional.
REF_IMAGES_BY_CLASS = {
    "avocado": (
        [REF_IMAGES_DIR / "avocado2.png"],
        [
            # REF_IMAGES_NEGATIVE_DIR / "cucumber_1.png",
            # REF_IMAGES_NEGATIVE_DIR / "cucumber_2.png",
            # REF_IMAGES_NEGATIVE_DIR / "lime.png",
        ],
    ),
    "roasted_tofu": (
        [
            Path("/home/leandrobmarinho/datasets/sam3_ds/examples/tofu_example.png"),
            Path("/home/leandrobmarinho/datasets/sam3_ds/examples/tofu2.png"),
        ],
        [],
    ),
    # "blackened_chicken": (
    #     [REF_IMAGES_DIR / "blackened_chicken_example.png"],
    #     [REF_IMAGES_NEGATIVE_DIR / f"carrot_{i}.png" for i in (1, 2, 3)],
    # ),
    "caramelized_garlic_steak": (
        [REF_IMAGES_DIR / "steak1.png"],
        [],
    ),
}


def load_labels(data_path: str) -> dict[str, set[str]]:
    """Load LABELS_FILENAME (path relative to DATA_PATH or absolute). Returns dict[image_name, set of class names with present=True]. Empty dict if file missing or invalid."""
    labels_path = Path(LABELS_FILENAME)
    if not labels_path.is_absolute():
        labels_path = Path(data_path).resolve() / LABELS_FILENAME
    if not labels_path.is_file():
        return {}
    out = {}
    for line in labels_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            image_name = rec.get("image") or ""
            labels_obj = rec.get("labels") or {}
            present = {
                k for k, v in labels_obj.items()
                if isinstance(v, dict) and v.get("present") is True
            }
            if image_name:
                out[image_name] = present
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def list_all_images(data_path: str):
    """List all image filenames (basenames) in data_path and data_path/images/. Returns sorted unique list."""
    base = Path(data_path)
    seen = set()
    for subdir in (base, base / "images"):
        if not subdir.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            for p in subdir.glob(ext):
                seen.add(p.name)
    out = sorted(seen)
    if MAX_IMAGES > 0:
        out = out[:MAX_IMAGES]
        print(f"Limited to first {MAX_IMAGES} images (MAX_IMAGES={MAX_IMAGES})")
    return out


def load_image(data_path: str, image_filename: str):
    """Load image as BGR numpy array (cv2). Returns (img, path_used) or (None, None) if not found. path_used is the local path used for loading."""
    base = Path(data_path)
    for p in (base / "images" / image_filename, base / image_filename):
        if p.exists():
            img = cv2.imread(str(p))
            return (img, str(p.resolve())) if img is not None else (None, None)
    return (None, None)


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


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def apply_postprocess(
    detections: list,
    width: int,
    height: int,
    min_area_frac: float = POST_MIN_AREA_FRAC,
    max_area_frac: float = POST_MAX_AREA_FRAC,
    min_ar: float = POST_MIN_AR,
    max_ar: float = POST_MAX_AR,
    nms_iou: float = POST_NMS_IOU,
    top_k: int = POST_TOP_K,
) -> list:
    """Filter by area fraction and aspect ratio, run NMS, keep top_k by score. Returns list of detection dicts."""
    if not detections:
        return []
    area_img = width * height + 1e-6
    kept = []
    for i, d in enumerate(detections):
        bbox = d.get("bbox") or {}
        box = d.get("box")
        if box and len(box) >= 4:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        else:
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))
            w = float(bbox.get("width", 0))
            h = float(bbox.get("height", 0))
            x1, y1, x2, y2 = x, y, x + w, y + h
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        area_frac = area / area_img
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        ar = w / (h + 1e-6)
        if ar < min_ar or ar > max_ar:
            continue
        score = d.get("confidence") if d.get("confidence") is not None else d.get("score")
        score = float(score) if score is not None else -1.0
        kept.append((i, d, np.array([x1, y1, x2, y2], dtype=np.float32), score))
    if not kept:
        return []
    kept.sort(key=lambda x: x[3], reverse=True)
    keep_indices = []
    boxes_xyxy = np.array([x[2] for x in kept], dtype=np.float32)
    scores = np.array([x[3] for x in kept], dtype=np.float32)
    order = np.argsort(-scores)
    for idx in order:
        ok = True
        for j in keep_indices:
            if _iou_xyxy(boxes_xyxy[idx], boxes_xyxy[j]) >= nms_iou:
                ok = False
                break
        if ok:
            keep_indices.append(idx)
    keep_indices = sorted(keep_indices, key=lambda j: -scores[j])[:top_k]
    return [kept[j][1] for j in keep_indices]


def limit_detections_for_class(detections: list, class_name: str, top_k: int = 2) -> list:
    """For avocado, keep only the top_k detections by confidence; other classes unchanged."""
    if class_name != "avocado":
        return detections
    key = lambda d: d.get("confidence") if d.get("confidence") is not None else -1.0
    return sorted(detections, key=key, reverse=True)[:top_k]


def build_one_detector(class_name: str) -> FilterSAM3Detector:
    """Create and setup one FilterSAM3Detector for the given class (frees GPU after shutdown)."""
    if class_name not in CLASS_CONFIG:
        raise ValueError(f"Unknown class: {class_name}")
    prompt, conf = CLASS_CONFIG[class_name]
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
    return det


def main():
    if not DATA_PATH:
        print("Set DATA_PATH to the local folder containing labels and images.")
        return 1
    if DATA_PATH.startswith("gs://"):
        print("Only local DATA_PATH is supported; gs:// is not allowed.")
        return 1
    print(f"Data path: {DATA_PATH}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Ref images base: {REF_IMAGES_ROOT}")

    image_list = list_all_images(DATA_PATH)
    if not image_list:
        print(f"No images found under {DATA_PATH} (or {DATA_PATH}/images/).")
        return 0
    print(f"Found {len(image_list)} images in {DATA_PATH}")

    labels_by_image = load_labels(DATA_PATH)
    if labels_by_image:
        print(f"Loaded labels from {LABELS_FILENAME}: {len(labels_by_image)} images; will run SAM3 only for classes present per image.")
    else:
        print("No labels file or empty; running SAM3 for all classes on all images.")

    for class_name in sorted(CLASS_CONFIG.keys()):
        # If we have labels, only run SAM3 for images where this class is present; still write one line per image (empty detections when skipped).
        if labels_by_image:
            image_set_for_class = {f for f in image_list if class_name in labels_by_image.get(f, set())}
        else:
            image_set_for_class = set(image_list)
        run_count = len(image_set_for_class)
        print(f"\n--- {class_name} (loading model, running SAM3 on {run_count} images; output 1 line per image, {len(image_list)} total) ---")
        detector = build_one_detector(class_name)
        out_dir = OUTPUT_ROOT / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_used = CLASS_CONFIG[class_name][0]
        written = 0
        try:
            with open(out_dir / "detections.jsonl", "w") as out_f:
                for idx, image_filename in enumerate(image_list):
                    if image_filename not in image_set_for_class:
                        # Same pattern: one record per image; no SAM3, empty detections
                        img, image_path = load_image(DATA_PATH, image_filename)
                        if img is None:
                            print(f"  Skip (could not load): {image_filename}")
                            continue
                        h, w = img.shape[:2]
                        rec = {
                            "image": image_filename,
                            "path": image_path or str(Path(DATA_PATH) / image_filename),
                            "class": class_name,
                            "prompt": prompt_used,
                            "score": None,
                            "width": w,
                            "height": h,
                            "detections": [],
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        continue
                    img, image_path = load_image(DATA_PATH, image_filename)
                    if img is None:
                        print(f"  Skip (could not load): {image_filename}")
                        continue
                    h, w = img.shape[:2]
                    frame = Frame(
                        image=img,
                        data={"meta": {"id": idx, "path": image_path}},
                        format="BGR",
                    )
                    out = detector.process({"main": frame})
                    raw = out["main"].data.get("meta", {}).get("sam3_detections", [])
                    detections = normalize_detections_to_standard(list(raw), class_name)
                    pp_kw = dict(
                        min_area_frac=POST_MIN_AREA_FRAC,
                        max_area_frac=POST_MAX_AREA_FRAC,
                        min_ar=POST_MIN_AR,
                        max_ar=POST_MAX_AR,
                        nms_iou=POST_NMS_IOU,
                        top_k=POST_TOP_K,
                    )
                    pp_kw.update(POSTPROCESS_BY_CLASS.get(class_name, {}))
                    detections = apply_postprocess(detections, w, h, **pp_kw)
                    detections = limit_detections_for_class(detections, class_name, top_k=2)
                    score = max((d.get("confidence") for d in detections if d.get("confidence") is not None), default=None)
                    rec = {
                        "image": image_filename,
                        "path": image_path,
                        "class": class_name,
                        "prompt": prompt_used,
                        "score": score,
                        "width": w,
                        "height": h,
                        "detections": detections,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 10 == 0 or written == 1:
                        print(f"  {class_name}: {written} lines written")
        finally:
            detector.shutdown()
        print(f"  {class_name}: done ({written} lines, SAM3 ran on {run_count} images), detector shut down (GPU freed)")

    print(f"\nDone. Output: {OUTPUT_ROOT}")
    print("  One detections.jsonl per class under <class>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
