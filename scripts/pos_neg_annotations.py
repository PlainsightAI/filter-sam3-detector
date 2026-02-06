#!/usr/bin/env python
"""
Generate detection annotations for sg_samples dataset using FilterSAM3Detector.

Expected input structure:
    SG_SAMPLES_ROOT/
        avocado/pos/, avocado/neg/
        roasted_chicken/pos/, roasted_chicken/neg/
        ... (one folder per ingredient, each with pos/ and neg/)
        ref_images/           # optional: positive exemplar images (e.g. tofu_example.png)
        ref_images_negative/  # optional: negative exemplar images (e.g. carrot_1.png, ...)

For each ingredient, runs SAM3 with the configured prompt on pos and neg folders
and writes annotations into separate output folders, e.g.:
    OUTPUT_ROOT/
        avocado/pos/
            detections.jsonl
            frames/           # original images
            frames_annotated/ # images with bounding boxes drawn
        avocado/neg/
            detections.jsonl, frames/, frames_annotated/
        roasted_chicken/pos/
            ...

Some ingredients (e.g. roasted_tofu, blackened_chicken) use ref_images and
ref_images_negative from SG_SAMPLES_ROOT/ref_images/ and
SG_SAMPLES_ROOT/ref_images_negative/ for exemplar-based detection.

Usage:
    # Default paths (override with env if needed)
    python scripts/generate_sg_annotations.py

    # Run only one ingredient (quick test)
    INGREDIENT=roasted_tofu python scripts/generate_sg_annotations.py

    # Custom paths
    SG_SAMPLES_ROOT=/path/to/sg_samples OUTPUT_ROOT=/path/to/out python scripts/generate_sg_annotations.py
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
from openfilter.filter_runtime.frame import Frame

from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig


# Ingredient folder name -> (text_prompt, confidence_threshold)
INGREDIENT_CONFIG = {
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

# Project root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SG_SAMPLES_ROOT = Path(os.getenv("SG_SAMPLES_ROOT", str(_PROJECT_ROOT / "sg_samples")))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(_PROJECT_ROOT / "sg_samples_annotations")))
DEVICE = os.getenv("FILTER_DEVICE", "cuda")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REF_IMAGES_DIR = SG_SAMPLES_ROOT / "ref_images"
REF_IMAGES_NEGATIVE_DIR = SG_SAMPLES_ROOT / "ref_images_negative"

# Ingredient -> (ref_images_list, ref_images_negative_list) for exemplar-based detection
REF_IMAGES_BY_INGREDIENT = {
    "avocado": (
        [],
        [
            # REF_IMAGES_NEGATIVE_DIR / "lime.png",
            REF_IMAGES_NEGATIVE_DIR / "cucumber_1.png",
            # REF_IMAGES_NEGATIVE_DIR / "cucumber_2.png",
            
        ],
    ),
    "roasted_tofu": (
        [REF_IMAGES_DIR / "tofu_example.png"],
        [REF_IMAGES_NEGATIVE_DIR / f"carrot_{i}.png" for i in (1, 2, 3)],
    ),
    "blackened_chicken": (
        [REF_IMAGES_DIR / "blackened_chicken_example.png"],
        [REF_IMAGES_NEGATIVE_DIR / f"carrot_{i}.png" for i in (1, 2, 3)],
    ),
}


def draw_detections_on_image(image, detections):
    """Draw bounding boxes and scores on a BGR image. Returns a new image."""
    out = image.copy()
    for det in detections:
        if "box" not in det:
            continue
        x1, y1, x2, y2 = det["box"]
        color = (0, 255, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if "score" in det:
            label = f"{det['score']:.2f}"
            cv2.putText(out, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def collect_image_paths(folder: Path):
    paths = []
    for p in folder.iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)


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


def _normalize_detections_to_standard(detections: list, class_name: str):
    """Convert each detection to standard format: class, confidence, bbox, rois only."""
    return [_detection_to_standard(d, class_name) for d in detections]


def run_detector_on_folder(detector, folder: Path, output_dir: Path, class_name: str, split: str, prompt: str):
    """Run detector on all images; save detections.jsonl, frames/, and frames_annotated/."""
    paths = collect_image_paths(folder)
    if not paths:
        print(f"    No images in {folder}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_annotated_dir = output_dir / "frames_annotated"
    frames_dir.mkdir(exist_ok=True)
    frames_annotated_dir.mkdir(exist_ok=True)
    records = []

    for i, img_path in enumerate(paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    Skip (not readable): {img_path.name}")
            continue
        frame = Frame(
            image=img,
            data={"meta": {"id": i, "path": str(img_path)}},
            format="BGR",
        )
        out = detector.process({"main": frame})
        raw = out["main"].data.get("meta", {}).get("sam3_detections", [])
        raw_list = list(raw)
        detections_standard = _normalize_detections_to_standard(raw_list, class_name)
        score = max((d.get("confidence") for d in detections_standard if d.get("confidence") is not None), default=None)
        h, w = img.shape[:2]
        records.append({
            "image": img_path.name,
            "path": str(img_path),
            "class": class_name,
            "prompt": prompt,
            "score": score,
            "width": w,
            "height": h,
            "detections": detections_standard,
        })
        # Save original frame
        frame_path = frames_dir / img_path.name
        cv2.imwrite(str(frame_path), img)
        # Save annotated frame (with boxes; draw uses raw detections with box/score)
        img_annotated = draw_detections_on_image(img, raw_list)
        cv2.imwrite(str(frames_annotated_dir / img_path.name), img_annotated)

    out_jsonl = output_dir / "detections.jsonl"
    with open(out_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    {split}: {len(records)} images -> {output_dir}")
    print(f"      detections.jsonl, frames/, frames_annotated/")
    return len(records)


def main():
    total_images = 0
    single_ingredient = os.getenv("INGREDIENT", "").strip()
    if single_ingredient and single_ingredient not in INGREDIENT_CONFIG:
        print(f"Unknown INGREDIENT={single_ingredient}, running all. Valid: {list(INGREDIENT_CONFIG.keys())}")
        single_ingredient = None
    items_to_run = (
        [(single_ingredient, INGREDIENT_CONFIG[single_ingredient])]
        if single_ingredient
        else list(INGREDIENT_CONFIG.items())
    )

    for ingredient, (prompt, conf) in items_to_run:
        ingredient_path = SG_SAMPLES_ROOT / ingredient
        if not ingredient_path.exists():
            print(f"Skipping (missing): {ingredient_path}")
            continue

        config_kw = dict(
            text_prompt=prompt,
            confidence_threshold=conf,
            device=DEVICE,
            output_boxes=True,
            output_scores=True,
            output_masks=False,
        )
        if ingredient in REF_IMAGES_BY_INGREDIENT:
            ref_images, ref_images_negative = REF_IMAGES_BY_INGREDIENT[ingredient]
            config_kw["ref_images"] = [str(p) for p in ref_images]
            config_kw["ref_images_negative"] = [str(p) for p in ref_images_negative]
        config = FilterSAM3DetectorConfig(**config_kw)
        config = FilterSAM3Detector.normalize_config(config)
        if ingredient in REF_IMAGES_BY_INGREDIENT:
            ref_images, ref_images_negative = REF_IMAGES_BY_INGREDIENT[ingredient]
            config["ref_images"] = [str(p) for p in ref_images]
            config["ref_images_negative"] = [str(p) for p in ref_images_negative]
        else:
            # Do not use ref_images from .env for other ingredients (text-only detection)
            config["ref_images"] = None
            config["ref_images_negative"] = None
        detector = FilterSAM3Detector(config)
        detector.setup(config)

        print(f"\n{ingredient} (prompt={prompt[:50]}..., conf={conf})")
        if ingredient in REF_IMAGES_BY_INGREDIENT:
            print("  using ref_images / ref_images_negative")
        for split in ("pos", "neg"):
            folder = ingredient_path / split
            if not folder.is_dir():
                print(f"  Skip (no dir): {folder}")
                continue
            out_dir = OUTPUT_ROOT / ingredient / split
            n = run_detector_on_folder(detector, folder, out_dir, ingredient, split, prompt)
            total_images += n

        detector.shutdown()

    print(f"\nTotal images annotated: {total_images}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
