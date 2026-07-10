#!/usr/bin/env python3
"""
benchmark_sam3_video_memory.py

Benchmark the memory usage of HuggingFace Sam3VideoModel while streaming
frames one-by-one, while optionally saving cropped objects, annotated frames,
and tinted segmentation overlays.

Example:

python benchmark_sam3_video_memory.py \
    --video data/bread_occlusion_fps5.mp4 \
    --prompt bread

python benchmark_sam3_video_memory.py \
    --video data/bread_occlusion_fps5.mp4 \
    --prompt bread \
    --prune-every 500
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()  # Load environment variables from .env file if present

from transformers import (
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video

################################################################################
# Visualization & Saving Utilities (Imported from test_transformers_stream.py)
################################################################################


def frame_to_bgr(frame):
    if isinstance(frame, Image.Image):
        return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[2] == 3:
        return cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

    return frame_array.copy()


def draw_top_right_counts(image, counts):
    lines = [f"{count} {class_name}" for class_name, count in counts.items() if count > 0]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 10
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_width = max(width for width, _ in text_sizes) + padding * 2
    box_height = sum(height for _, height in text_sizes) + line_gap * (len(lines) - 1) + padding * 2

    image_height, image_width = image.shape[:2]
    x1 = max(0, image_width - box_width - 10)
    y1 = 10
    x2 = min(image_width - 1, x1 + box_width)
    y2 = min(image_height - 1, y1 + box_height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)

    text_y = y1 + padding + text_sizes[0][1]
    for line, (text_width, text_height) in zip(lines, text_sizes):
        text_x = x2 - padding - text_width
        cv2.putText(image, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        text_y += text_height + line_gap


def mask_to_bool_array(mask, fill_holes=True, remove_sprinkles=True, sprinkle_threshold=150):
    mask_array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)

    while mask_array.ndim > 2 and 1 in mask_array.shape:
        mask_array = np.squeeze(mask_array)

    if mask_array.ndim == 3:
        mask_array = mask_array[0]

    if mask_array.ndim != 2:
        return None

    binary_mask = (mask_array > 0).astype(np.uint8) * 255
    if not np.any(binary_mask):
        return binary_mask > 0

    if fill_holes:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(binary_mask, contours, i, 255, -1)

    if remove_sprinkles:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        cleaned_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= sprinkle_threshold:
                cleaned_mask[labels == i] = 255
        binary_mask = cleaned_mask

    return binary_mask > 200


def mask_to_bbox(mask):
    mask_bool = mask_to_bool_array(mask)
    if mask_bool is None or not np.any(mask_bool):
        return None

    ys, xs = np.where(mask_bool)
    xmin = int(xs.min())
    ymin = int(ys.min())
    xmax = int(xs.max()) + 1
    ymax = int(ys.max()) + 1
    return xmin, ymin, xmax, ymax


def save_class_tinted_overlay(frame_bgr, object_ids, masks, id_to_clean_class, class_colors, output_path):
    grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR).astype(np.float32)

    for i, obj_id in enumerate(object_ids):
        obj_id_int = int(obj_id.item() if torch.is_tensor(obj_id) else obj_id)
        class_name = id_to_clean_class.get(obj_id_int, "unknown")
        
        if i >= len(masks):
            continue

        mask_bool = mask_to_bool_array(masks[i])
        if mask_bool is None or not np.any(mask_bool):
            continue

        color = np.array(class_colors.get(class_name, (0, 255, 255)), dtype=np.float32)
        blended = overlay[mask_bool] * 0.35 + color * 0.65
        overlay[mask_bool] = blended

    cv2.imwrite(output_path, np.clip(overlay, 0, 255).astype(np.uint8))


################################################################################
# Core Utilities
################################################################################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Input video.",
    )

    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt. Can be specified multiple times.",
    )

    parser.add_argument(
        "--prune-every",
        type=int,
        default=0,
        help="Restart inference session every N frames. 0 = never.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Frames processed before printing progress.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("memory_benchmark"),
    )

    parser.add_argument(
        "--model",
        default="facebook/sam3",
    )

    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )

    return parser.parse_args()


def get_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def gpu_stats(device):
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024 ** 3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024 ** 3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024 ** 3,
    }


def cpu_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 3

def log_memory(prefix, device):
    cpu = cpu_memory_gb()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(
            f"{prefix}"
            f"CPU={cpu:.2f} GB | "
            f"GPU alloc={alloc:.2f} GB | "
            f"GPU reserved={reserved:.2f} GB"
        )
    else:
        print(f"{prefix}CPU={cpu:.2f} GB")

################################################################################
# Session Helpers
################################################################################

def create_session(processor, prompts, device, dtype):
    session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    for prompt in prompts:
        session = processor.add_text_prompt(
            inference_session=session,
            text=prompt,
        )

    return session


################################################################################
# Benchmark Main Execution
################################################################################


def main():

    args = parse_args()

    # Define paths for artifact retention
    crop_base_dir = args.output_dir / "saved_bboxes"
    annotated_dir = args.output_dir / "annotated_frames"
    overlay_dir = args.output_dir / "segmentation_overlays"

    crop_base_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically build maps for short names and directory labels from prompt names
    long_prompt_to_short_name = {}
    all_short_names = []
    for prompt in args.prompt:
        short_name = prompt.lower().replace(" ", "_")[:20]
        long_prompt_to_short_name[prompt] = short_name
        all_short_names.append(short_name)
        (crop_base_dir / short_name).mkdir(parents=True, exist_ok=True)

    class_colors = {
        "bread": (0, 215, 255),
        "cheese": (0, 255, 128),
    }

    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

    dtype = get_dtype(args.dtype)

    print("Loading model...")

    model = Sam3VideoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)

    processor = Sam3VideoProcessor.from_pretrained(args.model)

    print("Loading video...")

    video_frames, _ = load_video(str(args.video))
    # if args.prune_every > 0:
    #     black_frame = np.zeros_like(video_frames[0])
    #     # verify that the model can redetect objects after a session restart by inserting black frames
    #     video_frames = np.concatenate([video_frames[0:args.prune_every-15], [black_frame] * (args.prune_every)], axis=0)
    video_frames = np.concatenate([video_frames] * 100, axis=0)  # Duplicate frames to simulate a longer video
    print(f"{len(video_frames)} frames")

    session = create_session(
        processor,
        args.prompt,
        device,
        dtype,
    )

    csv_path = args.output_dir / f"memory_prune_every_{args.prune_every}.csv"

    # Store mask data across iterations for session restarts
    last_mask_data = None

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "restart",
                "allocated_gb",
                "reserved_gb",
                "max_allocated_gb",
                "max_reserved_gb",
                "cpu_gb",
                "frame_time_ms",
                "num_objects",
            ]
        )

        for frame_idx, frame in enumerate(video_frames):

            restarted = False

            if (
                args.prune_every > 0
                and frame_idx > 0
                and frame_idx % args.prune_every == 0
            ):
                print(f"\n===== Pruning old session memory @ frame {frame_idx} (Preserving IDs) =====")


                history_buffer_window = 30
                cutoff_frame = frame_idx - history_buffer_window

                log_memory("Before prune: ", device)

                #
                # ------------------------------------------------------------------
                # GPU MEMORY
                # ------------------------------------------------------------------
                #

                # if hasattr(session, "output_dict_per_obj"):
                #     logged=False
                #     print("output_dict_per_obj keys:", session.output_dict_per_obj.keys())
                #     for obj_store in session.output_dict_per_obj.values():
                #         print("obj_store keys:", obj_store.keys())
                #         for frame_dict_key, frame_dict in obj_store.items():
                #             print(f"Frame dict '{frame_dict_key}' keys:", frame_dict.keys())
                #             if isinstance(frame_dict, dict):
                #                 for f_idx in list(frame_dict.keys()):
                #                     print(f"Checking frame_dict for key: {f_idx}")
                #                     print(frame_dict[f_idx].keys())
                #                     if isinstance(f_idx, int) and f_idx < cutoff_frame:
                #                         del frame_dict[f_idx]
                if hasattr(session, "output_dict_per_obj"):
                    print("output_dict_per_obj keys:", session.output_dict_per_obj.keys())
                    for obj_idx, obj_store in session.output_dict_per_obj.items():
                        # 1. STRICTLY target non-conditioning frames
                        if "non_cond_frame_outputs" in obj_store:
                            non_cond_dict = obj_store["non_cond_frame_outputs"]
                            print(f"Object {obj_idx} non_cond_frame_outputs keys before pruning: {list(non_cond_dict.keys())}")
                            for f_idx in list(non_cond_dict.keys()):
                                print(f"Checking non_cond_dict for key: {f_idx}")
                                print(non_cond_dict[f_idx].keys())
                                if isinstance(f_idx, int) and f_idx < cutoff_frame:
                                    del non_cond_dict[f_idx]
                            print(f"Object {obj_idx} non_cond_frame_outputs keys after pruning: {list(non_cond_dict.keys())}")
                        # Optional: If you MUST prune conditioning frames because of extreme memory limits,
                        # ensure you keep at least the very first frame (the initial prompt)
                        if "cond_frame_outputs" in obj_store:
                            cond_dict = obj_store["cond_frame_outputs"]
                            
                            if len(cond_dict) > 10:
                                print(f"Object {obj_idx} cond_frame_outputs keys before pruning: {list(cond_dict.keys())}")
                                sorted_keys = sorted(cond_dict.keys())
                                
                                # Identify the exact keys to keep
                                keys_to_keep = set(sorted_keys[:1] + sorted_keys[-9:])
                                
                                # Find the keys to delete by finding the difference between sets
                                keys_to_delete = set(cond_dict.keys()) - keys_to_keep
                                
                                for f_idx in keys_to_delete:
                                    if isinstance(f_idx, int):
                                        del cond_dict[f_idx]
                                        
                                print(f"Object {obj_idx} cond_frame_outputs keys after pruning: {list(cond_dict.keys())}")
                #
                # ------------------------------------------------------------------
                # CPU MEMORY
                # ------------------------------------------------------------------
                #

                # Cached processed image tensors
                if hasattr(session, "processed_frames"):
                    before = len(session.processed_frames)
                    for f_idx in list(session.processed_frames.keys()):
                        if f_idx < cutoff_frame:
                            del session.processed_frames[f_idx]
                    after = len(session.processed_frames)

                # tracker_meta = getattr(session, "tracker_metadata", None)

                # if tracker_meta is not None:

                #     score_history = tracker_meta.get("obj_id_to_tracker_score_frame_wise")
                #     if isinstance(score_history, dict):
                #         before = len(score_history)
                #         for f_idx in list(score_history.keys()):
                #             if f_idx < cutoff_frame:
                #                 del score_history[f_idx]
                #         print(f"tracker_score_history: {before} -> {len(score_history)}")

                #     suppressed = tracker_meta.get("suppressed_obj_ids")
                #     if isinstance(suppressed, dict):
                #         before = len(suppressed)
                #         for f_idx in list(suppressed.keys()):
                #             if f_idx < cutoff_frame:
                #                 del suppressed[f_idx]
                #         print(f"suppressed_obj_ids: {before} -> {len(suppressed)}")

                #     overlap = tracker_meta.get("overlap_pair_to_frame_inds")
                #     if isinstance(overlap, dict):
                #         removed = 0
                #         for pair, frame_list in overlap.items():
                #             if isinstance(frame_list, list):
                #                 old = len(frame_list)
                #                 overlap[pair] = [f for f in frame_list if f >= cutoff_frame]
                #                 removed += old - len(overlap[pair])
                #         print(f"overlap history removed {removed} frame references")

                #     unmatched = tracker_meta.get("unmatched_frame_inds")
                #     if isinstance(unmatched, dict):
                #         removed = 0
                #         for obj_id, frame_list in unmatched.items():
                #             if isinstance(frame_list, list):
                #                 old = len(frame_list)
                #                 unmatched[obj_id] = [f for f in frame_list if f >= cutoff_frame]
                #                 removed += old - len(unmatched[obj_id])
                #         print(f"unmatched history removed {removed} frame references")

                #
                # ------------------------------------------------------------------
                # Cleanup
                # ------------------------------------------------------------------
                #

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                log_memory("After prune : ", device)

               

                # Keep this True so your CSV logger still tracks when a memory clear occurs
                restarted = True

            frame_cv = frame_to_bgr(frame)
            frame_height, frame_width = frame_cv.shape[:2]

            inputs = processor(
                images=frame,
                return_tensors="pt",
            ).to(device)

            if dtype != torch.float32:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            start = time.perf_counter()

            outputs = model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                frame_idx=frame_idx,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            processed = processor.postprocess_outputs(
                session,
                outputs,
                original_sizes=inputs.original_sizes,
            )

            num_objects = len(processed.get("object_ids", []))
            print(processed.keys())
            # Update tracked mask data with current frame results
            # =====================================================================
            # Visual Processing and Saving Execution Engine
            # =====================================================================
            annotated_frame = frame_cv.copy()
            masks = processed.get("masks", processed.get("masks", []))
            scores = processed.get("scores", [])
            object_ids = processed.get("object_ids", [])
            class_counts = {class_name: 0 for class_name in all_short_names}
            
            prompt_to_obj_ids = processed.get("prompt_to_obj_ids", {})
            id_to_clean_class = {}

            for long_prompt, detected_ids in prompt_to_obj_ids.items():
                short_name = long_prompt_to_short_name.get(long_prompt, "unknown_text")
                for oid in detected_ids:
                    id_to_clean_class[int(oid)] = short_name

            for i, obj_id in enumerate(object_ids):
                obj_id_int = int(obj_id.item() if torch.is_tensor(obj_id) else obj_id)
                score_tensor = scores[i] if len(scores) > i else torch.tensor(0.0)
                score = float(score_tensor.item() if torch.is_tensor(score_tensor) else score_tensor)
                
                bbox = mask_to_bbox(masks[i]) if len(masks) > i else None
                short_name = id_to_clean_class.get(obj_id_int, f"object_{obj_id_int}")
                
                if short_name in class_counts:
                    class_counts[short_name] += 1

                if bbox is None:
                    continue

                xmin, ymin, xmax, ymax = bbox
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(frame_width, xmax), min(frame_height, ymax)
                
                if xmax > xmin and ymax > ymin:
                    crop = frame_cv[ymin:ymax, xmin:xmax]
                    crop_filename = f"frame_{frame_idx:04d}_id_{obj_id_int}.jpg"
                    crop_path = crop_base_dir / short_name / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                
                label = f"{short_name} {obj_id_int} ({score:.2f})"
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_top = max(0, ymin - text_h - 10)
                cv2.rectangle(annotated_frame, (xmin, label_top), (xmin + text_w, ymin), (0, 255, 0), -1)
                
                text_y = ymin - 5 if ymin - 5 > text_h else min(frame_height - 5, ymax + text_h + 5)
                cv2.putText(annotated_frame, label, (xmin, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            
            draw_top_right_counts(annotated_frame, class_counts)

            cv2.imwrite(str(annotated_dir / f"frame_{frame_idx:04d}.jpg"), annotated_frame)

            save_class_tinted_overlay(
                frame_cv,
                object_ids,
                masks,
                id_to_clean_class,
                class_colors,
                str(overlay_dir / f"frame_{frame_idx:04d}.jpg")
            )

            # =====================================================================
            # Stats Logging Execution
            # =====================================================================
            stats = gpu_stats(device)

            writer.writerow(
                [
                    frame_idx,
                    restarted,
                    stats.get("allocated_gb", 0),
                    stats.get("reserved_gb", 0),
                    stats.get("max_allocated_gb", 0),
                    stats.get("max_reserved_gb", 0),
                    cpu_memory_gb(),
                    elapsed_ms,
                    num_objects,
                ]
            )

            if frame_idx % args.chunk_size == 0:

                print(
                    f"Frame {frame_idx:5d} | "
                    f"Alloc {stats.get('allocated_gb',0):6.2f} GB | "
                    f"Reserved {stats.get('reserved_gb',0):6.2f} GB | "
                    f"CPU {cpu_memory_gb():6.2f} GB | "
                    f"Objects {num_objects}"
                )

                # Optional introspection of inference session
                try:
                    attrs = vars(session)

                    interesting = [
                        k
                        for k in attrs.keys()
                        if any(
                            x in k.lower()
                            for x in (
                                "cache",
                                "memory",
                                "feature",
                                "embedding",
                                "prompt",
                            )
                        )
                    ]

                    if interesting:
                        print("Session state:")
                        for k in sorted(interesting):
                            value = attrs[k]

                            try:
                                size = len(value)
                            except Exception:
                                size = "-"

                            print(f"    {k:35s} len={size}")

                except Exception:
                    pass

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print()
    print("Done.")
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()