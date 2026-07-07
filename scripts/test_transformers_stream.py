import os
import json
import random  # For random chunk simulation
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video


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


def load_point_tracks(path):
    if path is None:
        return None

    track_path = Path(path)
    if not track_path.exists():
        return None

    with open(track_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    tracks = payload.get("tracks") if isinstance(payload, dict) else None
    if not isinstance(tracks, list):
        raise ValueError(f"Point prompt file must contain a top-level 'tracks' list: {track_path}")
    

    return tracks


def frame_points_for_tracks(tracks, frame_idx, frame_width, frame_height):
    if not tracks or frame_idx < 0 or frame_idx >= len(tracks):
        return []

    frame_points = tracks[frame_idx]
    if isinstance(frame_points, dict):
        frame_points = frame_points.get("points") or frame_points.get("tracks") or []

    if not isinstance(frame_points, list):
        return []

    cleaned_points = []
    for point in frame_points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        try:
            x_coord = float(point[0])
            y_coord = float(point[1])
        except (TypeError, ValueError):
            continue

        if 0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0:
            x_coord *= float(frame_width)
            y_coord *= float(frame_height)

        cleaned_points.append([x_coord, y_coord])

    return cleaned_points


def mask_to_bool_array(mask, fill_holes=True, remove_sprinkles=True, sprinkle_threshold=150):
    """
    Natively recreates SAM3's post-processing cleanup on the CPU using OpenCV.
    Fills internal mask holes and filters out noisy background components ('sprinkles').
    """
    mask_array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)

    while mask_array.ndim > 2 and 1 in mask_array.shape:
        mask_array = np.squeeze(mask_array)

    if mask_array.ndim == 3:
        mask_array = mask_array[0]

    if mask_array.ndim != 2:
        return None

    # Convert to standard binary image matrix (0 or 255)
    binary_mask = (mask_array > 0).astype(np.uint8) * 255
    if not np.any(binary_mask):
        return binary_mask > 0

    # 1. Custom Hole Filling: Find all structural contours and draw them solid
    if fill_holes:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(binary_mask, contours, i, 255, -1)

    # 2. Custom Sprinkle Removal: Compute component statistics and delete tiny island noise
    if remove_sprinkles:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        cleaned_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):  # Index 0 is background
            if stats[i, cv2.CC_STAT_AREA] >= sprinkle_threshold:
                cleaned_mask[labels == i] = 255
        binary_mask = cleaned_mask

    return binary_mask > 200


def mask_to_bbox(mask):
    # Runs the custom hole-filled and sprinkle-removed mask array
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


# 1. Device Configuration (Reverted to bfloat16 for peak VRAM efficiency)
if torch.cuda.is_available():
    device_index = 1 if torch.cuda.device_count() > 1 else 0
    device = f"cuda:{device_index}"
    model_dtype = torch.bfloat16
else:
    device = "cpu"
    model_dtype = torch.float32

# 2. Load Video Frames
video_frames, _ = load_video("./data/bread_occlusion_fps5.mp4")

# Extract properties from seed frame upfront to allow 0-frame session registrations
seed_frame_idx = 0
seed_frame_cv = frame_to_bgr(video_frames[seed_frame_idx])
frame_height, frame_width = seed_frame_cv.shape[:2]

# 3. CONFIGURATION TOGGLE
load_point_prompts = False  # Set to True for Point Tracking, False for Text Tracking
output_subname = "transformers_streamed" if load_point_prompts else "transformers_streamed_bread_occlusion"
output_dir = f"output/{output_subname}"

text_concepts = {
    "bread": "bread",
    # "cheese_text": "square yellow cheese slice with smooth flat surface",
}
long_prompt_to_short_name = {long_desc: short for short, long_desc in text_concepts.items()}

point_concepts = {
    "bread_point_1": "./data/fiveguys/bread_tracks.json",
    "bread_point_2": "./data/fiveguys/bread_tracks.json",
    "cheese_point": "./data/fiveguys/cheese_tracks.json"
}

point_id_to_class = {}

# =====================================================================
# BRANCH A: POINT TRACKING CONFIGURATION (TRUE EMPTY SESSION)
# =====================================================================
if load_point_prompts:
    model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=model_dtype).to(device)
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    all_short_names = list(point_concepts.keys())

    # Natively initialize with ZERO pre-loaded frames 
    inference_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",       
        video_storage_device="cpu",     
        dtype=torch.bfloat16            
    )

    obj_ids_for_frame = []
    input_points_for_frame = []
    input_labels_for_frame = []

    for idx, (class_name, tracks_path) in enumerate(point_concepts.items()):
        obj_id = 100 + idx  
        point_id_to_class[obj_id] = class_name
        print(f"Loading point tracks for {class_name} from {tracks_path}")
        tracks = load_point_tracks(tracks_path)
        if tracks:
            positive_points = frame_points_for_tracks(tracks, seed_frame_idx, frame_width, frame_height)
            if class_name == "bread_point_1":
                positive_points = positive_points[:2]  
            elif class_name == "bread_point_2":
                positive_points = positive_points[2:]  
            print("\t positive points: ", positive_points)
            
            if positive_points:
                obj_ids_for_frame.append(obj_id)
                input_points_for_frame.append(positive_points)
                input_labels_for_frame.append([1] * len(positive_points))
    
    negative_points_for_frame = [[] for _ in obj_ids_for_frame]
    negative_labels_for_frame = [[] for _ in obj_ids_for_frame]
    for i, obj_id in enumerate(obj_ids_for_frame):
        for j, other_obj_id in enumerate(obj_ids_for_frame):
            if i != j:
                negative_points_for_frame[i].extend(input_points_for_frame[j])
                negative_labels_for_frame[i].extend([0] * len(input_points_for_frame[j]))

    for i in range(len(obj_ids_for_frame)):
        input_points_for_frame[i].extend(negative_points_for_frame[i])
        input_labels_for_frame[i].extend(negative_labels_for_frame[i])
        
    for i, obj_id in enumerate(obj_ids_for_frame):
        debug_frame = seed_frame_cv.copy()
        class_name = point_id_to_class.get(obj_id, f"object_{obj_id}")
        points = input_points_for_frame[i]
        labels = input_labels_for_frame[i]
        
        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(debug_frame, (x, y), 5, color, -1)
        
        cv2.imwrite(f"{output_dir}/debug_input_frame_{class_name}.jpg", debug_frame)

    if obj_ids_for_frame:
        # Crucial configuration shift: must pass original_size when initializing an open stream
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=seed_frame_idx,
            obj_ids=obj_ids_for_frame,
            input_points=[input_points_for_frame],  
            input_labels=[input_labels_for_frame],
            original_size=(frame_height, frame_width)  
        )

# =====================================================================
# BRANCH B: TEXT TRACKING CONFIGURATION (TRUE EMPTY SESSION)
# =====================================================================
else:
    model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=model_dtype).to(device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    all_short_names = list(text_concepts.keys())

    # Initialize empty streaming session here too
    inference_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",       
        video_storage_device="cpu",     
        dtype=torch.bfloat16            
    )

    for detailed_description in text_concepts.values():
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=detailed_description,
        )


# Setup directory structures
crop_base_dir = f"{output_dir}/saved_bboxes"
annotated_dir = f"{output_dir}/annotated_frames"
overlay_dir = f"{output_dir}/segmentation_overlays"
os.makedirs(crop_base_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

for short_name in all_short_names:
    os.makedirs(os.path.join(crop_base_dir, short_name), exist_ok=True)

class_colors = {
    "bread_text": (0, 215, 255),
    "cheese_text": (0, 255, 128),
    "bread_point": (255, 0, 255),
    "cheese_point": (255, 255, 0),
}

# =====================================================================
# 5. TRUE ASYNCHRONOUS MICRO-BATCH STREAMING LOOP
# =====================================================================
outputs_per_frame = {}
total_frames = len(video_frames)
current_idx = 0

while current_idx < total_frames:
    # Pick a random size of frames arriving together from our "network stream"
    chunk_size = 15
    end_idx = min(current_idx + chunk_size, total_frames)
    print(f"\n>>> [OpenFilter Simulator] Chunk Arrived! Processing {end_idx - current_idx} frames step-by-step...")
    
    # Process sequential frames one-by-one exactly like an OpenFilter pipe
    for frame_idx in range(current_idx, end_idx):
        frame = video_frames[frame_idx]
        frame_cv = frame_to_bgr(frame)
        frame_height, frame_width = frame_cv.shape[:2]

        # Standard preprocessing on the standalone arriving image frame
        inputs = processor(images=frame, return_tensors="pt").to(device)
        if model_dtype == torch.bfloat16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # True streaming forward pass: call model() directly with the single pixel tensor
        # This appends features to the internal state memory structure on the fly
        model_outputs = model(
            inference_session=inference_session,
            frame=inputs.pixel_values[0]
        )

        if load_point_prompts:
            processed_masks = processor.post_process_masks(
                [model_outputs.pred_masks],
                original_sizes=[[frame_height, frame_width]],  # Swapped to localized frame sizes
                binarize=True,
            )[0]
            
            if hasattr(model_outputs, "object_score_logits") and model_outputs.object_score_logits is not None:
                scores = torch.sigmoid(model_outputs.object_score_logits).reshape(-1)
            else:
                scores = torch.ones(len(model_outputs.object_ids))
                
            processed_outputs = {
                "object_ids": list(model_outputs.object_ids or []),
                "masks": processed_masks,
                "scores": scores,
            }
        else:
            processed_outputs = processor.postprocess_outputs(
                inference_session, 
                model_outputs,
                original_sizes=inputs.original_sizes  # Automatically scales results back during streaming
            )
            
        outputs_per_frame[frame_idx] = processed_outputs
        annotated_frame = frame_cv.copy()

        masks = processed_outputs.get("masks", [])
        scores = processed_outputs.get("scores", [])
        object_ids = processed_outputs.get("object_ids", [])
        class_counts = {class_name: 0 for class_name in all_short_names}
        
        prompt_to_obj_ids = processed_outputs.get("prompt_to_obj_ids", {})
        id_to_clean_class = {}

        for long_prompt, detected_ids in prompt_to_obj_ids.items():
            short_name = long_prompt_to_short_name.get(long_prompt, "unknown_text")
            for oid in detected_ids:
                id_to_clean_class[int(oid)] = short_name

        for oid, short_name in point_id_to_class.items():
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
                crop_path = os.path.join(crop_base_dir, short_name, crop_filename)
                cv2.imwrite(crop_path, crop)
            
            label = f"{short_name} {obj_id_int} ({score:.2f})"
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(0, ymin - text_h - 10)
            cv2.rectangle(annotated_frame, (xmin, label_top), (xmin + text_w, ymin), (0, 255, 0), -1)
            
            text_y = ymin - 5 if ymin - 5 > text_h else min(frame_height - 5, ymax + text_h + 5)
            cv2.putText(annotated_frame, label, (xmin, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        
        draw_top_right_counts(annotated_frame, class_counts)

        cv2.imwrite(os.path.join(annotated_dir, f"frame_{frame_idx:04d}.jpg"), annotated_frame)

        save_class_tinted_overlay(
            frame_cv,
            object_ids,
            masks,
            id_to_clean_class,
            class_colors,
            os.path.join(overlay_dir, f"frame_{frame_idx:04d}.jpg")
        )
        
        print(f"Processed Frame {frame_idx} (Mode: {output_subname})")

    # Slide our micro-batch processing window pointer forward
    current_idx = end_idx