import copy
import hashlib
import logging
import os
import json
import time
from typing import Optional, ClassVar
from pathlib import Path
import pydantic

import torch
import numpy as np
from filter_sam3_detector.utils.bbox import to_xyxy
from PIL import Image


from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame
from openfilter.filter_runtime.shapes import DetectionSet
from openfilter.filter_runtime.config import FilterConfigBase

from pydantic import Field
from typing import List, Any, Union
from .coco_export import convert_jsonl_to_coco
from .temporal_intervals import DetectionInterval, IntervalTracker
from .streaming_video_processor import StreamingVideoProcessor


class FilterSAM3DetectorConfigSchema(FilterConfigBase):
    """Declarative config schema for FilterSAM3Detector."""

    model_config = {"extra": "forbid"}

    # Base SAM3 configuration
    model_id: str = Field(default="facebook/sam3", description="Model ID")
    device: str = Field(default="cuda", description="Compute device")

    # Prompt configuration
    prompt_mode: str = Field(
        default="Default/Visual",
        json_schema_extra={"x-openfilter-ui": {"widget": "select"}},
    )
    text_prompt: Optional[str] = Field(
        default=None, description="Text prompt for detection"
    )
    text_prompts: Optional[Union[str, List[str]]] = Field(
        default=None, description="Delimiter-separated prompts or list of prompts"
    )
    prompt_delimiter: str = Field(
        default="###", description="Delimiter for multiple prompts"
    )
    class_delimiter: str = Field(
        default="|||", description="Delimiter separating class label from prompt"
    )
    prompt_sets: Optional[List[Any]] = Field(
        default=None, description="Multi-output mode configuration"
    )

    # Exemplars configuration
    exemplars_path: Optional[str] = Field(
        default=None,
        description="Path to directory containing cropped example images",
        json_schema_extra={"x-openfilter-ui": {"group": "Exemplars"}},
    )
    exemplar_embeddings_cache: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Exemplars", "advanced": True}},
    )

    # Thresholds
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score to keep a detection",
        json_schema_extra={"x-openfilter-ui": {"widget": "slider"}},
    )
    mask_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for binarizing predicted masks",
        json_schema_extra={"x-openfilter-ui": {"widget": "slider"}},
    )
    max_detections: int = Field(
        default=100, description="Maximum number of detections to keep per frame"
    )

    # Output options
    output_masks: bool = Field(
        default=False,
        description="Include full segmentation masks in output (can increase payload size)",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs"}},
    )
    output_boxes: bool = Field(
        default=True,
        description="Include bounding boxes in output",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs"}},
    )
    output_scores: bool = Field(
        default=True,
        description="Include confidence scores in output",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs"}},
    )
    output_label: str = Field(
        default="sam3_detections",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs"}},
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path to save JSONL annotations",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs"}},
    )
    auto_export_coco: bool = Field(
        default=False,
        description="Export COCO JSON on shutdown when output_path is set",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs", "advanced": True}},
    )
    coco_output_path: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs", "advanced": True}},
    )
    output_filter_name: str = Field(
        default="SAM3Detector",
        json_schema_extra={"x-openfilter-ui": {"group": "Outputs", "advanced": True}},
    )

    # NMS (Non-Maximum Suppression)
    nms_enabled: bool = Field(
        default=True,
        description="Enable NMS to suppress overlapping detections",
        json_schema_extra={"x-openfilter-ui": {"group": "NMS Options"}},
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS (higher = more boxes kept)",
        json_schema_extra={
            "x-openfilter-ui": {"group": "NMS Options", "widget": "slider"}
        },
    )

    # Visualization
    frames_output_dir: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Visualization", "advanced": True}
        },
    )
    annotated_frames_output_dir: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Visualization", "advanced": True}
        },
    )
    save_annotated_frames: bool = Field(
        default=False, json_schema_extra={"x-openfilter-ui": {"group": "Visualization"}}
    )
    debug: bool = Field(
        default=False,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Visualization", "advanced": True}
        },
    )
    visualize: bool = Field(
        default=False, json_schema_extra={"x-openfilter-ui": {"group": "Visualization"}}
    )
    viz_topic: str = Field(
        default="",
        json_schema_extra={
            "x-openfilter-ui": {"group": "Visualization", "advanced": True}
        },
    )

    # Temporal Intervals
    enable_temporal_intervals: bool = Field(
        default=False,
        description="Enable inline temporal interval tracking",
        json_schema_extra={"x-openfilter-ui": {"group": "Temporal Tracking"}},
    )
    temporal_half_life: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Temporal Tracking"}},
    )
    temporal_full_decay_life: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Temporal Tracking"}},
    )
    temporal_presence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Temporal Tracking", "widget": "slider"}
        },
    )
    temporal_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Temporal Tracking", "widget": "slider"}
        },
    )
    temporal_output_json_path: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Temporal Tracking"}},
    )
    temporal_streaming_mode: bool = Field(
        default=False,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Temporal Tracking", "advanced": True}
        },
    )
    temporal_emit_on_change: bool = Field(
        default=True,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Temporal Tracking", "advanced": True}
        },
    )
    temporal_label_field: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Temporal Tracking", "advanced": True}
        },
    )

    # Video Mode
    enable_video_mode: bool = Field(
        default=False,
        description="Use video mode with temporal tracking",
        json_schema_extra={"x-openfilter-ui": {"group": "Video Tracking"}},
    )
    video_detection_interval: int = Field(
        default=5, json_schema_extra={"x-openfilter-ui": {"group": "Video Tracking"}}
    )
    video_min_tracking_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Video Tracking", "widget": "slider"}
        },
    )

    # Reference Boxes and Layout
    positive_boxes: Optional[List[Any]] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    negative_boxes: Optional[List[Any]] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_images: Optional[List[str]] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_images_negative: Optional[List[str]] = Field(
        default=None,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_margin: int = Field(
        default=10,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_gap: int = Field(
        default=5,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_layout: str = Field(
        default="overlay",
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_strip_width: int = Field(
        default=120,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    ref_max_height: int = Field(
        default=80,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    composite_topic: str = Field(
        default="",
        json_schema_extra={
            "x-openfilter-ui": {"group": "Reference Configuration", "advanced": True}
        },
    )
    confusion_detection_enabled: Optional[bool] = Field(
        default=None,
        json_schema_extra={"x-openfilter-ui": {"group": "Confusion Detection"}},
    )
    confusion_iou_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Confusion Detection", "widget": "slider"}
        },
    )
    remove_overlap: bool = Field(
        default=False,
        json_schema_extra={"x-openfilter-ui": {"group": "Confusion Detection"}},
    )
    mixed_precision: bool = Field(
        default=True,
        json_schema_extra={
            "x-openfilter-ui": {"group": "Base Configuration", "advanced": True}
        },
    )
    # Exclude inherited dynamic fields that we don't want explicitly validated
    sources: Optional[List[str]] = Field(default=None)
    outputs: Optional[List[str]] = Field(default=None)


class FilterSAM3DetectorOutput(DetectionSet):
    """Official frame.data payload schema for FilterSAM3Detector."""

    __schema_id__: ClassVar[str] = (
        "https://schemas.plainsight.ai/filters/sam3-detector/v1"
    )
    __frame_data_key__: ClassVar[str] = "detections"


__all__ = ["FilterSAM3DetectorConfig", "FilterSAM3Detector", "FilterSAM3DetectorOutput"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Image file extensions for exemplars and path expansion (first-level directory listing)
REF_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Local normalize_bbox (cxcywh) to avoid importing sam3.visualization_utils, which pulls in matplotlib.
def _normalize_bbox_cxcywh(bbox_cxcywh, img_w, img_h):
    """Normalize bbox [cx, cy, w, h] by image size. Same contract as sam3.visualization_utils.normalize_bbox."""
    if isinstance(bbox_cxcywh, torch.Tensor):
        out = bbox_cxcywh.clone()
        out[..., 0] /= img_w
        out[..., 1] /= img_h
        out[..., 2] /= img_w
        out[..., 3] /= img_h
        return out
    out = list(bbox_cxcywh)
    out[0] /= img_w
    out[1] /= img_h
    out[2] /= img_w
    out[3] /= img_h
    return out


# Try to import SAM3 from facebookresearch/sam3 (no matplotlib: we use local _normalize_bbox_cxcywh)
HAS_SAM3 = False
box_xywh_to_cxcywh = None
normalize_bbox = None
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xywh_to_cxcywh

    normalize_bbox = _normalize_bbox_cxcywh
    HAS_SAM3 = True
except Exception as e:
    HAS_SAM3 = False
    box_xywh_to_cxcywh = None
    normalize_bbox = None
    logger.warning(
        "SAM3 not available: %s. Install from: https://github.com/facebookresearch/sam3",
        e,
    )


class FilterSAM3DetectorConfig(FilterSAM3DetectorConfigSchema):
    """Configuration for SAM3 object detection filter."""

    model_config = {"extra": "allow"}

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            super().__init__(**{**args[0], **kwargs})
        else:
            super().__init__(*args, **kwargs)

    def clean(self) -> dict:
        """Return a dictionary of this config without any hidden items starting with '_'."""
        return {k: v for k, v in self.items() if not k.startswith("_")}

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        if key not in self.keys():
            return False
        return (
            key in self.model_fields_set
            or (self.__pydantic_extra__ is not None and key in self.__pydantic_extra__)
            or getattr(self, key) is not None
        )

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.keys():
            return getattr(self, key)
        return default

    def setdefault(self, key: str, default: Any = None) -> Any:
        if hasattr(self.__class__, key):
            return getattr(self, key)
        if key not in self.keys() or getattr(self, key) is None:
            setattr(self, key, default)
        return getattr(self, key)

    def keys(self):
        base = list(self.__class__.model_fields.keys())
        extra = list(self.__pydantic_extra__ or {})
        return base + [k for k in extra if k not in set(base)]

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]


class FilterSAM3Detector(Filter):
    """
    SAM3 object detection filter.

    This filter performs open-set object detection using SAM3 (Segment Anything Model 3).
    It supports two prompting modes:
    - Text prompts: Natural language descriptions (e.g., "person", "car")
    - Image exemplars: Few-shot learning with cropped example images

    Typical workflow:
    1. Configure with text prompt and/or exemplar images directory
    2. For each input frame, run SAM3 inference
    3. Output detections with boxes and scores to frame metadata

    Exemplar images:
    - Provide a directory path containing cropped JPG/PNG images
    - Each image should show exactly what you want to detect
    - Images are encoded and averaged to create visual embeddings
    - Example: exemplars_path="/path/to/containers/" with container1.jpg, container2.jpg, etc.
    """

    # Defensive marker for future SDK changes that may consume class attributes for schema discovery
    __output_schema__ = FilterSAM3DetectorOutput

    @classmethod
    def normalize_config(cls, config: FilterConfig) -> FilterConfig:
        """
        Normalize and validate configuration parameters.

        This method MUST BE IDEMPOTENT - calling it multiple times should produce the same result.
        """
        # First, call parent normalize_config to handle sources, outputs, etc.
        config = super().normalize_config(config)
        config = FilterSAM3DetectorConfig(config)

        # NOTE: Existing comma-separated configs may need updating
        # NOTE: List inputs are treated as raw prompts (no class mapping)

        # Set defaults if not present
        defaults = {
            "model_id": "facebook/sam3",
            "device": "cuda",
            "text_prompt": None,  # Single prompt (backward compatible)
            "text_prompts": None,  # Delimiter-separated prompts, e.g. "vehicle|||car,truck###animal|||cat, dogs"
            "prompt_delimiter": "###",  # Multiple prompts separated by '###'
            "class_delimiter": "|||",  # Separates class label from prompt, e.g. "vehicle|||car"
            "prompt_sets": None,  # Multi-output mode: list of {name, prompts, topic, ...}
            "exemplars_path": None,
            "exemplar_embeddings_cache": None,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "max_detections": 100,
            "output_masks": False,  # Don't save masks by default (can be large)
            "output_boxes": True,
            "output_scores": True,
            "output_label": "sam3_detections",
            "output_path": None,  # Path to save JSONL annotations
            "auto_export_coco": False,  # Opt-in: export COCO JSON on shutdown when output_path is set
            "coco_output_path": None,  # Optional explicit path for COCO JSON output
            "output_filter_name": "SAM3Detector",  # Filter name for event sink format
            # NMS (Non-Maximum Suppression) options
            "nms_enabled": True,  # Enable NMS to suppress overlapping detections
            "nms_threshold": 0.5,  # IoU threshold for NMS (higher = more boxes kept)
            "frames_output_dir": None,  # Directory to save original frames
            "annotated_frames_output_dir": None,  # Directory to save annotated frames (separate from original)
            "save_annotated_frames": False,  # Save frames with visual annotations (boxes, scores, masks)
            "debug": False,
            "visualize": False,
            "viz_topic": "",  # When set (e.g. "viz"), main gets original frame + meta; this topic gets drawn frame + meta
            # Temporal interval tracking options (integrated from TemporalIntervalFilter)
            "enable_temporal_intervals": False,  # Enable inline temporal interval tracking
            "temporal_half_life": None,  # Frames for 50% EMA decay (fast signal)
            "temporal_full_decay_life": None,  # Frames for ~99.3% EMA decay (slow crossing)
            "temporal_presence_threshold": 0.5,  # EMA threshold for presence detection
            "temporal_min_confidence": 0.0,  # Min detection confidence to consider
            "temporal_output_json_path": None,  # Path to write intervals JSON
            "temporal_streaming_mode": False,  # Emit intervals incrementally to JSON
            "temporal_emit_on_change": True,  # Add interval state to frame metadata on changes
            "temporal_label_field": None,  # Detection field for class label (None = use text_prompt)
            # Video mode options (experimental - uses memory-based tracking)
            "enable_video_mode": False,  # Use video mode with temporal tracking
            "video_detection_interval": 5,  # Frames between full detection runs in video mode
            "video_min_tracking_confidence": 0.3,  # Re-detect if tracking confidence drops
            # Reference boxes on the original image (SAM3-style): list of [x, y, w, h] in pixels per box
            "positive_boxes": None,
            "negative_boxes": None,
            # Ref images (pasted on composite): list of paths; when set, REF_IMGS are used only if no ref boxes
            "ref_images": None,
            "ref_images_negative": None,
            "ref_margin": 10,
            "ref_gap": 5,
            "composite_topic": "",  # When set (e.g. "composite"), publish composite image when REF_IMGS in use
            "ref_layout": "overlay",  # "overlay" = refs on frame; "side_strips" = refs in lateral strips
            "ref_strip_width": 120,  # Width of each lateral strip when ref_layout == "side_strips"
            "ref_max_height": 80,  # Max height (px) of each ref image when pasted on composite
            # Cross-prompt overlap detection (confusion detection)
            "confusion_detection_enabled": None,  # None = auto (True when >1 prompt)
            "confusion_iou_threshold": 0.95,  # IoU gate for cross-class overlap; 95% targets near-identical boxes
            "remove_overlap": False,  # FILTER_REMOVE_OVERLAP; when true, shutdown pass removes lower-confidence duplicates
            # Mixed precision inference
            "mixed_precision": True,  # Use bfloat16 autocast for inference (CUDA only)
        }

        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        # Load from environment variables (override config values)
        env_mapping = {
            "model_id": str,
            "device": str,
            "text_prompt": str,
            "text_prompts": str,  # delimiter separated list of prompts
            "class_delimiter": str,
            "prompt_delimiter": str,
            "prompt_sets": str,  # JSON array of prompt set configs
            "exemplars_path": str,
            "exemplar_embeddings_cache": str,
            "confidence_threshold": float,
            "mask_threshold": float,
            "max_detections": int,
            "output_masks": bool,
            "output_boxes": bool,
            "output_scores": bool,
            "output_label": str,
            "output_path": str,
            "auto_export_coco": bool,
            "coco_output_path": str,
            "nms_enabled": bool,
            "nms_threshold": float,
            "frames_output_dir": str,
            "annotated_frames_output_dir": str,
            "save_annotated_frames": bool,
            "visualize": bool,
            "viz_topic": str,
            # Temporal interval env mappings
            "enable_temporal_intervals": bool,
            "temporal_half_life": float,
            "temporal_full_decay_life": float,
            "temporal_presence_threshold": float,
            "temporal_min_confidence": float,
            "temporal_output_json_path": str,
            "temporal_streaming_mode": bool,
            "temporal_emit_on_change": bool,
            "temporal_label_field": str,
            # Grounding cache (skip grounding for K-1 out of every K frames)
            "grounding_cache_frames": int,
            # Video mode env mappings
            "enable_video_mode": bool,
            "video_detection_interval": int,
            "video_min_tracking_confidence": float,
            # Cross-prompt overlap / confusion detection
            "confusion_detection_enabled": bool,
            "confusion_iou_threshold": float,
            "remove_overlap": bool,
            "mixed_precision": bool,
        }

        def _parse_boxes_env(env_val: str):
            if not env_val or not env_val.strip():
                return None
            try:
                raw = json.loads(env_val.strip())
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"FILTER_POSITIVE_BOXES/FILTER_NEGATIVE_BOXES must be valid JSON array: {e}"
                )
            if not isinstance(raw, list):
                raise ValueError(
                    f"FILTER_POSITIVE_BOXES/FILTER_NEGATIVE_BOXES must be a JSON array, got {type(raw)}"
                )
            out = []
            for i, item in enumerate(raw):
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    raise ValueError(
                        f"Box {i} must be [x, y, w, h] with 4 numbers, got {item}"
                    )
                out.append(
                    [float(item[0]), float(item[1]), float(item[2]), float(item[3])]
                )
            return out if out else None

        env_positive_boxes = os.getenv("FILTER_POSITIVE_BOXES")
        if env_positive_boxes is not None:
            config["positive_boxes"] = _parse_boxes_env(env_positive_boxes)

        env_negative_boxes = os.getenv("FILTER_NEGATIVE_BOXES")
        if env_negative_boxes is not None:
            config["negative_boxes"] = _parse_boxes_env(env_negative_boxes)

        # FILTER_REF_IMAGES / FILTER_REF_IMAGES_NEGATIVE: comma-separated paths (list or None)
        def _parse_ref_images_env(env_val: str):
            if not env_val or not env_val.strip():
                return None
            parts = [p.strip() for p in env_val.strip().split(",") if p.strip()]
            return parts if parts else None

        env_ref_images = os.getenv("FILTER_REF_IMAGES")
        if env_ref_images is not None:
            config["ref_images"] = _parse_ref_images_env(env_ref_images)
        env_ref_images_negative = os.getenv("FILTER_REF_IMAGES_NEGATIVE")
        if env_ref_images_negative is not None:
            config["ref_images_negative"] = _parse_ref_images_env(
                env_ref_images_negative
            )

        env_composite_topic = os.getenv("FILTER_COMPOSITE_TOPIC")
        if env_composite_topic is not None:
            config["composite_topic"] = env_composite_topic.strip()

        env_ref_layout = os.getenv("FILTER_REF_IMAGES_LAYOUT")
        if env_ref_layout is not None:
            v = env_ref_layout.strip().lower()
            config["ref_layout"] = v if v in ("overlay", "side_strips") else "overlay"
        env_ref_strip_width = os.getenv("FILTER_REF_STRIP_WIDTH")
        if env_ref_strip_width is not None:
            try:
                config["ref_strip_width"] = int(env_ref_strip_width.strip())
            except ValueError:
                pass
        env_ref_max_height = os.getenv("FILTER_REF_MAX_HEIGHT")
        if env_ref_max_height is not None:
            try:
                config["ref_max_height"] = int(env_ref_max_height.strip())
            except ValueError:
                pass

        # Special handling for FILTER_OUTPUT_PATH (maps to output_path)
        env_output_path = os.getenv("FILTER_OUTPUT_PATH")
        if env_output_path is not None:
            config["output_path"] = env_output_path.strip()

        # Special handling for FILTER_OUTPUT_FILTER_NAME (maps to output_filter_name)
        env_output_filter_name = os.getenv("FILTER_OUTPUT_FILTER_NAME")
        if env_output_filter_name is not None:
            config["output_filter_name"] = env_output_filter_name.strip()

        # Special handling for FILTER_FRAMES_OUTPUT_DIR (maps to frames_output_dir)
        env_frames_output_dir = os.getenv("FILTER_FRAMES_OUTPUT_DIR")
        if env_frames_output_dir is not None:
            config["frames_output_dir"] = env_frames_output_dir.strip()

        # Special handling for FILTER_ANNOTATED_FRAMES_OUTPUT_DIR (maps to annotated_frames_output_dir)
        env_annotated_frames_output_dir = os.getenv(
            "FILTER_ANNOTATED_FRAMES_OUTPUT_DIR"
        )
        if env_annotated_frames_output_dir is not None:
            config["annotated_frames_output_dir"] = (
                env_annotated_frames_output_dir.strip()
            )

        for key, expected_type in env_mapping.items():
            env_key = f"FILTER_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                if expected_type is bool:
                    config[key] = env_val.strip().lower() in ("true", "1", "yes")
                elif expected_type is float:
                    config[key] = float(env_val.strip())
                elif expected_type is int:
                    config[key] = int(env_val.strip())
                else:
                    config[key] = env_val.strip()

        # Validate device
        valid_devices = ["cuda", "cpu", "mps"]
        device = config.get("device", "cuda")
        if isinstance(device, str):
            device = device.lower().strip()
            if device not in valid_devices:
                raise ValueError(
                    f"Invalid device: {device}. Must be one of {valid_devices}"
                )
            config["device"] = device

        # Validate numeric ranges
        confidence_threshold = config.get("confidence_threshold", 0.5)
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
            )

        mask_threshold = config.get("mask_threshold", 0.5)
        if not (0.0 <= mask_threshold <= 1.0):
            raise ValueError(
                f"mask_threshold must be between 0 and 1, got {mask_threshold}"
            )

        max_detections = config.get("max_detections", 100)
        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")

        nms_threshold = config.get("nms_threshold", 0.5)
        if not (0.0 <= nms_threshold <= 1.0):
            raise ValueError(
                f"nms_threshold must be between 0 and 1, got {nms_threshold}"
            )

        confusion_iou_threshold = config.get("confusion_iou_threshold", 0.95)
        if not (0.0 <= confusion_iou_threshold <= 1.0):
            raise ValueError(
                f"confusion_iou_threshold must be between 0 and 1, got {confusion_iou_threshold}"
            )

        class_delimiter = config.get("class_delimiter")
        prompt_delimiter = config.get("prompt_delimiter")
        if not class_delimiter or not prompt_delimiter:
            raise ValueError("Delimiters must be non-empty")

        if class_delimiter == prompt_delimiter:
            raise ValueError("class_delimiter and prompt_delimiter must differ")

        # Parse text_prompts from a string using a configurable delimiter into a list
        text_prompts = config.get("text_prompts")
        config.setdefault("prompt_label_map", {})
        if isinstance(text_prompts, str):
            # Map detected prompt text to output label/class
            config["prompt_label_map"] = {}
            items = [
                item.strip()
                for item in text_prompts.split(prompt_delimiter)
                if item.strip()
            ]
            for item in items:
                if class_delimiter in item:
                    k, v = item.split(class_delimiter, 1)
                    prompt = v.strip()
                    label = k.strip()

                    if prompt in config["prompt_label_map"]:
                        raise ValueError(
                            f"Duplicate prompt mapping for '{prompt}': "
                            f"'{config['prompt_label_map'][prompt]}' vs '{label}'"
                        )
                    config["prompt_label_map"][prompt] = label
                else:
                    config["prompt_label_map"][item] = item
            config["text_prompts"] = [
                item.split(class_delimiter, 1)[1].strip()
                if class_delimiter in item
                else item
                for item in items
            ]
        elif text_prompts is None:
            config["text_prompts"] = None
        elif not isinstance(text_prompts, list):
            raise ValueError(
                f"text_prompts must be a list or delimiter separated string, got {type(text_prompts)}"
            )

        # Parse prompt_sets from JSON string to list of dicts
        # Format: [{"name": "bowl", "prompts": ["bowl"], "topic": "main", "confidence_threshold": 0.5, "max_detections": 1}, ...]
        prompt_sets = config.get("prompt_sets")
        if isinstance(prompt_sets, str):
            try:
                config["prompt_sets"] = json.loads(prompt_sets)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"prompt_sets must be valid JSON array, got error: {e}"
                )
        elif prompt_sets is not None and not isinstance(prompt_sets, list):
            raise ValueError(
                f"prompt_sets must be a list or JSON string, got {type(prompt_sets)}"
            )

        # Validate prompt_sets structure
        if config.get("prompt_sets"):
            for i, ps in enumerate(config["prompt_sets"]):
                if not isinstance(ps, dict):
                    raise ValueError(f"prompt_sets[{i}] must be a dict, got {type(ps)}")
                if "name" not in ps:
                    raise ValueError(f"prompt_sets[{i}] missing required 'name' field")
                if "prompts" not in ps:
                    raise ValueError(
                        f"prompt_sets[{i}] missing required 'prompts' field"
                    )
                # Ensure prompts is a list
                if isinstance(ps["prompts"], str):
                    ps["prompts"] = [
                        p.strip() for p in ps["prompts"].split(",") if p.strip()
                    ]

        return config

    def setup(self, config: FilterConfig):
        """
        Initialize the filter with the given configuration.

        This method is called once when the filter starts. It should:
        - Load the SAM3 model
        - Load and process exemplar images
        - Compute exemplar embeddings
        - Initialize any required state
        """
        logger.info("==========================================")
        logger.info(f"FilterSAM3Detector setup: {config}")
        logger.info("==========================================")

        self.cfg = config

        # Initialize jsonl_file to None (will be set if output_path is provided)
        self.jsonl_file = None

        # Store configuration (access as dict since FilterConfig is dict-like)
        self.model_id = config.get("model_id", "facebook/sam3")
        self.text_prompt = config.get(
            "text_prompt"
        )  # Single prompt (backward compatible)
        self.text_prompts = config.get(
            "text_prompts"
        )  # Multiple prompts for parallel detection
        self.prompt_sets = config.get("prompt_sets")  # Multi-output mode
        self.exemplars_path = config.get("exemplars_path")
        # Reference boxes on original image: list of [x, y, w, h] in pixels (SAM3-style)
        self.positive_boxes = config.get("positive_boxes") or []
        self.negative_boxes = config.get("negative_boxes") or []
        # Cache for normalized ref boxes (lazy-filled per resolution to avoid per-frame recompute)
        self._cached_norm_boxes_size = None
        self._cached_norm_positive_boxes = None
        self._cached_norm_negative_boxes = None
        # Ref images (pasted on composite): only used when no ref boxes (rule: boxes take priority)
        ref_images_raw = config.get("ref_images")
        ref_images_negative_raw = config.get("ref_images_negative")
        has_ref_boxes = bool(self.positive_boxes or self.negative_boxes)
        if has_ref_boxes and (ref_images_raw or ref_images_negative_raw):
            logger.info(
                "FILTER_POSITIVE_BOXES/FILTER_NEGATIVE_BOXES set; ignoring REF_IMGS (ref images disabled)."
            )
            ref_images_raw = None
            ref_images_negative_raw = None
        self.ref_images_paths = (
            self._expand_ref_paths(ref_images_raw) if ref_images_raw else None
        )
        self.ref_images_negative_paths = (
            self._expand_ref_paths(ref_images_negative_raw)
            if ref_images_negative_raw
            else None
        )
        self.ref_margin = config.get("ref_margin", 10)
        self.ref_gap = config.get("ref_gap", 5)
        self.composite_topic = (config.get("composite_topic") or "").strip()
        self.ref_layout = (config.get("ref_layout") or "overlay").strip().lower()
        if self.ref_layout not in ("overlay", "side_strips"):
            self.ref_layout = "overlay"
        self.ref_strip_width = max(1, int(config.get("ref_strip_width", 120)))
        self.ref_max_height = max(1, min(4000, int(config.get("ref_max_height", 80))))
        if self.ref_images_paths or self.ref_images_negative_paths:
            n_pos = len(self.ref_images_paths) if self.ref_images_paths else 0
            n_neg = (
                len(self.ref_images_negative_paths)
                if self.ref_images_negative_paths
                else 0
            )
            logger.info(
                f"Using reference images: {n_pos} positive, {n_neg} negative (REF_IMGS; disabled when FILTER_POSITIVE_BOXES/FILTER_NEGATIVE_BOXES are set)"
            )
        # Load and resize ref images once so we don't read from disk every frame
        self._cached_ref_images_pos = (
            self._load_ref_images_for_cache(self.ref_images_paths)
            if self.ref_images_paths
            else None
        )
        self._cached_ref_images_neg = (
            self._load_ref_images_for_cache(self.ref_images_negative_paths)
            if self.ref_images_negative_paths
            else None
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.max_detections = config.get("max_detections", 100)
        self.output_masks = config.get("output_masks", True)
        self.output_boxes = config.get("output_boxes", True)
        self.output_scores = config.get("output_scores", True)
        if not self.output_boxes or not self.output_scores:
            raise ValueError(
                "output_boxes and output_scores must both be True. "
                "The canonical FilterSAM3DetectorOutput schema requires bbox and score fields."
            )
        self.output_label = config.get("output_label", "sam3_detections")
        self.output_path = config.get("output_path", None)
        self.auto_export_coco = config.get("auto_export_coco", False)
        self.coco_output_path = config.get("coco_output_path", None)
        self.output_filter_name = config.get("output_filter_name", "SAM3Detector")
        self.nms_enabled = config.get("nms_enabled", True)
        self.nms_threshold = config.get("nms_threshold", 0.5)
        self.frames_output_dir = config.get("frames_output_dir", None)
        self.annotated_frames_output_dir = config.get(
            "annotated_frames_output_dir", None
        )
        self.save_annotated_frames = config.get("save_annotated_frames", False)
        self.visualize = config.get("visualize", False)
        self.viz_topic = (config.get("viz_topic") or "").strip()

        # Initialize JSONL output file if path is provided
        self.jsonl_file = None
        if self.output_path:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = open(output_path, "w")
            logger.info(f"Saving annotations to: {output_path}")

        # Initialize frames output directories if provided
        self.frames_dir = None
        self.annotated_frames_dir = None
        self.frame_counter = 0  # Counter for unique frame numbering
        self.global_detection_id = (
            0  # Global counter for unique detection IDs across all frames
        )

        # Always save original frames if frames_output_dir is configured (default: true)
        if self.frames_output_dir:
            self.frames_dir = Path(self.frames_output_dir)
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving original frames to: {self.frames_dir}")

        # Save annotated frames only if save_annotated_frames is enabled (default: false)
        if self.save_annotated_frames:
            if self.annotated_frames_output_dir:
                # Use explicitly configured directory
                self.annotated_frames_dir = Path(self.annotated_frames_output_dir)
            elif self.frames_output_dir:
                # Default: use frames_output_dir parent + "frames_annotated"
                # e.g., ./output/frames -> ./output/frames_annotated
                frames_parent = Path(self.frames_output_dir).parent
                self.annotated_frames_dir = frames_parent / "frames_annotated"
            else:
                # Fallback: use ./output/frames_annotated
                self.annotated_frames_dir = Path("./output/frames_annotated")

            self.annotated_frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving annotated frames to: {self.annotated_frames_dir}")

        # Determine device
        device_str = config.get("device", "cuda")
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif (
            device_str == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Mixed precision: bfloat16 autocast on CUDA (matches SAM3 video path)
        self.mixed_precision = (
            config.get("mixed_precision", True) and self.device.type == "cuda"
        )
        self._autocast_persistent = None  # initialized in _load_model()
        if self.mixed_precision:
            logger.info("Mixed precision enabled: bfloat16 autocast for inference")

        logger.info(f"Using device: {self.device}")
        logger.info(f"NMS enabled: {self.nms_enabled}, threshold: {self.nms_threshold}")

        # Video mode configuration
        self.enable_video_mode = config.get("enable_video_mode", False)
        self.video_detection_interval = config.get("video_detection_interval", 5)
        self.video_min_tracking_confidence = config.get(
            "video_min_tracking_confidence", 0.3
        )
        self.video_processor = None

        # Load SAM3 model
        self.model = None
        self.processor = None

        if self.enable_video_mode:
            self._load_video_model()
        else:
            self._load_model()

        # Load exemplar images if provided
        self.visual_prompt_embed = None
        self.visual_prompt_mask = None
        if self.exemplars_path:
            self._load_exemplar_images()

        n_pos = len(self.positive_boxes) if self.positive_boxes else 0
        n_neg = len(self.negative_boxes) if self.negative_boxes else 0
        if n_pos or n_neg:
            logger.info(
                f"Using reference boxes: {n_pos} positive, {n_neg} negative (FILTER_POSITIVE_BOXES / FILTER_NEGATIVE_BOXES)"
            )
            # In ref-boxes mode only the first text prompt is used per frame
            if self.text_prompts and len(self.text_prompts) > 1:
                logger.warning(
                    "Reference-boxes mode uses only the first text prompt; other prompt(s) are ignored (use text-prompt-only mode for multi-prompt detection)"
                )

        # Log multi-output mode configuration and pre-cache text embeddings
        # This is the KEY OPTIMIZATION: text prompts are static, so we encode them ONCE
        # at startup instead of on every frame. This saves significant inference time.
        self.cached_text_embeddings = {}  # prompt -> {language_features, language_mask, language_embeds}

        if self.prompt_sets:
            logger.info(
                f"Multi-output mode enabled with {len(self.prompt_sets)} prompt sets:"
            )
            for ps in self.prompt_sets:
                logger.info(
                    f"  - {ps['name']}: prompts={ps['prompts']}, topic={ps.get('topic', 'main')}"
                )

        # Pre-cache text embeddings whenever we have text_prompt, text_prompts, or prompt_sets
        # (required for single text_prompt mode to use cached embeddings; multi-output already relied on this)
        if self.model is not None and (
            self.text_prompt or self.text_prompts or self.prompt_sets
        ):
            self._cache_text_embeddings()

        # Initialize temporal interval tracking if enabled
        self.enable_temporal_intervals = config.get("enable_temporal_intervals", False)
        self.interval_tracker: Optional[IntervalTracker] = None

        if self.enable_temporal_intervals:
            self._setup_temporal_intervals(config)

        self._setup_confusion_detector(config)

        # Batched backbone state: set by process_batch() so process() can skip set_image()
        self._cached_backbone_state = None

        logger.info("FilterSAM3Detector setup complete")

    def _setup_temporal_intervals(self, config: FilterConfig):
        """Initialize temporal interval tracking using reusable IntervalTracker."""
        # Store config for detection aggregation
        self.temporal_min_confidence = config.get("temporal_min_confidence", 0.0)
        self.temporal_emit_on_change = config.get("temporal_emit_on_change", True)
        self.temporal_label_field = config.get("temporal_label_field")
        self.temporal_default_label = self.text_prompt or "foreground"

        # Create IntervalTracker with all tracking logic
        self.interval_tracker = IntervalTracker(
            half_life=config.get("temporal_half_life"),
            full_decay_life=config.get("temporal_full_decay_life"),
            presence_threshold=config.get("temporal_presence_threshold", 0.5),
            output_json_path=config.get("temporal_output_json_path"),
            streaming_mode=config.get("temporal_streaming_mode", False),
        )

        logger.info(
            f"Temporal intervals enabled: half_life={config.get('temporal_half_life')}, "
            f"full_decay_life={config.get('temporal_full_decay_life')}, "
            f"threshold={config.get('temporal_presence_threshold', 0.5)}, "
            f"streaming={config.get('temporal_streaming_mode', False)}"
        )

    def _setup_confusion_detector(self, config: FilterConfig):
        """Initialize cross-prompt confusion detector.

        Auto-enabled when more than one text prompt is active (i.e. when
        ``FILTER_TEXT_PROMPTS`` or ``prompt_sets`` produces multiple classes).
        Single-prompt runs see zero overhead.
        """
        from .confusion_detector import ConfusionDetector

        has_multi_text = bool(self.text_prompts and len(self.text_prompts) > 1)
        has_multi_sets = bool(self.prompt_sets and len(self.prompt_sets) > 1)
        auto_enabled = has_multi_text or has_multi_sets

        cfg_enabled = config.get("confusion_detection_enabled")  # None = auto
        if cfg_enabled is None:
            self.confusion_detection_enabled = auto_enabled
        else:
            self.confusion_detection_enabled = bool(cfg_enabled)

        self.confusion_iou_threshold = config.get("confusion_iou_threshold", 0.95)
        self.remove_overlap = config.get("remove_overlap", False)
        self.confusion_detector: Optional[ConfusionDetector] = None

        if self.confusion_detection_enabled:
            self.confusion_detector = ConfusionDetector(
                iou_threshold=self.confusion_iou_threshold
            )
            n_prompts = len(self.text_prompts or []) + sum(
                len(ps.get("prompts", [])) for ps in (self.prompt_sets or [])
            )
            logger.info(
                f"Confusion detection enabled (iou_threshold={self.confusion_iou_threshold}, "
                f"remove_overlap={self.remove_overlap}, prompts={n_prompts})."
            )
        else:
            logger.debug(
                "Confusion detection disabled (single prompt or explicitly disabled)."
            )

    def shutdown(self):
        """
        Clean up resources when the filter is stopped.

        This method should release any held resources like:
        - GPU memory
        - File handles
        - Cached data
        """
        logger.info("FilterSAM3Detector shutdown")

        # Finalize temporal intervals if enabled
        if self.interval_tracker is not None:
            self.interval_tracker.finalize()

        # Close JSONL file if open; overlap finalize + COCO run after successful close only
        if hasattr(self, "jsonl_file") and self.jsonl_file is not None:
            try:
                self.jsonl_file.close()
            except Exception as e:
                logger.warning(f"Error closing annotation file: {e}")
            else:
                if hasattr(self, "output_path") and self.output_path:
                    logger.info(f"Closed annotation file: {self.output_path}")
                    try:
                        coco_jsonl = self._finalize_cross_prompt_overlaps()
                        if self.auto_export_coco:
                            self._run_coco_export(jsonl_source=coco_jsonl)
                    except Exception as e:
                        logger.warning(
                            "Post-close annotation processing failed (overlap finalize or COCO export): %s",
                            e,
                            exc_info=True,
                        )
            self.jsonl_file = None

        # Exit persistent autocast context if active
        if (
            hasattr(self, "_autocast_persistent")
            and self._autocast_persistent is not None
        ):
            self._autocast_persistent.__exit__(None, None, None)
            self._autocast_persistent = None

        # Release model resources
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("FilterSAM3Detector shutdown complete")

    def _run_coco_export(self, jsonl_source: Optional[Path] = None) -> None:
        if not self.output_path:
            return
        primary = Path(self.output_path)
        if jsonl_source is not None and jsonl_source.exists():
            input_path = jsonl_source
            coco_from = "cleaned JSONL"
        else:
            input_path = primary
            coco_from = "primary JSONL"
        if not input_path.exists():
            logger.warning(f"COCO export skipped; input JSONL not found: {input_path}")
            return

        # Default COCO output as sibling of primary detections JSONL, independent of CWD.
        if self.coco_output_path:
            output_path = Path(self.coco_output_path)
        else:
            output_path = primary.parent / "labels_coco.json"

        try:
            coco = convert_jsonl_to_coco(input_path, output_path, self.output_label)
            logger.info(
                "COCO export completed: %s (images=%d, annotations=%d, categories=%d) from %s (%s)",
                output_path,
                len(coco.get("images", [])),
                len(coco.get("annotations", [])),
                len(coco.get("categories", [])),
                input_path,
                coco_from,
            )
        except Exception as e:
            logger.warning(f"COCO export failed: {e}", exc_info=True)

    def _finalize_cross_prompt_overlaps(self) -> Optional[Path]:
        """Shutdown pass: count and optionally remove cross-class overlapping detections.

        Runs once at shutdown after the JSONL is flushed.  Uses a **streaming** read
        (no full-file list of records).  When ``remove_overlap`` is False (default), only
        counts are logged.  When True, writes ``*_cleaned.jsonl`` with overlapping
        lower-confidence boxes removed per frame (second streaming pass over the file).

        Returns:
            Path to the cleaned JSONL when written this run; ``None`` otherwise.
            ``_run_coco_export`` uses this so ``labels_coco.json`` matches the cleaned
            stream when ``FILTER_REMOVE_OVERLAP=true``.

        A *cross-class overlap* is a pair of detections with **different** class/label
        and IoU ≥ ``confusion_iou_threshold``.  Same-class pairs are unchanged.
        """
        if not self.confusion_detection_enabled:
            return None
        if not self.output_path:
            return None

        input_path = Path(self.output_path)
        if not input_path.exists():
            logger.warning("Confusion pass skipped; JSONL not found: %s", input_path)
            return None

        detector = self.confusion_detector
        if detector is None:
            return None

        remove = self.remove_overlap

        # ---- Streaming pass: counts, overlap pairs (before), class set — no full-file buffer ----
        before_count = 0
        total_detections_before = 0
        invalid_jsonl_lines = 0
        all_classes: set[str] = set()

        def _accumulate_frame_stats(record: dict) -> None:
            nonlocal before_count, total_detections_before
            dets = self._extract_detections_from_record(record)
            total_detections_before += len(dets)
            for d in dets:
                cls = d.get("class") or d.get("class_name") or d.get("label") or ""
                if cls:
                    all_classes.add(cls)
            if len(dets) >= 2:
                by_class: dict[str, list] = {}
                for d in dets:
                    cls = d.get("class") or d.get("class_name") or d.get("label") or ""
                    by_class.setdefault(cls, []).append(d)
                if len(by_class) > 1:
                    confusions = detector.detect(by_class)
                    before_count += len(confusions)

        try:
            with open(input_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_jsonl_lines += 1
                        continue

                    _accumulate_frame_stats(record)

        except Exception as e:
            logger.warning("Confusion pass failed during read: %s", e, exc_info=True)
            return None

        if invalid_jsonl_lines:
            logger.warning(
                "Confusion overlap pass: %d JSONL line(s) skipped (invalid JSON); "
                "detection totals below count only successfully parsed lines.",
                invalid_jsonl_lines,
            )

        skip_note = (
            f" invalid_jsonl_lines_skipped={invalid_jsonl_lines}"
            if invalid_jsonl_lines
            else ""
        )

        after_count = before_count  # default: unchanged

        # ---- Optional rewrite: stream input → cleaned JSONL line by line ----
        if remove and before_count > 0:
            if len(all_classes) <= 1:
                logger.info(
                    "Cross-prompt overlaps: overlap_pairs before=%d after=%d (single class — removal skipped); "
                    "detections total=%d (FILTER_REMOVE_OVERLAP=%s)%s",
                    before_count,
                    after_count,
                    total_detections_before,
                    remove,
                    skip_note,
                )
                return None

            cleaned_path = input_path.parent / (
                input_path.stem + "_cleaned" + input_path.suffix
            )
            after_count = 0
            total_detections_after = 0
            total_boxes_removed = 0
            try:
                with open(input_path, "r") as in_fh, open(cleaned_path, "w") as out_fh:
                    for line in in_fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            out_fh.write(line + "\n")
                            continue

                        dets = self._extract_detections_from_record(record)
                        kept, dropped = detector.remove_overlapping(dets)
                        total_detections_after += len(kept)
                        total_boxes_removed += len(dropped)

                        by_class: dict[str, list] = {}
                        for d in kept:
                            cls = (
                                d.get("class")
                                or d.get("class_name")
                                or d.get("label")
                                or ""
                            )
                            by_class.setdefault(cls, []).append(d)
                        if len(by_class) > 1:
                            after_count += len(detector.detect(by_class))

                        if dropped:
                            record = self._rewrite_record_detections(record, kept)

                        out_fh.write(json.dumps(record) + "\n")

                pairs_removed = before_count - after_count
                logger.info(
                    "Cross-prompt overlaps: overlap_pairs before=%d after=%d removed=%d | "
                    "detections before=%d after=%d removed=%d — cleaned JSONL: %s (FILTER_REMOVE_OVERLAP=%s)%s",
                    before_count,
                    after_count,
                    pairs_removed,
                    total_detections_before,
                    total_detections_after,
                    total_boxes_removed,
                    cleaned_path,
                    remove,
                    skip_note,
                )
                return cleaned_path
            except Exception as e:
                logger.warning("Confusion rewrite failed: %s", e, exc_info=True)
                return None
        else:
            logger.info(
                "Cross-prompt overlaps: overlap_pairs before=%d after=%d | detections total=%d "
                "(FILTER_REMOVE_OVERLAP=%s)%s",
                before_count,
                after_count,
                total_detections_before,
                remove,
                skip_note,
            )

            if before_count > 0:
                logger.warning(
                    "%d cross-class overlap(s) detected at shutdown. "
                    "Set FILTER_REMOVE_OVERLAP=true to keep only the highest-confidence class per pair, "
                    "or run: python scripts/analyze_confusions.py %s",
                    before_count,
                    input_path,
                )
        return None

    @staticmethod
    def _extract_detections_from_record(record: dict) -> list:
        """Extract the flat detections list from a JSONL event record."""
        from filter_sam3_detector.utils.detections import extract_items

        data = record.get("data", record)
        return extract_items(data)

    @staticmethod
    def _rewrite_record_detections(record: dict, kept: list) -> dict:
        """Return a deep copy of record with detections replaced by kept."""
        record = copy.deepcopy(record)
        data = record.get("data", record)
        if isinstance(data, dict):
            if "detections" in data:
                detections_payload = data["detections"]
                if (
                    isinstance(detections_payload, dict)
                    and "items" in detections_payload
                ):
                    detections_payload["items"] = kept
                    return record
                elif isinstance(detections_payload, list):
                    data["detections"] = kept
                    return record

            meta = data.setdefault("meta", {})
            for key in ("detections", "sam3_detections"):
                if key in meta:
                    meta[key] = kept
        return record

    def _process_temporal_intervals(self, frame: Frame, detections: list):
        """
        Process temporal interval tracking for the current frame.

        Uses IntervalTracker for EMA updates, state changes, and interval management.
        """
        # Get frame ID from metadata if available
        meta = frame.data.get("meta", {})
        frame_id = int(meta["id"]) if "id" in meta else None

        # Aggregate detections by label (max score per label)
        detected_labels = self._aggregate_temporal_detections(detections)

        # Update tracker and get state changes
        state_changes = self.interval_tracker.update(detected_labels, frame_id)

        # Add interval info to frame metadata if state changed
        if state_changes and self.temporal_emit_on_change:
            meta = frame.data.setdefault("meta", {})
            meta["temporal_intervals"] = {
                "frame_id": frame_id or self.interval_tracker.frame_count,
                "state_changes": [
                    {"label": label, "present": present, "ema": round(ema, 4)}
                    for label, present, ema in state_changes
                ],
            }

    def _aggregate_temporal_detections(self, detections: list) -> dict[str, float]:
        """
        Aggregate detections by label, returning max confidence per label.

        Returns:
            Dict mapping label -> max confidence score
        """
        label_scores: dict[str, float] = {}

        for det in detections:
            if not isinstance(det, dict):
                continue

            # Get confidence score
            score = det.get("score", 1.0)
            if score < self.temporal_min_confidence:
                continue

            # Get label from detection or use default
            if self.temporal_label_field and self.temporal_label_field in det:
                label = str(det[self.temporal_label_field])
            else:
                # Use class field if available, otherwise default label
                label = (
                    det.get("label")
                    or det.get("class")
                    or det.get("class_name")
                    or self.temporal_default_label
                )

            # Track max score per label
            if label not in label_scores or score > label_scores[label]:
                label_scores[label] = score

        return label_scores

    def get_temporal_intervals(self) -> list[DetectionInterval]:
        """Get all completed temporal intervals (for programmatic access)."""
        if self.interval_tracker is None:
            return []
        return self.interval_tracker.get_intervals()

    def get_temporal_state(self) -> dict[str, dict]:
        """Get current temporal presence state for all tracked labels."""
        if self.interval_tracker is None:
            return {}
        return self.interval_tracker.get_current_state()

    def _extract_filter_frame_id(self, frames: dict[str, Frame]) -> Optional[int]:
        """Extract frame ID from _filter hidden topic (TI-130).

        The _filter topic is emitted by openfilter runtime and contains:
        - id: Frame ID(s) from input frames' meta.id or auto-generated

        Returns frame ID if found, None otherwise.
        """
        # Check for _filter topic (format: SourceName___filter or _filter)
        for topic, frame in frames.items():
            # Match _filter hidden topic:
            # - Standalone: '_filter'
            # - With source prefix: 'SourceName___filter' (SourceName + __ + _filter)
            if topic == "_filter" or topic.endswith("___filter"):
                if frame and frame.data and isinstance(frame.data, dict):
                    frame_id = frame.data.get("id")
                    if frame_id is not None:
                        logger.debug(f"Extracted frame id from {topic}: {frame_id}")
                        return (
                            int(frame_id)
                            if isinstance(frame_id, (int, float))
                            else frame_id
                        )
                break  # Only use first _filter topic found

        return None

    @torch.no_grad()
    def process(self, frames: dict[str, Frame]) -> dict[str, Frame]:
        """
        Process input frames and detect objects.

        Args:
            frames: Dictionary of input frames keyed by topic name

        Returns:
            Dictionary of output frames with detection results
        """
        output_frames = {}

        # Extract frame ID from _filter topic (TI-130)
        # The _filter topic is emitted by openfilter runtime and contains frame IDs
        filter_frame_id = self._extract_filter_frame_id(frames)

        for topic, frame in frames.items():
            if frame is None:
                continue

            if not frame.has_image:
                # Forward non-image frames unchanged
                output_frames[topic] = frame
                continue

            # Check if model is loaded
            if self.model is None or self.processor is None:
                logger.warning("SAM3 model not loaded, forwarding frame unchanged")
                output_frames[topic] = frame
                continue

            # Multi-output mode: process each prompt_set and output to different topics
            if self.prompt_sets:
                try:
                    multi_output = self._process_multi_output(frame, filter_frame_id)
                    output_frames.update(multi_output)
                except Exception as e:
                    logger.error(f"Error in multi-output processing: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())
                    output_frames[topic] = frame
                continue

            # Standard single-output mode
            # Determine which prompts to use
            # Priority: text_prompts (list) > text_prompt (single) > visual embeddings
            prompts_to_use = None
            if self.text_prompts:
                prompts_to_use = self.text_prompts
            elif self.text_prompt:
                prompts_to_use = [self.text_prompt]

            # Need either text prompts, visual embeddings from exemplars, reference boxes, or reference images
            has_ref_boxes = bool(self.positive_boxes or self.negative_boxes)
            has_ref_images = bool(
                self.ref_images_paths or self.ref_images_negative_paths
            )
            if (
                prompts_to_use is None
                and self.visual_prompt_embed is None
                and not has_ref_boxes
                and not has_ref_images
            ):
                logger.warning(
                    "No text prompt(s), exemplars, reference boxes, or reference images configured, forwarding frame unchanged"
                )
                output_frames[topic] = frame
                continue

            try:
                # Extract image from frame (convert BGR to RGB PIL)
                image_bgr = frame.rw_bgr.image
                image_rgb = image_bgr[:, :, ::-1]  # BGR to RGB
                pil_image = Image.fromarray(image_rgb)

                # Get image dimensions for clipping boxes
                img_height, img_width = image_bgr.shape[:2]

                # Collect all detections across all prompts
                detections = (
                    None  # None = didn't run (pass-through); [] = ran, found nothing
                )
                all_scores = []  # Track all scores for detection_confidence calculation

                if has_ref_boxes:
                    if not HAS_SAM3:
                        logger.warning(
                            "Reference boxes configured but SAM3 is not available (install sam3); forwarding frame unchanged"
                        )
                    else:
                        # Reference-boxes mode: use original image and add geometric prompts (no composite)
                        # Only the first text prompt is used; additional prompts are ignored (see setup warning if text_prompts has multiple)
                        state = self.processor.set_image(pil_image)
                        self.processor.reset_all_prompts(state)
                        prompt = prompts_to_use[0] if prompts_to_use else "visual"
                        if prompt in self.cached_text_embeddings:
                            state = self._inject_cached_text_embedding(state, prompt)
                        else:
                            state = self.processor.set_text_prompt_no_grounding(
                                prompt, state
                            )
                        cache_key = (img_width, img_height)
                        if self._cached_norm_boxes_size != cache_key:
                            self._cached_norm_positive_boxes = (
                                self._boxes_xywh_to_norm_cxcywh(
                                    self.positive_boxes, img_width, img_height
                                )
                            )
                            self._cached_norm_negative_boxes = (
                                self._boxes_xywh_to_norm_cxcywh(
                                    self.negative_boxes, img_width, img_height
                                )
                            )
                            self._cached_norm_boxes_size = cache_key
                        norm_positive = self._cached_norm_positive_boxes
                        norm_negative = self._cached_norm_negative_boxes
                        for norm_box in norm_positive:
                            state = self.processor.add_geometric_prompt(
                                norm_box, True, state
                            )
                        for norm_box in norm_negative:
                            state = self.processor.add_geometric_prompt(
                                norm_box, False, state
                            )
                        # add_geometric_prompt runs _forward_grounding(state) internally, so state has boxes/scores
                        detections = self._extract_detections_from_state(
                            state,
                            prompt,
                            img_width,
                            img_height,
                            self.global_detection_id,
                        )
                        # Remove detections that overlap any negative reference box. When a detection overlaps both
                        # positive and negative, keep it only if its center is inside a positive box (positive ref prediction).
                        if self.negative_boxes:
                            neg_regions = [
                                [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                                for b in self.negative_boxes
                                if len(b) == 4
                            ]
                            pos_regions = [
                                [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                                for b in (self.positive_boxes or [])
                                if len(b) == 4
                            ]
                            if neg_regions:

                                def _keep(d):
                                    if "box" not in d:
                                        return True
                                    if not self._box_overlaps_any_region(
                                        d["box"], neg_regions
                                    ):
                                        return True
                                    if (
                                        pos_regions
                                        and self._box_center_inside_any_region(
                                            d["box"], pos_regions
                                        )
                                    ):
                                        return True  # center in positive: keep (positive ref prediction)
                                    return False

                                detections = [d for d in detections if _keep(d)]
                        num_extracted = len(detections)
                        all_scores.extend(
                            float(d["score"]) for d in detections if "score" in d
                        )
                        self.global_detection_id += num_extracted
                elif has_ref_images and HAS_SAM3:
                    # REF_IMGS mode: composite with refs pasted + geometric prompts (only when no ref boxes).
                    # Text prompt is optional; when omitted, use "visual" so geometric prompts drive detection.
                    build_result = self._build_composite_with_refs(pil_image)
                    composite_pil = build_result[0]
                    all_norm_labels = build_result[1]
                    frame_offset_x = build_result[2] if len(build_result) >= 3 else None
                    ref_regions_frame = (
                        build_result[3] if len(build_result) > 3 else None
                    )
                    state = self.processor.set_image(composite_pil)
                    self.processor.reset_all_prompts(state)
                    prompt = prompts_to_use[0] if prompts_to_use else "visual"
                    if prompt in self.cached_text_embeddings:
                        state = self._inject_cached_text_embedding(state, prompt)
                    else:
                        state = self.processor.set_text_prompt_no_grounding(
                            prompt, state
                        )
                    for norm_box, label in all_norm_labels:
                        state = self.processor.add_geometric_prompt(
                            norm_box, label, state
                        )
                    # add_geometric_prompt runs _forward_grounding(state) internally, so state has boxes/scores
                    if frame_offset_x is not None:
                        composite_w, composite_h = composite_pil.size
                        detections = self._extract_detections_from_state(
                            state,
                            prompt,
                            composite_w,
                            composite_h,
                            self.global_detection_id,
                        )
                        frame_x1, frame_y1 = frame_offset_x, 0
                        frame_x2, frame_y2 = frame_offset_x + img_width, img_height
                        detections_in_frame = []
                        for d in detections:
                            if "box" not in d:
                                continue
                            x1, y1, x2, y2 = d["box"]
                            if (
                                x2 <= frame_x1
                                or x1 >= frame_x2
                                or y2 <= frame_y1
                                or y1 >= frame_y2
                            ):
                                continue
                            x1_f = max(0, int(x1 - frame_offset_x))
                            y1_f = max(0, int(y1))
                            x2_f = min(img_width, int(x2 - frame_offset_x))
                            y2_f = min(img_height, int(y2))
                            if x2_f <= x1_f or y2_f <= y1_f:
                                continue
                            d_f = dict(d)
                            d_f["box"] = [x1_f, y1_f, x2_f, y2_f]
                            if "rois" in d_f and d_f["rois"]:
                                d_f["rois"] = [[x1_f, y1_f, x2_f, y2_f]]
                            detections_in_frame.append(d_f)
                        detections = detections_in_frame
                    else:
                        detections = self._extract_detections_from_state(
                            state,
                            prompt,
                            img_width,
                            img_height,
                            self.global_detection_id,
                        )
                        if ref_regions_frame:
                            detections = [
                                d
                                for d in detections
                                if "box" not in d
                                or not self._box_overlaps_any_region(
                                    d["box"], ref_regions_frame
                                )
                            ]
                    num_extracted = len(detections)
                    all_scores.extend(
                        float(d["score"]) for d in detections if "score" in d
                    )
                    self.global_detection_id += num_extracted
                    # Publish composite to composite_topic when set (with detections in blue)
                    if self.composite_topic:
                        composite_rgb = np.array(composite_pil)
                        composite_bgr = composite_rgb[:, :, ::-1].copy()
                        if frame_offset_x is not None:
                            detections_for_viz = []
                            for d in detections:
                                if "box" in d:
                                    x1, y1, x2, y2 = d["box"]
                                    detections_for_viz.append(
                                        {
                                            **d,
                                            "box": [
                                                x1 + frame_offset_x,
                                                y1,
                                                x2 + frame_offset_x,
                                                y2,
                                            ],
                                        }
                                    )
                                else:
                                    detections_for_viz.append(d)
                            composite_bgr = self._visualize_detections_on_image(
                                composite_bgr, detections_for_viz, None, None
                            )
                        else:
                            composite_bgr = self._visualize_detections_on_image(
                                composite_bgr, detections, None, None
                            )
                        frame_composite = Frame(composite_bgr, frame.data, "BGR")
                        output_frames[self.composite_topic] = frame_composite
                elif has_ref_images and not HAS_SAM3:
                    logger.warning(
                        "Reference images configured but SAM3 is not available; forwarding frame unchanged"
                    )
                else:
                    # Standard mode: set image once, then loop over prompts (and optionally visual exemplars)
                    detections = []
                    if self._cached_backbone_state is not None:
                        state = self._cached_backbone_state
                    else:
                        state = self.processor.set_image(pil_image)

                    # Process each prompt using cached image features
                    if prompts_to_use:
                        for prompt in prompts_to_use:
                            # Use cached text embeddings if available, otherwise encode on-the-fly
                            # This reuses the cached image features from set_image()
                            if prompt in self.cached_text_embeddings:
                                prompt_state = self._inject_cached_text_embedding(
                                    state, prompt
                                )
                            else:
                                prompt_state = (
                                    self.processor.set_text_prompt_no_grounding(
                                        prompt, state
                                    )
                                )
                            prompt_state = self.processor.forward_grounding(
                                prompt_state
                            )

                            # Extract detections for this prompt (use global ID counter for uniqueness)
                            prompt_detections = self._extract_detections_from_state(
                                prompt_state,
                                prompt,
                                img_width,
                                img_height,
                                self.global_detection_id,
                            )
                            detections.extend(prompt_detections)
                            self.global_detection_id += len(prompt_detections)

                            # Track scores (only for kept detections; state["scores"] includes sub-threshold)
                            all_scores.extend(
                                float(d["score"])
                                for d in prompt_detections
                                if "score" in d
                            )

                    # If we have visual embeddings from exemplar images, run grounding with them
                    if self.visual_prompt_embed is not None:
                        # Ensure we have language features (use "visual" as placeholder if no text prompt)
                        if "language_features" not in state["backbone_out"]:
                            dummy_text_outputs = self.model.backbone.forward_text(
                                ["visual"], device=str(self.device)
                            )
                            # Move all dummy text outputs to match the backbone output device
                            # (the model may have weights on a different device than self.device)
                            target_device = state["backbone_out"][
                                "vision_features"
                            ].device
                            for key, value in dummy_text_outputs.items():
                                if hasattr(value, "to"):
                                    dummy_text_outputs[key] = value.to(target_device)
                            state["backbone_out"].update(dummy_text_outputs)

                        # Initialize geometric prompt if not present
                        if "geometric_prompt" not in state:
                            state["geometric_prompt"] = self.model._get_dummy_prompt()

                        # Run grounding with visual prompt embeddings
                        visual_state = self._forward_grounding_with_visual_prompt(state)

                        # Extract detections for visual prompt (use global ID counter)
                        visual_detections = self._extract_detections_from_state(
                            visual_state,
                            "visual",
                            img_width,
                            img_height,
                            self.global_detection_id,
                        )
                        detections.extend(visual_detections)
                        self.global_detection_id += len(visual_detections)

                        # Track scores (only for kept detections; state["scores"] includes sub-threshold)
                        all_scores.extend(
                            float(d["score"]) for d in visual_detections if "score" in d
                        )

                # If no inference ran (SAM3 unavailable), forward frame unchanged
                if detections is None:
                    output_frames[topic] = frame
                    continue

                # Set scores variable for detection_confidence calculation (1:1 with detections)
                scores = all_scores if all_scores else None

                # Calculate detection_confidence (average or max score)
                detection_confidence = None
                if detections and scores is not None and len(scores) > 0:
                    # Use the maximum confidence score (scores are already aligned with detections)
                    max_score = max(float(s) for s in scores)
                    detection_confidence = float(max_score)
                elif detections and any("score" in d for d in detections):
                    # Fallback: use max score from detections
                    max_score = max(
                        d.get("score", 0.0) for d in detections if "score" in d
                    )
                    detection_confidence = float(max_score)
                # append class name to detections
                for d in detections:
                    prompt = d.get("class", "object")
                    label = self.config.get("prompt_label_map", {}).get(prompt, prompt)
                    d["label"] = label

                # Store results in frame data under the canonical key
                canonical_dict, protege_list, classification_dict = (
                    self._normalize_detections(detections)
                )
                frame.data[FilterSAM3DetectorOutput.__frame_data_key__] = canonical_dict
                serialized_detections = canonical_dict

                # Store width and height in meta for backward-compatibility and logging
                frame_meta = frame.data.setdefault("meta", {})
                frame_meta["width"] = img_width
                frame_meta["height"] = img_height

                # Restore legacy dual-writes for unmigrated consumers
                frame_meta["detections"] = protege_list
                frame_meta[self.output_label] = detections
                frame_meta["classification"] = classification_dict

                # Add detection_confidence to meta
                if detection_confidence is not None:
                    frame_meta["detection_confidence"] = detection_confidence

                # Process temporal intervals if enabled
                if self.enable_temporal_intervals:
                    self._process_temporal_intervals(frame, detections)

                # Get frame metadata for JSONL and filename
                frame_meta = frame.data.get("meta", {})

                # Get timestamp from frame (OpenFilter provides this)
                # Try multiple sources: frame.timestamp, meta.ts, meta.timestamp, data.timestamp
                frame_ts = (
                    getattr(frame, "timestamp", None)
                    or frame_meta.get("ts", None)
                    or frame_meta.get("timestamp", None)
                    or frame.data.get("timestamp", None)
                )

                # Get frame ID - priority: _filter topic > meta['id'] > frame_counter
                # The _filter topic (TI-130) is the idiomatic way to get frame IDs in openfilter
                frame_id_num = (
                    filter_frame_id
                    if filter_frame_id is not None
                    else frame_meta.get("id", None)
                )

                # Use frame counter for unique numbering (increments for each frame processed)
                frame_counter = self.frame_counter
                self.frame_counter += 1

                # Generate unique filename using timestamp and frame counter
                # Format similar to filter-frame-selector: frame_{id}_ts{timestamp}_count{counter}
                if frame_ts is not None:
                    # Format timestamp: replace dot with underscore to avoid dots in filename
                    timestamp_str = f"{float(frame_ts):.3f}".replace(".", "_")
                else:
                    # Fallback: use current time
                    current_time = time.time()
                    timestamp_str = f"{current_time:.3f}".replace(".", "_")

                # Handle frame_id_num - convert to int if possible, otherwise use counter
                if isinstance(frame_id_num, (int, float)):
                    frame_id_str = f"{int(frame_id_num):06d}"
                else:
                    # Use counter if no frame_id available
                    frame_id_str = f"{frame_counter:06d}"

                # Create filename: frame_{id}_ts{timestamp}_count{counter}.jpg
                filename_base = (
                    f"frame_{frame_id_str}_ts{timestamp_str}_count{frame_counter:06d}"
                )
                frame_filename_str = f"{filename_base}.jpg"

                # Generate full paths for original and annotated frames
                frame_filename = None
                annotated_frame_filename = None

                if self.frames_dir is not None:
                    frame_filename = self.frames_dir / frame_filename_str

                if self.annotated_frames_dir is not None:
                    annotated_frame_filename = (
                        self.annotated_frames_dir / frame_filename_str
                    )

                # Save frames if output directories are configured
                try:
                    import cv2

                    # Get original image from frame
                    image_bgr_original = frame.rw_bgr.image.copy()

                    # Save original frame (always save if frames_output_dir is configured)
                    if self.frames_dir is not None and frame_filename:
                        cv2.imwrite(str(frame_filename), image_bgr_original)

                    # Save annotated frame (if annotated_frames_dir is configured and there are detections or ref boxes)
                    if (
                        self.annotated_frames_dir is not None
                        and annotated_frame_filename
                        and (detections or has_ref_boxes)
                    ):
                        image_bgr_annotated = image_bgr_original.copy()
                        image_bgr_annotated = self._visualize_detections_on_image(
                            image_bgr_annotated,
                            detections,
                            self.positive_boxes if has_ref_boxes else None,
                            self.negative_boxes if has_ref_boxes else None,
                        )
                        cv2.imwrite(str(annotated_frame_filename), image_bgr_annotated)

                except Exception as e:
                    logger.warning(f"Failed to save frame: {e}")

                # Save to JSONL file if output_path is configured (save ALL frames, even without detections)
                if hasattr(self, "jsonl_file") and self.jsonl_file is not None:
                    try:
                        output_frame_id = (
                            frame_id_num if frame_id_num is not None else frame_counter
                        )
                        jsonl_meta = {
                            "frame_id": output_frame_id,
                            "width": img_width,
                            "height": img_height,
                            "filename": frame_filename_str,
                        }

                        # Event sink format matching frame.data structure
                        event_record = {
                            "filter_name": self.output_filter_name,
                            "topic": "main",
                            "data": {
                                "id": output_frame_id,
                                "detections": serialized_detections,
                                "meta": jsonl_meta,
                            },
                        }
                        self.jsonl_file.write(json.dumps(event_record) + "\n")
                        self.jsonl_file.flush()  # Ensure immediate write
                    except Exception as e:
                        logger.warning(f"Failed to save annotation to JSONL: {e}")

                # Optional visualization (for output frame): ref boxes (green/red) and detections (blue)
                if self.viz_topic:
                    output_frames[topic] = frame  # main = original + meta
                    if self.visualize and (detections or has_ref_boxes):
                        frame_viz = self._visualize_detections(
                            frame,
                            detections,
                            self.positive_boxes if has_ref_boxes else None,
                            self.negative_boxes if has_ref_boxes else None,
                        )
                        output_frames[self.viz_topic] = frame_viz
                else:
                    if self.visualize and (detections or has_ref_boxes):
                        frame = self._visualize_detections(
                            frame,
                            detections,
                            self.positive_boxes if has_ref_boxes else None,
                            self.negative_boxes if has_ref_boxes else None,
                        )
                    output_frames[topic] = frame

            except Exception as e:
                logger.error(f"Error processing frame from {topic}: {e}")
                import traceback

                logger.debug(traceback.format_exc())

                # Mirror multi-output error handler: ensure canonical/legacy fields are present
                try:
                    img_height, img_width = frame.rw_bgr.image.shape[:2]
                except Exception:
                    img_height, img_width = 0, 0

                canonical_dict, protege_list, classification_dict = (
                    self._normalize_detections([])
                )
                frame.data[FilterSAM3DetectorOutput.__frame_data_key__] = canonical_dict

                frame_meta = frame.data.setdefault("meta", {})
                if img_width > 0 and img_height > 0:
                    frame_meta["width"] = img_width
                    frame_meta["height"] = img_height
                frame_meta["detections"] = protege_list
                frame_meta[self.output_label] = []
                frame_meta["classification"] = classification_dict

                output_frames[topic] = frame
                continue

        return output_frames

    def _process_multi_output(
        self, frame: Frame, filter_frame_id: Optional[int]
    ) -> dict[str, Frame]:
        """
        Process frame with multiple prompt sets, outputting to different topics.

        Multi-output mode allows a single SAM3 instance to serve multiple "virtual detectors",
        each with its own prompts, thresholds, and output topic. The model runs inference
        once per frame (expensive backbone pass), then processes each prompt set using the
        cached image features.

        Args:
            frame: Input frame with image
            filter_frame_id: Frame ID from _filter topic (or None)

        Returns:
            Dictionary mapping output topics to frames with filtered detections
        """

        output_frames = {}

        # Extract image from frame (convert BGR to RGB PIL)
        image_bgr = frame.rw_bgr.image
        image_rgb = image_bgr[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(image_rgb)

        # Get image dimensions for clipping boxes
        img_height, img_width = image_bgr.shape[:2]

        # Set image in processor ONCE (this is the expensive backbone pass)
        # The state contains cached image features that we reuse for ALL prompt sets
        state = self.processor.set_image(pil_image)

        # Get frame metadata for ID tracking and filename (same format as single-output)
        frame_meta_orig = frame.data.get("meta", {})
        frame_id_num = (
            filter_frame_id
            if filter_frame_id is not None
            else frame_meta_orig.get("id", None)
        )
        frame_counter = self.frame_counter
        self.frame_counter += 1

        frame_ts = (
            getattr(frame, "timestamp", None)
            or frame_meta_orig.get("ts", None)
            or frame_meta_orig.get("timestamp", None)
            or frame.data.get("timestamp", None)
        )
        if frame_ts is not None:
            timestamp_str = f"{float(frame_ts):.3f}".replace(".", "_")
        else:
            timestamp_str = f"{time.time():.3f}".replace(".", "_")
        if isinstance(frame_id_num, (int, float)):
            frame_id_str = f"{int(frame_id_num):06d}"
        else:
            frame_id_str = f"{frame_counter:06d}"
        frame_filename_str = (
            f"frame_{frame_id_str}_ts{timestamp_str}_count{frame_counter:06d}.jpg"
        )

        # Save original (unannotated) frame once, before prompt set loop
        if self.frames_dir is not None:
            try:
                import cv2

                cv2.imwrite(str(self.frames_dir / frame_filename_str), image_bgr.copy())
            except Exception as e:
                logger.warning(f"Failed to save original frame: {e}")

        has_ref_boxes = bool(self.positive_boxes or self.negative_boxes)

        # Process each prompt set
        for prompt_set in self.prompt_sets:
            serialized_detections = {"items": []}
            try:
                ps_name = prompt_set["name"]
                ps_prompts = prompt_set["prompts"]
                ps_topic = prompt_set.get("topic", "main")
                ps_threshold = prompt_set.get(
                    "confidence_threshold", self.confidence_threshold
                )
                ps_max_detections = prompt_set.get(
                    "max_detections", self.max_detections
                )
                ps_filter_name = prompt_set.get("filter_name", f"SAM3_{ps_name}")
                # Optional label aliases: maps prompt -> output label (e.g., "printed order ticket" -> "chit")
                ps_label_aliases = prompt_set.get("label_aliases", {})

                # Collect detections for this prompt set
                ps_detections = []

                for prompt in ps_prompts:
                    # Use cached text embeddings (pre-computed at startup) instead of encoding per-frame
                    # This is the KEY OPTIMIZATION that avoids re-running the text encoder on every frame
                    prompt_state = self._inject_cached_text_embedding(state, prompt)
                    prompt_state = self.processor.forward_grounding(prompt_state)

                    # Determine output label: use alias if defined, otherwise use prompt
                    output_label = ps_label_aliases.get(prompt, prompt)

                    # Extract detections for this prompt
                    prompt_detections = self._extract_detections_from_state(
                        prompt_state,
                        output_label,
                        img_width,
                        img_height,
                        self.global_detection_id,
                        confidence_threshold=ps_threshold,
                        max_detections=ps_max_detections,
                    )
                    ps_detections.extend(prompt_detections)
                    self.global_detection_id += len(prompt_detections)

                # Apply max_detections limit to entire prompt set (sort by score, take top N)
                if len(ps_detections) > ps_max_detections:
                    ps_detections.sort(key=lambda d: d.get("score", 0), reverse=True)
                    ps_detections = ps_detections[:ps_max_detections]

                # Create output frame for this topic (deep copy to avoid shared state)
                output_frame = copy.deepcopy(frame)
                output_meta = output_frame.data.setdefault("meta", {})

                # Store results in frame data under the canonical 'detections' key
                canonical_dict, protege_list, classification_dict = (
                    self._normalize_detections(ps_detections)
                )
                output_frame.data[FilterSAM3DetectorOutput.__frame_data_key__] = (
                    canonical_dict
                )

                # Store width and height in meta for backward-compatibility
                output_meta["width"] = img_width
                output_meta["height"] = img_height

                # Restore legacy dual-writes for unmigrated consumers
                output_meta["detections"] = protege_list
                output_meta[self.output_label] = ps_detections
                output_meta["classification"] = classification_dict

                # Keep serialized_detections pointing to canonical for the JSONL write below
                serialized_detections = canonical_dict

                # Add to output frames with the prompt set's topic
                output_frames[ps_topic] = output_frame

                # Save annotated frame for this prompt set (per-set detections drawn on original image)
                if self.annotated_frames_dir is not None and ps_detections:
                    try:
                        import cv2

                        # Include prompt set name in filename to disambiguate across sets
                        annotated_filename = f"frame_{frame_id_str}_ts{timestamp_str}_count{frame_counter:06d}_{ps_name}.jpg"
                        image_bgr_annotated = image_bgr.copy()
                        image_bgr_annotated = self._visualize_detections_on_image(
                            image_bgr_annotated,
                            ps_detections,
                            self.positive_boxes if has_ref_boxes else None,
                            self.negative_boxes if has_ref_boxes else None,
                        )
                        cv2.imwrite(
                            str(self.annotated_frames_dir / annotated_filename),
                            image_bgr_annotated,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to save annotated frame for prompt set {ps_name}: {e}"
                        )

                # Write to JSONL if configured (one record per prompt set per frame)
                if hasattr(self, "jsonl_file") and self.jsonl_file is not None:
                    try:
                        output_frame_id = (
                            frame_id_num if frame_id_num is not None else frame_counter
                        )
                        jsonl_meta = {
                            "frame_id": output_frame_id,
                            "width": img_width,
                            "height": img_height,
                            "frame_filename": frame_filename_str,
                        }

                        event_record = {
                            "filter_name": ps_filter_name,
                            "topic": ps_topic,
                            "data": {
                                "id": output_frame_id,
                                "detections": serialized_detections,
                                "meta": jsonl_meta,
                            },
                        }
                        self.jsonl_file.write(json.dumps(event_record) + "\n")
                        self.jsonl_file.flush()
                    except Exception as e:
                        logger.warning(
                            f"Failed to save JSONL for prompt set {ps_name}: {e}"
                        )

            except Exception as e:
                logger.error(
                    f"Error processing prompt set {prompt_set.get('name', 'unknown')}: {e}",
                    exc_info=True,
                )
                ps_topic = prompt_set.get("topic", "main")
                output_frame = copy.deepcopy(frame)

                # Normalize empty detections for the degraded frame
                canonical_dict, protege_list, classification_dict = (
                    self._normalize_detections([])
                )
                output_frame.data[FilterSAM3DetectorOutput.__frame_data_key__] = (
                    canonical_dict
                )

                output_meta = output_frame.data.setdefault("meta", {})
                output_meta["width"] = img_width
                output_meta["height"] = img_height
                output_meta["detections"] = protege_list
                output_meta[self.output_label] = []
                output_meta["classification"] = classification_dict

                output_frames[ps_topic] = output_frame
        return output_frames

    def _normalize_detections(
        self, detections: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """
        Single normalization pass to produce both the canonical DetectionSet dict
        and the protege-compatible legacy detections list.
        Returns: (canonical_dict, protege_list, classification_dict)
        """
        clean_items = []
        protege_items = []
        class_max_scores = {}

        for d in detections:
            clean_d = {}
            protege_d = {}

            # --- Box Geometry ---
            xyxy = to_xyxy(d)
            if not xyxy:
                continue
            x1, y1, x2, y2 = xyxy

            clean_d["bbox"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            protege_d["bbox"] = {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
            protege_d["rois"] = [[int(x1), int(y1), int(x2), int(y2)]]

            # --- Score ---
            score = 0.0
            score_val = d.get("score")
            if score_val is None:
                score_val = d.get("confidence")
            if score_val is not None:
                try:
                    score = max(0.0, min(float(score_val), 1.0))
                except (ValueError, TypeError):
                    score = 0.0
            clean_d["score"] = score
            protege_d["confidence"] = score

            # --- Label ---
            label_val = d.get("label")
            if label_val is None:
                label_val = d.get("class")
            if label_val is None:
                label_val = d.get("class_name")
            if label_val is None:
                label_val = "object"
            label = str(label_val)

            prompt_val = d.get("prompt")
            if prompt_val is None:
                prompt_val = d.get("class")
            if prompt_val is None:
                prompt_val = label
            prompt = str(prompt_val)

            clean_d["label"] = label
            protege_d["label"] = label
            protege_d["class"] = label
            protege_d["prompt"] = prompt

            if "label_id" in d and d["label_id"] is not None:
                try:
                    try:
                        clean_d["label_id"] = int(d["label_id"])
                    except ValueError:
                        clean_d["label_id"] = int(float(d["label_id"]))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Skipping invalid non-integer label_id: {d['label_id']}"
                    )

            # --- Mask ---
            if "mask" in d and isinstance(d["mask"], dict):
                mask = d["mask"]
                polygons = mask.get("polygons", [])
                valid_polygons = []
                for poly in polygons:
                    if isinstance(poly, dict):
                        pts = poly.get("points", [])
                    elif isinstance(poly, (list, tuple)):
                        pts = poly
                    else:
                        pts = getattr(poly, "points", [])
                    if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                        valid_polygons.append({"points": pts})
                if valid_polygons:
                    clean_mask = {"polygons": valid_polygons}
                    if "area" in mask:
                        clean_mask["area"] = int(mask["area"])
                    clean_d["mask"] = clean_mask

            # Validate the single item against the official Detection schema
            from openfilter.filter_runtime.shapes import Detection

            try:
                validated_item = Detection(**clean_d)
                clean_items.append(validated_item.model_dump(mode="json"))
                protege_items.append(protege_d)

                # Only track class scores of validated and kept detections
                if label not in class_max_scores:
                    class_max_scores[label] = 0.0
                class_max_scores[label] = max(class_max_scores[label], score)
            except Exception as e:
                logger.warning(
                    f"Filtering invalid detection item due to validation failure: {clean_d}. Error: {e}"
                )

        # Build canonical
        try:
            canonical_dict = FilterSAM3DetectorOutput(items=clean_items).model_dump(
                mode="json"
            )
        except pydantic.ValidationError as e:
            logger.warning(
                f"Detection schema validation failed, falling back to clean items: {e}"
            )
            canonical_dict = {
                "items": clean_items,
                "__schema_id__": FilterSAM3DetectorOutput.__schema_id__,
            }

        # Build classification
        sorted_classes = sorted(
            class_max_scores.items(), key=lambda x: x[1], reverse=True
        )
        classification = {
            "classes": [cls for cls, _ in sorted_classes],
            "confidences": [score for _, score in sorted_classes],
            "architecture": "sam3",
        }

        return canonical_dict, protege_items, classification

    def _boxes_xywh_to_norm_cxcywh(
        self, boxes_xywh: list, img_w: int, img_h: int
    ) -> list:
        """Convert list of [x, y, w, h] (pixels) to normalized [cx, cy, w, h] in [0, 1] for add_geometric_prompt."""
        if not boxes_xywh:
            return []
        out = []
        for box in boxes_xywh:
            if len(box) != 4:
                continue
            box_t = torch.tensor(
                [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_t)
            norm = normalize_bbox(box_cxcywh, img_w, img_h)
            out.append(norm.flatten().tolist())
        return out

    def _extract_detections_from_state(
        self,
        state: dict,
        class_name: str,
        img_width: int,
        img_height: int,
        id_offset: int = 0,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> list:
        """
        Extract detections from processor state with class labeling.

        This helper extracts boxes, scores, and masks from the state dict
        and formats them as detection dictionaries with proper class labels.

        Args:
            state: State dict from processor with boxes, scores, masks
            class_name: Class name to assign to all detections from this prompt
            img_width: Image width for ROI normalization
            img_height: Image height for ROI normalization
            id_offset: Starting ID for detection numbering (for multi-prompt)
            confidence_threshold: Override for minimum confidence (default: self.confidence_threshold)
            max_detections: Override for max detections (default: self.max_detections)

        Returns:
            List of detection dictionaries
        """
        detections = []

        # Use provided overrides or instance defaults
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        max_dets = max_detections if max_detections is not None else self.max_detections

        if "boxes" not in state or "scores" not in state:
            return detections

        boxes = state["boxes"]
        scores = state["scores"]
        masks = state.get("masks", None)

        # Apply NMS if enabled to suppress overlapping detections
        if self.nms_enabled and len(boxes) > 0:
            boxes, scores, masks = self._apply_nms(boxes, scores, masks)

        actual_id = 0  # Track actual number of detections added
        for i in range(len(boxes)):
            # Apply confidence threshold filter
            score = scores[i]
            if hasattr(score, "item"):
                score_val = score.item()
            else:
                score_val = float(score)

            if score_val < conf_thresh:
                continue

            # Stop if we've reached max detections
            if actual_id >= max_dets:
                break

            detection = {}
            detection_id = id_offset + actual_id + 1  # COCO annotation ID (1-indexed)
            actual_id += 1

            box = boxes[i]
            if hasattr(box, "tolist"):
                box = box.tolist()
            box = [float(x) for x in box]

            # Clip box coordinates to image boundaries
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(x1, img_width))
            y1 = max(0.0, min(y1, img_height))
            x2 = max(0.0, min(x2, img_width))
            y2 = max(0.0, min(y2, img_height))

            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = min(x1 + 1.0, img_width)
            if y2 <= y1:
                y2 = min(y1 + 1.0, img_height)

            # Keep original format [x1, y1, x2, y2] for compatibility (clipped)
            detection["box"] = [float(x1), float(y1), float(x2), float(y2)]

            # Create canonical bbox dict
            detection["bbox"] = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }

            score = scores[i]
            if hasattr(score, "item"):
                score = score.item()
            detection["score"] = float(score)
            # Also add 'confidence' for protege compatibility
            detection["confidence"] = float(score)

            if self.output_masks and masks is not None and i < len(masks):
                mask = masks[i]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                if hasattr(mask, "squeeze"):
                    mask = mask.squeeze()

                # Convert to binary mask
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Keep in-memory for internal visualization
                if self.visualize or self.annotated_frames_dir is not None:
                    detection["mask_np"] = binary_mask

                # Convert mask to COCO format (polygons)
                segmentation = self._mask_to_coco_polygons(binary_mask)

                if segmentation:
                    detection["segmentation"] = segmentation

                    # Calculate area (number of pixels in mask)
                    area = int(np.sum(binary_mask))
                    detection["area"] = area

                    # Build canonical mask schema dict from segmentation
                    polygons_list = []
                    for seg_poly in segmentation:
                        # seg_poly is [x1, y1, x2, y2, ...]
                        points = [
                            (float(seg_poly[j]), float(seg_poly[j + 1]))
                            for j in range(0, len(seg_poly), 2)
                        ]
                        if len(points) >= 3:
                            polygons_list.append({"points": points})

                    if polygons_list:
                        detection["mask"] = {
                            "polygons": polygons_list,
                            "area": area,
                        }

                    # Category ID (1 = object, since we don't have specific categories)
                    detection["category_id"] = 1

                    # iscrowd (0 = single object, 1 = crowd)
                    detection["iscrowd"] = 0

            if detection:
                detection["id"] = detection_id

                # Add class name from the prompt
                detection["class"] = class_name
                detection["class_name"] = class_name
                detection["category_name"] = class_name
                detection["label"] = class_name

                # Add rois with pixel coordinates [x1, y1, x2, y2]
                # This matches protege detection model output format for aggregator compatibility
                if "box" in detection:
                    x1, y1, x2, y2 = detection["box"]
                    detection["rois"] = [[int(x1), int(y1), int(x2), int(y2)]]

                # Add category_id if not already set (for COCO compatibility)
                if "category_id" not in detection:
                    detection["category_id"] = 1

                detections.append(detection)

        return detections

    def _cache_text_embeddings(self):
        """
        Pre-cache text embeddings for all prompts at startup.

        This is a critical optimization: text prompts are static, so we encode them
        ONCE during setup instead of on every frame. The text encoder is a significant
        portion of inference time, so caching these embeddings gives us a major speedup.

        The cached embeddings are stored in self.cached_text_embeddings[prompt] and
        contain the language_features, language_mask, and language_embeds that would
        normally be computed by set_text_prompt_no_grounding().
        """
        if self.model is None:
            logger.warning("Cannot cache text embeddings: model not loaded")
            return

        # Collect all unique prompts from all prompt sets
        all_prompts = set()
        if self.prompt_sets:
            for ps in self.prompt_sets:
                for prompt in ps.get("prompts", []):
                    all_prompts.add(prompt)

        # Also cache single text_prompt and text_prompts if configured
        if self.text_prompt:
            all_prompts.add(self.text_prompt)
        if self.text_prompts:
            for prompt in self.text_prompts:
                all_prompts.add(prompt)

        if not all_prompts:
            logger.debug("No prompts to cache")
            return

        logger.info(
            f"Pre-caching text embeddings for {len(all_prompts)} unique prompts..."
        )

        with torch.no_grad():
            for prompt in all_prompts:
                try:
                    # Encode the text prompt using the model's backbone
                    text_outputs = self.model.backbone.forward_text(
                        [prompt], device=str(self.device)
                    )

                    # Store the relevant tensors (detach to avoid memory leaks)
                    self.cached_text_embeddings[prompt] = {
                        "language_features": text_outputs.get("language_features"),
                        "language_mask": text_outputs.get("language_mask"),
                        "language_embeds": text_outputs.get("language_embeds"),
                    }
                    logger.debug(f"Cached text embedding for prompt: '{prompt}'")

                except Exception as e:
                    logger.warning(
                        f"Failed to cache text embedding for '{prompt}': {e}"
                    )

        logger.info(f"Cached {len(self.cached_text_embeddings)} text embeddings")

    def _inject_cached_text_embedding(self, state: dict, prompt: str) -> dict:
        """
        Inject cached text embedding into state, avoiding re-encoding.

        This replaces the call to processor.set_text_prompt_no_grounding() with
        a direct injection of pre-computed language features.

        Args:
            state: State dict from set_image() with backbone_out
            prompt: Text prompt to inject embedding for

        Returns:
            Updated state with language features injected
        """
        if prompt not in self.cached_text_embeddings:
            # Fallback: encode the prompt on-the-fly (shouldn't happen normally)
            logger.warning(
                f"Text embedding not cached for '{prompt}', encoding on-the-fly"
            )
            return self.processor.set_text_prompt_no_grounding(prompt, state)

        cached = self.cached_text_embeddings[prompt]

        # Inject cached language features into backbone_out
        if cached.get("language_features") is not None:
            state["backbone_out"]["language_features"] = cached["language_features"]
        if cached.get("language_mask") is not None:
            state["backbone_out"]["language_mask"] = cached["language_mask"]
        if cached.get("language_embeds") is not None:
            state["backbone_out"]["language_embeds"] = cached["language_embeds"]

        # Initialize geometric prompt if not present (same as set_text_prompt_no_grounding)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return state

    # -- Batched backbone inference (FILTER-369) ----------------------------------

    @torch.no_grad()
    def process_batch(
        self, batch: list[dict[str, Frame]]
    ) -> list[dict[str, Frame] | Frame | None]:
        if not self._can_batch():
            return [self.process(frames) for frames in batch]

        pil_images = []
        valid_indices: set[int] = set()
        batch_idx_map: dict[int, int] = {}
        for i, frames in enumerate(batch):
            pil = self._extract_pil_image(frames)
            if pil is not None:
                batch_idx_map[i] = len(pil_images)
                pil_images.append(pil)
                valid_indices.add(i)

        if not pil_images:
            return [self.process(frames) for frames in batch]

        try:
            batched_state = self.processor.set_image_batch(pil_images)
            per_frame_states = self._split_backbone_states(batched_state)

            results: list[dict[str, Frame] | Frame | None] = [None] * len(batch)
            for i in range(len(batch)):
                if i in valid_indices:
                    frame_state = per_frame_states[batch_idx_map[i]]
                    self._cached_backbone_state = {
                        **frame_state,
                        "backbone_out": {**frame_state["backbone_out"]},
                    }
                    try:
                        results[i] = self.process(batch[i])
                    except Exception as frame_err:
                        logger.error(f"Batched frame {i} failed: {frame_err}")
                        results[i] = batch[i]
                    finally:
                        self._cached_backbone_state = None
                else:
                    results[i] = self.process(batch[i])

            return results

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(
                f"set_image_batch OOM: {e}. Clearing cache and falling back to per-frame."
            )
            self._cached_backbone_state = None
            torch.cuda.empty_cache()
            return [self.process(frames) for frames in batch]
        except Exception as e:
            logger.warning(f"set_image_batch failed: {e}. Falling back to per-frame.")
            self._cached_backbone_state = None
            return [self.process(frames) for frames in batch]

    def _can_batch(self) -> bool:
        if self.model is None or self.processor is None:
            return False
        if self.prompt_sets:
            return False
        if self.positive_boxes or self.negative_boxes:
            return False
        if self.ref_images_paths or self.ref_images_negative_paths:
            return False
        if self.enable_video_mode:
            return False
        has_prompts = bool(self.text_prompts or self.text_prompt)
        has_visual = self.visual_prompt_embed is not None
        return has_prompts or has_visual

    def _extract_pil_image(self, frames: dict[str, Frame]) -> Optional[Image.Image]:
        # Returns the first non-auxiliary image topic. Standard pipelines have
        # one image topic; if multiple exist, dict iteration order determines
        # which is used for backbone inference.
        for topic, frame in frames.items():
            # Skip auxiliary topics (e.g. _filter, SourceName___filter)
            if topic == "_filter" or topic.endswith("___filter"):
                continue
            if frame is not None and frame.has_image:
                image_bgr = frame.rw_bgr.image
                image_rgb = image_bgr[:, :, ::-1]
                return Image.fromarray(image_rgb)
        return None

    def _split_backbone_states(self, batched_state: dict) -> list[dict]:
        n = len(batched_state["original_heights"])
        split_backbone = self._split_tensor_dict(batched_state["backbone_out"], n)
        return [
            {
                "original_height": batched_state["original_heights"][i],
                "original_width": batched_state["original_widths"][i],
                "backbone_out": split_backbone[i],
            }
            for i in range(n)
        ]

    def _split_tensor_dict(self, d: dict, n: int) -> list[dict]:
        """Split a dict of batched tensors into a list of per-frame dicts.

        Return shape per value type:
        - Tensor (dim >= 1): split along dim 0 → each frame gets a (1, ...) tensor
        - list[Tensor]: transposed → each frame gets a list of (1, ...) tensors
        - dict: recursed → each frame gets a nested dict with the same structure
        - scalar: replicated → each frame gets the same value
        """
        split_vals: dict[str, list] = {}
        for key, val in d.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1:
                split_vals[key] = val.split(1, dim=0)
            elif isinstance(val, list):
                split_vals[key] = list(
                    zip(
                        *(
                            t.split(1, dim=0)
                            if isinstance(t, torch.Tensor) and t.dim() >= 1
                            else (t,) * n
                            for t in val
                        )
                    )
                )
            elif isinstance(val, dict):
                split_vals[key] = self._split_tensor_dict(val, n)
            else:
                split_vals[key] = (val,) * n

        keys = list(split_vals.keys())
        return [
            {
                k: split_vals[k][i]
                if not isinstance(d[k], list)
                else list(split_vals[k][i])
                for k in keys
            }
            for i in range(n)
        ]

    # -- End batched backbone inference -------------------------------------------

    def _load_model(self):
        """
        Load the SAM3 model from HuggingFace.
        """
        if not HAS_SAM3:
            logger.error(
                "SAM3 not available. Install from: https://github.com/facebookresearch/sam3"
            )
            return

        try:
            logger.info(f"Loading SAM3 model on device: {self.device}")

            # Find BPE path - check vendorized sam3 first, then installed package
            bpe_path = None
            # Try vendorized sam3 (in project root/sam3/assets/)
            vendorized_bpe = (
                Path(__file__).parent.parent
                / "sam3"
                / "assets"
                / "bpe_simple_vocab_16e6.txt.gz"
            )
            if vendorized_bpe.exists():
                bpe_path = str(vendorized_bpe)
                logger.info(f"Using vendorized BPE file: {bpe_path}")
            else:
                # Try vendorized sam3/sam3/assets/ (alternative location)
                vendorized_bpe2 = (
                    Path(__file__).parent.parent
                    / "sam3"
                    / "sam3"
                    / "assets"
                    / "bpe_simple_vocab_16e6.txt.gz"
                )
                if vendorized_bpe2.exists():
                    bpe_path = str(vendorized_bpe2)
                    logger.info(f"Using vendorized BPE file: {bpe_path}")
                else:
                    # Try to find in installed package
                    try:
                        import sam3

                        sam3_path = Path(sam3.__file__).parent
                        installed_bpe = (
                            sam3_path / "assets" / "bpe_simple_vocab_16e6.txt.gz"
                        )
                        if installed_bpe.exists():
                            bpe_path = str(installed_bpe)
                            logger.info(f"Using installed BPE file: {bpe_path}")
                    except Exception as e:
                        logger.debug(f"Could not find BPE in installed package: {e}")

            if bpe_path is None:
                logger.warning("BPE file not found, SAM3 will try to use default path")

            # Build SAM3 model
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                device=str(self.device),
                eval_mode=True,
                load_from_HF=True,
            )

            # Create processor
            self.processor = Sam3Processor(
                self.model,
                device=str(self.device),
                confidence_threshold=self.confidence_threshold,
            )

            # Enable persistent bfloat16 autocast for all subsequent inference
            # (mirrors SAM3 video path in sam3_tracking_predictor.py)
            if self.mixed_precision:
                if not torch.cuda.is_bf16_supported():
                    logger.warning(
                        "bfloat16 not natively supported on this GPU, disabling mixed precision"
                    )
                    self.mixed_precision = False
                else:
                    self._autocast_persistent = torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    )
                    self._autocast_persistent.__enter__()
                    logger.info(
                        "Entered persistent bfloat16 autocast context for inference"
                    )

            logger.info(f"SAM3 model loaded successfully on {self.device}")

        except Exception as e:
            # Clean up autocast context if it was entered before the failure
            if self._autocast_persistent is not None:
                self._autocast_persistent.__exit__(None, None, None)
                self._autocast_persistent = None
            logger.error(f"Failed to load SAM3 model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.model = None
            self.processor = None

    def _load_video_model(self):
        """
        Load the SAM3 video model for streaming video processing.

        Video mode uses memory-based tracking for faster inference:
        - First frame: Full detection
        - Subsequent frames: Memory-based propagation (10x faster)
        - Periodic re-detection to correct drift

        This is ideal for video streams at 5-6 fps (matches SAM3's training).
        """
        if not HAS_SAM3:
            logger.error(
                "SAM3 not available. Install from: https://github.com/facebookresearch/sam3"
            )
            return

        try:
            logger.info(f"Loading SAM3 VIDEO model on device: {self.device}")
            logger.info(f"  detection_interval={self.video_detection_interval}")
            logger.info(
                f"  min_tracking_confidence={self.video_min_tracking_confidence}"
            )

            # Find BPE path
            bpe_path = self._find_bpe_path()

            # Create streaming video processor
            self.video_processor = StreamingVideoProcessor(
                device=str(self.device),
                confidence_threshold=self.confidence_threshold,
                detection_interval=self.video_detection_interval,
                min_tracking_confidence=self.video_min_tracking_confidence,
                bpe_path=bpe_path,
            )

            if not self.video_processor.load_model():
                logger.error("Failed to load video model")
                self.video_processor = None
                return

            # Pre-cache text embeddings for all configured prompts
            all_prompts = set()
            if self.prompt_sets:
                for ps in self.prompt_sets:
                    for prompt in ps.get("prompts", []):
                        all_prompts.add(prompt)
            if self.text_prompt:
                all_prompts.add(self.text_prompt)
            if self.text_prompts:
                for prompt in self.text_prompts:
                    all_prompts.add(prompt)

            if all_prompts:
                self.video_processor.cache_text_embeddings(list(all_prompts))

            # Also set model and processor for compatibility with existing code paths
            self.model = self.video_processor.model
            self.processor = self.video_processor.processor

            logger.info(f"SAM3 video model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load SAM3 video model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.video_processor = None

    def _find_bpe_path(self) -> Optional[str]:
        """Find the BPE tokenizer file path."""
        # Try vendorized sam3 (in project root/sam3/assets/)
        vendorized_bpe = (
            Path(__file__).parent.parent
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        if vendorized_bpe.exists():
            return str(vendorized_bpe)

        # Try vendorized sam3/sam3/assets/ (alternative location)
        vendorized_bpe2 = (
            Path(__file__).parent.parent
            / "sam3"
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        if vendorized_bpe2.exists():
            return str(vendorized_bpe2)

        # Try to find in installed package
        try:
            import sam3

            sam3_path = Path(sam3.__file__).parent
            installed_bpe = sam3_path / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if installed_bpe.exists():
                return str(installed_bpe)
        except Exception as e:
            logger.debug(f"Could not find BPE in installed package: {e}")

        logger.warning("BPE file not found, SAM3 will try to use default path")
        return None

    def _load_exemplar_images(self):
        """
        Load exemplar images from a directory and compute their visual embeddings.

        The directory should contain cropped images (jpg/png) showing examples of what to detect.
        Each image will be encoded with SAM3's backbone and the features averaged
        to create a visual prompt embedding.
        """
        if not self.exemplars_path:
            return

        if self.model is None:
            logger.error("Model must be loaded before loading exemplar images")
            return

        try:
            exemplars_path = Path(self.exemplars_path)
            if not exemplars_path.exists():
                logger.warning(f"Exemplars path not found: {self.exemplars_path}")
                return

            # Find all image files
            if exemplars_path.is_file():
                # Single image file
                image_files = [exemplars_path]
            else:
                # Directory of images
                image_files = [
                    f
                    for f in exemplars_path.iterdir()
                    if f.suffix.lower() in REF_IMAGE_EXTENSIONS
                ]

            if not image_files:
                logger.warning(f"No image files found in {self.exemplars_path}")
                return

            logger.info(
                f"Loading {len(image_files)} exemplar images from {self.exemplars_path}"
            )

            # Encode each exemplar image and collect features
            all_embeddings = []

            for img_path in image_files:
                try:
                    # Load image
                    pil_image = Image.open(img_path).convert("RGB")

                    # Encode with SAM3 backbone
                    with torch.no_grad():
                        # Use the processor's transform
                        image_tensor = self.processor.transform(
                            torch.from_numpy(np.array(pil_image))
                            .permute(2, 0, 1)
                            .to(self.device)
                        ).unsqueeze(0)

                        # Get backbone features
                        backbone_out = self.model.backbone.forward_image(image_tensor)

                        # Extract the main image embedding and pool it
                        # The backbone_out contains multi-scale features; we use the highest level
                        # Note: sam2_backbone_out key may exist but be None, so check value not just key
                        sam2_out = backbone_out.get("sam2_backbone_out")
                        if sam2_out is not None and "backbone_fpn" in sam2_out:
                            # Use SAM2 backbone features
                            feats = sam2_out["backbone_fpn"][-1]
                        elif (
                            "backbone_fpn" in backbone_out
                            and backbone_out["backbone_fpn"]
                        ):
                            # Use direct backbone_fpn features (SAM3 format)
                            feats = backbone_out["backbone_fpn"][-1]
                        elif "vision_features" in backbone_out:
                            # Fallback to vision_features
                            feats = backbone_out["vision_features"]
                        else:
                            raise ValueError(
                                f"Cannot extract features from backbone output. Keys: {backbone_out.keys()}"
                            )

                        # Global average pooling to get a single embedding per image
                        # Shape: [1, C, H, W] -> [1, C]
                        pooled = feats.mean(dim=[2, 3])
                        all_embeddings.append(pooled)

                    logger.debug(f"Encoded exemplar: {img_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to load exemplar {img_path}: {e}")
                    continue

            if not all_embeddings:
                logger.error("No exemplar images could be loaded")
                return

            # Stack all embeddings as separate visual tokens (no averaging)
            # This allows each exemplar to contribute independently to detection
            # The model will see all exemplars and can match against any of them
            stacked = torch.cat(all_embeddings, dim=0)  # [N, C]

            # Format for SAM3's visual prompt: [seq_len, batch, hidden_dim]
            # Each exemplar becomes a separate visual token in the prompt sequence
            num_exemplars = len(all_embeddings)
            self.visual_prompt_embed = stacked.unsqueeze(1)  # [N, 1, C]
            self.visual_prompt_mask = torch.zeros(
                (1, num_exemplars), device=self.device, dtype=torch.bool
            )  # No masking for any exemplar

            logger.info(
                f"Created visual prompt with {num_exemplars} exemplar tokens (shape: {self.visual_prompt_embed.shape})"
            )

        except Exception as e:
            logger.error(f"Failed to load exemplar images: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.visual_prompt_embed = None
            self.visual_prompt_mask = None

    def _expand_ref_paths(self, paths):
        """
        Expand ref path list: directories become sorted list of image files; files kept as-is.
        Returns None if input is None/empty or if expansion yields no files.
        """
        if not paths:
            return None
        result = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                logger.warning(f"Ref path does not exist: {p}")
                continue
            if p.is_dir():
                files = sorted(
                    f
                    for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in REF_IMAGE_EXTENSIONS
                )
                if not files:
                    logger.warning(f"No image files found in ref directory: {p}")
                result.extend(files)
            else:
                result.append(p)
        return result if result else None

    def _box_overlaps_any_region(self, box, regions):
        """Return True if box [x1,y1,x2,y2] overlaps any region in regions (list of [x1,y1,x2,y2])."""
        if not regions or not box or len(box) != 4:
            return False
        bx1, by1, bx2, by2 = box
        for r in regions:
            if len(r) != 4:
                continue
            rx1, ry1, rx2, ry2 = r
            if not (bx2 <= rx1 or bx1 >= rx2 or by2 <= ry1 or by1 >= ry2):
                return True
        return False

    def _box_center_inside_any_region(self, box, regions):
        """Return True if the center of box [x1,y1,x2,y2] lies inside any region [x1,y1,x2,y2]."""
        if not regions or not box or len(box) != 4:
            return False
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        for r in regions:
            if len(r) != 4:
                continue
            rx1, ry1, rx2, ry2 = r
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    def _load_ref_images_for_cache(self, paths: list) -> list:
        """Load and resize ref images to ref_max_height once (used in setup). Returns list of PIL Image or None per path."""
        out = []
        max_h = self.ref_max_height
        for path in paths:
            try:
                im = Image.open(path).convert("RGB")
                w, h = im.size
                if h > max_h:
                    im = im.resize(
                        (int(w * max_h / h), max_h), Image.Resampling.LANCZOS
                    )
                out.append(im)
            except Exception as e:
                logger.warning(f"Failed to load ref image {path}: {e}")
                out.append(None)
        return out

    def _build_composite_with_refs(self, pil_image: Image.Image):
        """
        Build composite image with ref images: positive left, negative right.
        ref_layout "overlay": refs on frame (bottom-left = positive, bottom-right = negative).
        ref_layout "side_strips": lateral strips; left = positive, center = frame, right = negative.
        Returns (composite_pil, all_norm_labels, frame_offset_x, ref_regions_frame).
        frame_offset_x is None for overlay; for side_strips it is the x offset of the frame in the composite.
        ref_regions_frame: list of [x1,y1,x2,y2] in frame coords for overlay (negative ref regions only; detections overlapping these are removed; positive ref may keep detections); None for side_strips.
        """
        cached_pos = self._cached_ref_images_pos or []
        cached_neg = self._cached_ref_images_neg or []
        if not cached_pos and not cached_neg:
            return (pil_image, [], None, None)

        img_w, img_h = pil_image.size
        margin = self.ref_margin
        gap = self.ref_gap
        # Cap ref height by frame height so layout never breaks (overlay or side_strips)
        max_ref_height = min(self.ref_max_height, img_h)
        all_norm_labels = []
        ref_regions_frame = []  # negative ref regions only (for filtering detections; positive ref may keep model detections)

        def ref_im_for_frame(cached_im: Optional[Image.Image]) -> Optional[Image.Image]:
            """Use cached image, optionally resize down when frame is smaller than ref_max_height."""
            if cached_im is None:
                return None
            w, h = cached_im.size
            if h > max_ref_height:
                return cached_im.resize(
                    (int(w * max_ref_height / h), max_ref_height),
                    Image.Resampling.LANCZOS,
                )
            return cached_im

        if self.ref_layout == "side_strips":
            strip_w = self.ref_strip_width
            composite_w = strip_w + img_w + strip_w
            composite_h = img_h
            composite = Image.new("RGB", (composite_w, composite_h), (128, 128, 128))
            composite.paste(pil_image, (strip_w, 0))

            def paste_refs_side(cached_list: list, positive: bool):
                nonlocal all_norm_labels
                x_start = 0 if positive else (strip_w + img_w)
                strip_width = strip_w
                y_cur = composite_h - margin
                for cached_im in cached_list:
                    ref_im = ref_im_for_frame(cached_im)
                    if ref_im is None:
                        continue
                    rw, rh = ref_im.size
                    rw = min(rw, strip_width - 2 * margin)
                    if rw <= 0:
                        continue
                    ref_im = (
                        ref_im.resize((int(rw), int(rh)), Image.Resampling.LANCZOS)
                        if ref_im.size[0] != rw
                        else ref_im
                    )
                    rw, rh = ref_im.size
                    y_cur -= rh
                    py = y_cur
                    if py < margin:
                        break
                    px = (
                        x_start + margin
                        if positive
                        else x_start + (strip_width - rw - margin)
                    )
                    composite.paste(ref_im, (px, py))
                    box_xywh = [float(px), float(py), float(rw), float(rh)]
                    norm_list = self._boxes_xywh_to_norm_cxcywh(
                        [box_xywh], composite_w, composite_h
                    )
                    if norm_list:
                        all_norm_labels.append((norm_list[0], positive))
                    y_cur -= gap

            paste_refs_side(cached_pos, True)
            paste_refs_side(cached_neg, False)
            return (composite, all_norm_labels, strip_w, None)
        else:
            # overlay
            composite = pil_image.copy()
            y_cur = img_h - margin

            def paste_refs_overlay(cached_list: list, positive: bool):
                nonlocal all_norm_labels, y_cur, ref_regions_frame
                for cached_im in cached_list:
                    ref_im = ref_im_for_frame(cached_im)
                    if ref_im is None:
                        continue
                    rw, rh = ref_im.size
                    y_cur -= rh
                    py = y_cur
                    if py < margin:
                        break
                    if positive:
                        px = margin
                    else:
                        px = img_w - margin - rw
                    composite.paste(ref_im, (px, py))
                    # Only filter detections in negative ref regions; positive ref may keep model detections
                    if not positive:
                        ref_regions_frame.append([px, py, px + rw, py + rh])
                    box_xywh = [float(px), float(py), float(rw), float(rh)]
                    norm_list = self._boxes_xywh_to_norm_cxcywh(
                        [box_xywh], img_w, img_h
                    )
                    if norm_list:
                        all_norm_labels.append((norm_list[0], positive))
                    y_cur -= gap

            paste_refs_overlay(cached_pos, True)
            y_cur = img_h - margin
            paste_refs_overlay(cached_neg, False)
            return (composite, all_norm_labels, None, ref_regions_frame)

    def _forward_grounding_with_visual_prompt(self, state):
        """
        Run SAM3 grounding with visual prompt embeddings from exemplar images.

        This manually calls the model's internal methods to inject visual prompts,
        since forward_grounding doesn't accept visual_prompt_embed directly.
        """
        import torch
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        backbone_out = state["backbone_out"]
        find_input = self.processor.find_stage
        geometric_prompt = state["geometric_prompt"]

        # Ensure visual prompt is on the correct device (match backbone output)
        # Get the device from the backbone output
        target_device = backbone_out.get(
            "vision_features", self.visual_prompt_embed
        ).device
        visual_embed = self.visual_prompt_embed.to(target_device)
        visual_mask = self.visual_prompt_mask.to(target_device)

        # Encode prompt with visual embeddings - call internal method directly
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out,
                find_input,
                geometric_prompt,
                visual_prompt_embed=visual_embed,
                visual_prompt_mask=visual_mask,
            )

        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )

        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self.model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        outputs = out

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # Convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes.device)
        boxes = boxes * scale_fct[None, :]

        # Apply NMS to reduce overlapping detections
        if len(boxes) > 0 and self.nms_threshold > 0:
            from torchvision.ops import nms

            nms_keep = nms(boxes, out_probs, self.nms_threshold)
            boxes = boxes[nms_keep]
            out_probs = out_probs[nms_keep]
            out_masks = out_masks[nms_keep]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _apply_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, masks: torch.Tensor = None
    ) -> tuple:
        """
        Apply Non-Maximum Suppression (NMS) to filter overlapping detections.

        Uses torchvision.ops.nms which iteratively removes lower-scoring boxes
        that have IoU greater than the threshold with a higher-scoring box.

        Args:
            boxes: Tensor of shape [N, 4] with boxes in (x1, y1, x2, y2) format
            scores: Tensor of shape [N] with confidence scores
            masks: Optional tensor of masks, shape [N, ...]

        Returns:
            Tuple of (filtered_boxes, filtered_scores, filtered_masks)
            where filtered_masks is None if input masks is None
        """
        try:
            from torchvision.ops import nms
        except ImportError:
            logger.warning("torchvision.ops.nms not available, skipping NMS")
            return boxes, scores, masks

        if len(boxes) == 0:
            return boxes, scores, masks

        # Ensure tensors are on the same device and have correct dtype
        device = boxes.device
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, device=device)
        if scores.device != device:
            scores = scores.to(device)

        # torchvision.ops.nms expects boxes as float and scores as 1D tensor
        boxes_float = boxes.float()
        scores_1d = scores.flatten() if scores.dim() > 1 else scores

        # Apply NMS - returns indices of boxes to keep, sorted by score (descending)
        keep_indices = nms(boxes_float, scores_1d, self.nms_threshold)

        # Filter boxes, scores, and masks
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        filtered_masks = masks[keep_indices] if masks is not None else None

        num_before = len(boxes)
        num_after = len(keep_indices)
        if num_before > num_after:
            logger.debug(
                f"NMS: {num_before} -> {num_after} detections (IoU threshold={self.nms_threshold})"
            )

        return filtered_boxes, filtered_scores, filtered_masks

    @staticmethod
    def _viz_bgr_for_class_name(class_name: Optional[str]) -> tuple[int, int, int]:
        """Stable BGR color per class; default blue when class unknown."""
        if not class_name or not str(class_name).strip():
            return (255, 0, 0)
        digest = hashlib.md5(str(class_name).strip().lower().encode("utf-8")).digest()
        b, g, r = digest[0], digest[1], digest[2]
        return (max(int(b), 50), max(int(g), 50), max(int(r), 50))

    def _viz_text_origin_xyxy(
        self,
        text: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_h: int,
        img_w: int,
        *,
        placement: str,
        pad: int = 4,
    ) -> tuple[int, int]:
        """Fixed text placement: score above top-left, label below bottom-left (no overlap)."""
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        if placement == "score":
            tx, ty = x1, y1 - pad
        else:
            tx, ty = x1, y2 + th + pad
        tx = int(max(0, min(tx, max(0, img_w - tw))))
        ty_hi = max(th, img_h - bl - 1)
        ty = int(max(th, min(ty, ty_hi)))
        return tx, ty

    def _draw_ref_boxes_on_image(
        self, image: np.ndarray, positive_boxes: list, negative_boxes: list
    ) -> None:
        """Draw reference boxes on image in-place: green for positive, red for negative. Boxes are [x, y, w, h]."""
        try:
            import cv2

            for box in positive_boxes or []:
                if len(box) != 4:
                    continue
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(
                    image, (x, y), (x + w, y + h), (0, 255, 0), 2
                )  # Green BGR
                cv2.putText(
                    image,
                    "ref+",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            for box in negative_boxes or []:
                if len(box) != 4:
                    continue
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red BGR
                cv2.putText(
                    image,
                    "ref-",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
        except Exception as e:
            logger.warning(f"Failed to draw ref boxes: {e}")

    def _visualize_detections(
        self,
        frame: Frame,
        detections: list,
        positive_boxes: list = None,
        negative_boxes: list = None,
    ) -> Frame:
        """
        Draw detection results on the frame. Optionally draw ref boxes first (green=positive, red=negative); detection BB color per class.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            positive_boxes: Optional list of [x,y,w,h] ref boxes to draw in green
            negative_boxes: Optional list of [x,y,w,h] ref boxes to draw in red

        Returns:
            New Frame with visualizations drawn
        """
        try:
            import cv2

            image = frame.rw_bgr.image.copy()

            if positive_boxes or negative_boxes:
                self._draw_ref_boxes_on_image(
                    image, positive_boxes or [], negative_boxes or []
                )

            boxes_drawn = 0
            img_h, img_w = image.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            for det in detections:
                # Draw bounding box
                if "box" in det:
                    x1, y1, x2, y2 = (
                        int(det["box"][0]),
                        int(det["box"][1]),
                        int(det["box"][2]),
                        int(det["box"][3]),
                    )
                    raw_cls = (
                        det.get("label") or det.get("class") or det.get("class_name")
                    )
                    class_text = (
                        str(raw_cls)
                        if raw_cls is not None and str(raw_cls) != ""
                        else None
                    )
                    color_bgr = self._viz_bgr_for_class_name(class_text)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, thickness)
                    boxes_drawn += 1

                    score_text = f"{det['score']:.2f}" if "score" in det else None

                    if score_text:
                        sx, sy = self._viz_text_origin_xyxy(
                            score_text, x1, y1, x2, y2, img_h, img_w, placement="score"
                        )
                        cv2.putText(
                            image,
                            score_text,
                            (sx, sy),
                            font,
                            font_scale,
                            color_bgr,
                            thickness,
                        )
                    if class_text:
                        cx, cy = self._viz_text_origin_xyxy(
                            class_text, x1, y1, x2, y2, img_h, img_w, placement="label"
                        )
                        cv2.putText(
                            image,
                            class_text,
                            (cx, cy),
                            font,
                            font_scale,
                            color_bgr,
                            thickness,
                        )

                # Draw mask overlay (semi-transparent, same hue as class BB)
                mask = None
                if "mask_np" in det:
                    mask = det["mask_np"]
                elif "mask" in det:
                    m_val = det["mask"]
                    if isinstance(m_val, np.ndarray):
                        mask = m_val
                    elif isinstance(m_val, dict) and "polygons" in m_val:
                        # Draw polygons
                        color_mask = np.zeros_like(image)
                        raw_cls = (
                            det.get("label")
                            or det.get("class")
                            or det.get("class_name")
                        )
                        class_text = (
                            str(raw_cls)
                            if raw_cls is not None and str(raw_cls) != ""
                            else None
                        )
                        mbgr = self._viz_bgr_for_class_name(class_text)
                        for poly in m_val["polygons"]:
                            points = (
                                poly.get("points")
                                if isinstance(poly, dict)
                                else getattr(poly, "points", None)
                            )
                            if points:
                                pts = np.array(points, dtype=np.int32).reshape(
                                    (-1, 1, 2)
                                )
                                cv2.fillPoly(color_mask, [pts], mbgr)
                        image = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)
                    else:
                        try:
                            mask = np.array(m_val, dtype=np.uint8)
                        except Exception:
                            pass

                if (
                    mask is not None
                    and isinstance(mask, np.ndarray)
                    and mask.shape == image.shape[:2]
                ):
                    raw_cls = (
                        det.get("label") or det.get("class") or det.get("class_name")
                    )
                    class_text = (
                        str(raw_cls)
                        if raw_cls is not None and str(raw_cls) != ""
                        else None
                    )
                    mbgr = self._viz_bgr_for_class_name(class_text)
                    color_mask = np.zeros_like(image)
                    color_mask[mask > 0] = mbgr
                    image = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)

            # Create new Frame with visualized image (Frame.image is read-only)
            new_frame = Frame(image, frame.data, "BGR")
            if boxes_drawn > 0:
                logger.info(f"Visualized {boxes_drawn} boxes on frame")
            return new_frame

        except Exception as e:
            logger.warning(f"Failed to visualize detections: {e}")
            import traceback

            logger.warning(traceback.format_exc())

        return frame

    def _visualize_detections_on_image(
        self,
        image: np.ndarray,
        detections: list,
        positive_boxes: list = None,
        negative_boxes: list = None,
    ) -> np.ndarray:
        """
        Draw detection results on an image array. Optionally draw ref boxes first (green=positive, red=negative); detection BB color per class.

        Args:
            image: BGR image array
            detections: List of detection dictionaries
            positive_boxes: Optional list of [x,y,w,h] ref boxes to draw in green
            negative_boxes: Optional list of [x,y,w,h] ref boxes to draw in red

        Returns:
            Image array with visualizations drawn
        """
        try:
            import cv2

            image = image.copy()

            if positive_boxes or negative_boxes:
                self._draw_ref_boxes_on_image(
                    image, positive_boxes or [], negative_boxes or []
                )

            img_h, img_w = image.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            for det in detections:
                # Draw bounding box
                if "box" in det:
                    x1, y1, x2, y2 = (
                        int(det["box"][0]),
                        int(det["box"][1]),
                        int(det["box"][2]),
                        int(det["box"][3]),
                    )
                    raw_cls = (
                        det.get("label") or det.get("class") or det.get("class_name")
                    )
                    class_text = (
                        str(raw_cls)
                        if raw_cls is not None and str(raw_cls) != ""
                        else None
                    )
                    color_bgr = self._viz_bgr_for_class_name(class_text)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, thickness)
                    score_text = f"{det['score']:.2f}" if "score" in det else None
                    if score_text:
                        sx, sy = self._viz_text_origin_xyxy(
                            score_text, x1, y1, x2, y2, img_h, img_w, placement="score"
                        )
                        cv2.putText(
                            image,
                            score_text,
                            (sx, sy),
                            font,
                            font_scale,
                            color_bgr,
                            thickness,
                        )
                    if class_text:
                        cx, cy = self._viz_text_origin_xyxy(
                            class_text, x1, y1, x2, y2, img_h, img_w, placement="label"
                        )
                        cv2.putText(
                            image,
                            class_text,
                            (cx, cy),
                            font,
                            font_scale,
                            color_bgr,
                            thickness,
                        )
                # Draw mask overlay (semi-transparent, same hue as class BB)
                mask_drawn = False
                color_mask = None
                raw_cls = det.get("label") or det.get("class") or det.get("class_name")
                class_text = (
                    str(raw_cls) if raw_cls is not None and str(raw_cls) != "" else None
                )
                mbgr = self._viz_bgr_for_class_name(class_text)

                if "mask_np" in det:
                    mask = det["mask_np"]
                    if isinstance(mask, np.ndarray) and mask.shape == image.shape[:2]:
                        color_mask = np.zeros_like(image)
                        color_mask[mask > 0] = mbgr
                        mask_drawn = True
                elif "mask" in det:
                    m_val = det["mask"]
                    if isinstance(m_val, np.ndarray) and m_val.shape == image.shape[:2]:
                        color_mask = np.zeros_like(image)
                        color_mask[m_val > 0] = mbgr
                        mask_drawn = True
                    elif isinstance(m_val, dict) and "polygons" in m_val:
                        color_mask = np.zeros_like(image)
                        for poly in m_val["polygons"]:
                            points = (
                                poly.get("points")
                                if isinstance(poly, dict)
                                else getattr(poly, "points", None)
                            )
                            if points:
                                pts = np.array(points, dtype=np.int32).reshape(
                                    (-1, 1, 2)
                                )
                                cv2.fillPoly(color_mask, [pts], mbgr)
                                mask_drawn = True
                    else:
                        try:
                            mask = np.array(m_val, dtype=np.uint8)
                            if mask.shape == image.shape[:2]:
                                color_mask = np.zeros_like(image)
                                color_mask[mask > 0] = mbgr
                                mask_drawn = True
                        except Exception:
                            pass

                if mask_drawn and color_mask is not None:
                    image = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)
        except Exception as e:
            logger.warning(f"Failed to visualize detections on image: {e}")
        return image

    def _mask_to_coco_polygons(self, mask: np.ndarray) -> list:
        """
        Convert binary mask to COCO polygon format.

        COCO format: list of polygons, where each polygon is a list of coordinates
        [x1, y1, x2, y2, x3, y3, ...] representing the boundary of the mask.

        Args:
            mask: Binary mask (H, W) with values 0 or 1

        Returns:
            List of polygons in COCO format, or empty list if mask is empty
        """
        try:
            import cv2

            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            polygons = []
            for contour in contours:
                # Simplify contour to reduce points
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 3:  # Need at least 3 points for a polygon
                    # Flatten to [x1, y1, x2, y2, ...] format
                    polygon = approx.flatten().tolist()
                    polygons.append(polygon)

            return polygons

        except Exception as e:
            logger.warning(f"Failed to convert mask to COCO polygons: {e}")
            return []


if __name__ == "__main__":
    FilterSAM3Detector.run()
