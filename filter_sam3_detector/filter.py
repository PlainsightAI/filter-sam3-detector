import logging
import os
import json
import multiprocessing
import time
from typing import Optional
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# Fix CUDA multiprocessing issue - set spawn method before any CUDA operations
try:
    multiprocessing.set_start_method('spawn', force=True)  # CUDA doesn't like fork()
except RuntimeError:
    # Method already set, ignore
    pass

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

from .temporal_intervals import EMATracker, DetectionInterval, IntervalTracker
from .streaming_video_processor import StreamingVideoProcessor

__all__ = ["FilterSAM3DetectorConfig", "FilterSAM3Detector"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Image file extensions for ref_images and exemplars (first-level directory listing)
REF_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

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
except ImportError:
    box_xywh_to_cxcywh = None
    normalize_bbox = None
    logger.warning("SAM3 not available. Install from: https://github.com/facebookresearch/sam3")
except Exception:
    HAS_SAM3 = False
    box_xywh_to_cxcywh = None
    normalize_bbox = None
    logger.warning("SAM3 not available. Install from: https://github.com/facebookresearch/sam3")


class FilterSAM3DetectorConfig(FilterConfig):
    """Configuration for SAM3 object detection filter."""
    
    # FilterConfig is a dict-like class, so we define defaults in normalize_config
    pass


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

    @classmethod
    def normalize_config(cls, config: FilterConfig) -> FilterConfig:
        """
        Normalize and validate configuration parameters.
        
        This method MUST BE IDEMPOTENT - calling it multiple times should produce the same result.
        """
        # First, call parent normalize_config to handle sources, outputs, etc.
        config = super().normalize_config(config)
        config = FilterSAM3DetectorConfig(config)
        
        # Set defaults if not present
        defaults = {
            "model_id": "facebook/sam3",
            "device": "cuda",
            "text_prompt": None,  # Single prompt (backward compatible)
            "text_prompts": None,  # Multiple prompts for parallel detection
            "prompt_sets": None,  # Multi-output mode: list of {name, prompts, topic, ...}
            "exemplars_path": None,
            "exemplar_embeddings_cache": None,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "nms_threshold": 0.5,  # IoU threshold for NMS (0 = disabled, lower = more aggressive)
            "max_detections": 100,
            "output_masks": False,  # Don't save masks by default (can be large)
            "output_boxes": True,
            "output_scores": True,
            "output_label": "sam3_detections",
            "output_path": None,  # Path to save JSONL annotations
            "output_filter_name": "SAM3Detector",  # Filter name for event sink format
            # NMS (Non-Maximum Suppression) options
            "nms_enabled": True,  # Enable NMS to suppress overlapping detections
            "nms_threshold": 0.5,  # IoU threshold for NMS (higher = more boxes kept)
            "frames_output_dir": None,  # Directory to save original frames
            "annotated_frames_output_dir": None,  # Directory to save annotated frames (separate from original)
            "save_annotated_frames": False,  # Save frames with visual annotations (boxes, scores, masks)
            "visualize": False,
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
            # Reference image prompts (positive = bottom-left, negative = bottom-right)
            "ref_images": None,
            "ref_images_negative": None,
            "ref_margin": 10,
            "ref_gap": 5,
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Load from environment variables (override config values)
        env_mapping = {
            "model_id": str,
            "device": str,
            "text_prompt": str,
            "text_prompts": str,  # Comma-separated list of prompts
            "prompt_sets": str,  # JSON array of prompt set configs
            "exemplars_path": str,
            "exemplar_embeddings_cache": str,
            "confidence_threshold": float,
            "mask_threshold": float,
            "nms_threshold": float,
            "max_detections": int,
            "output_masks": bool,
            "output_boxes": bool,
            "output_scores": bool,
            "output_label": str,
            "output_path": str,
            "nms_enabled": bool,
            "nms_threshold": float,
            "frames_output_dir": str,
            "annotated_frames_output_dir": str,
            "save_annotated_frames": bool,
            "visualize": bool,
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
            # Ref images (margin/gap; ref path lists handled below)
            "ref_margin": int,
            "ref_gap": int,
        }
        
        # Special handling for FILTER_REF_IMAGES (comma-separated paths -> list)
        env_ref_images = os.getenv("FILTER_REF_IMAGES")
        if env_ref_images is not None:
            config["ref_images"] = [p.strip() for p in env_ref_images.split(",") if p.strip()] or None

        # Special handling for FILTER_REF_IMAGES_NEGATIVE (comma-separated paths -> list)
        env_ref_images_neg = os.getenv("FILTER_REF_IMAGES_NEGATIVE")
        if env_ref_images_neg is not None:
            config["ref_images_negative"] = [p.strip() for p in env_ref_images_neg.split(",") if p.strip()] or None

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
        env_annotated_frames_output_dir = os.getenv("FILTER_ANNOTATED_FRAMES_OUTPUT_DIR")
        if env_annotated_frames_output_dir is not None:
            config["annotated_frames_output_dir"] = env_annotated_frames_output_dir.strip()

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
        valid_devices = ['cuda', 'cpu', 'mps']
        device = config.get("device", "cuda")
        if isinstance(device, str):
            device = device.lower().strip()
            if device not in valid_devices:
                raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}")
            config["device"] = device

        # Validate numeric ranges
        confidence_threshold = config.get("confidence_threshold", 0.5)
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")

        mask_threshold = config.get("mask_threshold", 0.5)
        if not (0.0 <= mask_threshold <= 1.0):
            raise ValueError(f"mask_threshold must be between 0 and 1, got {mask_threshold}")

        max_detections = config.get("max_detections", 100)
        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")

        nms_threshold = config.get("nms_threshold", 0.5)
        if not (0.0 <= nms_threshold <= 1.0):
            raise ValueError(f"nms_threshold must be between 0 and 1, got {nms_threshold}")

        # Parse text_prompts from comma-separated string to list
        text_prompts = config.get("text_prompts")
        if isinstance(text_prompts, str):
            config["text_prompts"] = [p.strip() for p in text_prompts.split(",") if p.strip()]
        elif text_prompts is None:
            config["text_prompts"] = None
        elif not isinstance(text_prompts, list):
            raise ValueError(f"text_prompts must be a list or comma-separated string, got {type(text_prompts)}")

        # Parse prompt_sets from JSON string to list of dicts
        # Format: [{"name": "bowl", "prompts": ["bowl"], "topic": "main", "confidence_threshold": 0.5, "max_detections": 1}, ...]
        prompt_sets = config.get("prompt_sets")
        if isinstance(prompt_sets, str):
            try:
                config["prompt_sets"] = json.loads(prompt_sets)
            except json.JSONDecodeError as e:
                raise ValueError(f"prompt_sets must be valid JSON array, got error: {e}")
        elif prompt_sets is not None and not isinstance(prompt_sets, list):
            raise ValueError(f"prompt_sets must be a list or JSON string, got {type(prompt_sets)}")

        # Validate prompt_sets structure
        if config.get("prompt_sets"):
            for i, ps in enumerate(config["prompt_sets"]):
                if not isinstance(ps, dict):
                    raise ValueError(f"prompt_sets[{i}] must be a dict, got {type(ps)}")
                if "name" not in ps:
                    raise ValueError(f"prompt_sets[{i}] missing required 'name' field")
                if "prompts" not in ps:
                    raise ValueError(f"prompt_sets[{i}] missing required 'prompts' field")
                # Ensure prompts is a list
                if isinstance(ps["prompts"], str):
                    ps["prompts"] = [p.strip() for p in ps["prompts"].split(",") if p.strip()]

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
        self.text_prompt = config.get("text_prompt")  # Single prompt (backward compatible)
        self.text_prompts = config.get("text_prompts")  # Multiple prompts for parallel detection
        self.prompt_sets = config.get("prompt_sets")  # Multi-output mode
        self.exemplars_path = config.get("exemplars_path")
        # Reference image paths (positive = bottom-left, negative = bottom-right on composite)
        # Accept files or directories; directories are expanded to sorted list of images
        _ref_images = config.get("ref_images")
        self.ref_images = [_ref_images] if isinstance(_ref_images, (str, Path)) else (_ref_images if _ref_images else None)
        _ref_neg = config.get("ref_images_negative")
        self.ref_images_negative = [_ref_neg] if isinstance(_ref_neg, (str, Path)) else (_ref_neg if _ref_neg else None)
        self.ref_images = self._expand_ref_paths(self.ref_images)
        self.ref_images_negative = self._expand_ref_paths(self.ref_images_negative)
        self.ref_margin = config.get("ref_margin", 10)
        self.ref_gap = config.get("ref_gap", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.5)
        self.max_detections = config.get("max_detections", 100)
        self.output_masks = config.get("output_masks", True)
        self.output_boxes = config.get("output_boxes", True)
        self.output_scores = config.get("output_scores", True)
        self.output_label = config.get("output_label", "sam3_detections")
        self.output_path = config.get("output_path", None)
        self.output_filter_name = config.get("output_filter_name", "SAM3Detector")
        self.nms_enabled = config.get("nms_enabled", True)
        self.nms_threshold = config.get("nms_threshold", 0.5)
        self.frames_output_dir = config.get("frames_output_dir", None)
        self.annotated_frames_output_dir = config.get("annotated_frames_output_dir", None)
        self.save_annotated_frames = config.get("save_annotated_frames", False)
        self.visualize = config.get("visualize", False)
        
        # Initialize JSONL output file if path is provided
        self.jsonl_file = None
        if self.output_path:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = open(output_path, 'w')
            logger.info(f"Saving annotations to: {output_path}")
        
        # Initialize frames output directories if provided
        self.frames_dir = None
        self.annotated_frames_dir = None
        self.frame_counter = 0  # Counter for unique frame numbering
        self.global_detection_id = 0  # Global counter for unique detection IDs across all frames
        
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
        elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        logger.info(f"NMS enabled: {self.nms_enabled}, threshold: {self.nms_threshold}")

        # Video mode configuration
        self.enable_video_mode = config.get("enable_video_mode", False)
        self.video_detection_interval = config.get("video_detection_interval", 5)
        self.video_min_tracking_confidence = config.get("video_min_tracking_confidence", 0.3)
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

        # Optional: validate ref image paths exist (log only, do not block)
        for attr in ("ref_images", "ref_images_negative"):
            paths = getattr(self, attr, None)
            if paths:
                for p in paths:
                    if not Path(p).exists():
                        logger.warning(f"Ref image path does not exist: {p}")

        # Log multi-output mode configuration and pre-cache text embeddings
        # This is the KEY OPTIMIZATION: text prompts are static, so we encode them ONCE
        # at startup instead of on every frame. This saves significant inference time.
        self.cached_text_embeddings = {}  # prompt -> {language_features, language_mask, language_embeds}

        if self.prompt_sets:
            logger.info(f"Multi-output mode enabled with {len(self.prompt_sets)} prompt sets:")
            for ps in self.prompt_sets:
                logger.info(f"  - {ps['name']}: prompts={ps['prompts']}, topic={ps.get('topic', 'main')}")

            # Pre-cache text embeddings for all prompts in all prompt sets
            if self.model is not None:
                self._cache_text_embeddings()

        # Initialize temporal interval tracking if enabled
        self.enable_temporal_intervals = config.get("enable_temporal_intervals", False)
        self.interval_tracker: Optional[IntervalTracker] = None

        if self.enable_temporal_intervals:
            self._setup_temporal_intervals(config)

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

        # Close JSONL file if open
        if hasattr(self, 'jsonl_file') and self.jsonl_file is not None:
            try:
                self.jsonl_file.close()
                if hasattr(self, 'output_path') and self.output_path:
                    logger.info(f"Closed annotation file: {self.output_path}")
            except Exception as e:
                logger.warning(f"Error closing annotation file: {e}")
            self.jsonl_file = None

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

    def _process_temporal_intervals(self, frame: Frame, detections: list):
        """
        Process temporal interval tracking for the current frame.

        Uses IntervalTracker for EMA updates, state changes, and interval management.
        """
        # Get frame ID from metadata if available
        meta = frame.data.get('meta', {})
        frame_id = int(meta['id']) if 'id' in meta else None

        # Aggregate detections by label (max score per label)
        detected_labels = self._aggregate_temporal_detections(detections)

        # Update tracker and get state changes
        state_changes = self.interval_tracker.update(detected_labels, frame_id)

        # Add interval info to frame metadata if state changed
        if state_changes and self.temporal_emit_on_change:
            meta = frame.data.setdefault('meta', {})
            meta['temporal_intervals'] = {
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
            score = det.get('score', 1.0)
            if score < self.temporal_min_confidence:
                continue

            # Get label from detection or use default
            if self.temporal_label_field and self.temporal_label_field in det:
                label = str(det[self.temporal_label_field])
            else:
                # Use class field if available, otherwise default label
                label = det.get('class') or det.get('class_name') or self.temporal_default_label

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
            if topic == '_filter' or topic.endswith('___filter'):
                if frame and frame.data and isinstance(frame.data, dict):
                    frame_id = frame.data.get('id')
                    if frame_id is not None:
                        logger.debug(f"Extracted frame id from {topic}: {frame_id}")
                        return int(frame_id) if isinstance(frame_id, (int, float)) else frame_id
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

            # Need either text prompts or visual embeddings from exemplars
            if prompts_to_use is None and self.visual_prompt_embed is None:
                logger.warning("No text prompt(s) or exemplars configured, forwarding frame unchanged")
                output_frames[topic] = frame
                continue

            # When using ref images, a text prompt is required
            if (self.ref_images or self.ref_images_negative) and not prompts_to_use:
                logger.warning("ref_images or ref_images_negative set but no text prompt; forwarding frame unchanged")
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
                detections = []
                all_scores = []  # Track all scores for detection_confidence calculation

                has_refs = bool(self.ref_images or self.ref_images_negative)

                if has_refs and prompts_to_use:
                    # Reference-image mode: composite with pasted refs + geometric prompts
                    composite, all_norm_labels = self._build_composite_with_refs(pil_image)
                    state = self.processor.set_image(composite)
                    self.processor.reset_all_prompts(state)
                    prompt = prompts_to_use[0]
                    if prompt in self.cached_text_embeddings:
                        state = self._inject_cached_text_embedding(state, prompt)
                    else:
                        state = self.processor.set_text_prompt_no_grounding(prompt, state)
                    for (norm_box, label) in all_norm_labels:
                        state = self.processor.add_geometric_prompt(norm_box, label, state)
                    detections = self._extract_detections_from_state(
                        state, prompt, img_width, img_height, self.global_detection_id
                    )
                    self.global_detection_id += len(detections)
                    if "scores" in state:
                        all_scores.extend(
                            float(s.item() if hasattr(s, 'item') else s)
                            for s in state["scores"]
                        )
                else:
                    # Standard mode: set image once, then loop over prompts (and optionally visual exemplars)
                    state = self.processor.set_image(pil_image)

                    # Process each prompt using cached image features
                    if prompts_to_use:
                        for prompt in prompts_to_use:
                            # Use cached text embeddings if available, otherwise encode on-the-fly
                            # This reuses the cached image features from set_image()
                            if prompt in self.cached_text_embeddings:
                                prompt_state = self._inject_cached_text_embedding(state, prompt)
                            else:
                                prompt_state = self.processor.set_text_prompt_no_grounding(prompt, state)
                            prompt_state = self.processor.forward_grounding(prompt_state)

                            # Extract detections for this prompt (use global ID counter for uniqueness)
                            prompt_detections = self._extract_detections_from_state(
                                prompt_state, prompt, img_width, img_height, self.global_detection_id
                            )
                            detections.extend(prompt_detections)
                            self.global_detection_id += len(prompt_detections)

                            # Track scores
                            if "scores" in prompt_state:
                                all_scores.extend(
                                    float(s.item() if hasattr(s, 'item') else s)
                                    for s in prompt_state["scores"]
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
                            target_device = state["backbone_out"]["vision_features"].device
                            for key, value in dummy_text_outputs.items():
                                if hasattr(value, 'to'):
                                    dummy_text_outputs[key] = value.to(target_device)
                            state["backbone_out"].update(dummy_text_outputs)

                        # Initialize geometric prompt if not present
                        if "geometric_prompt" not in state:
                            state["geometric_prompt"] = self.model._get_dummy_prompt()

                        # Run grounding with visual prompt embeddings
                        visual_state = self._forward_grounding_with_visual_prompt(state)

                        # Extract detections for visual prompt (use global ID counter)
                        visual_detections = self._extract_detections_from_state(
                            visual_state, "visual", img_width, img_height, self.global_detection_id
                        )
                        detections.extend(visual_detections)
                        self.global_detection_id += len(visual_detections)

                        # Track scores
                        if "scores" in visual_state:
                            all_scores.extend(
                                float(s.item() if hasattr(s, 'item') else s)
                                for s in visual_state["scores"]
                            )

                # Set scores variable for detection_confidence calculation
                scores = all_scores if all_scores else None

                # Calculate detection_confidence (average or max score)
                detection_confidence = None
                if detections and scores is not None and len(scores) > 0:
                    # Use the maximum confidence score
                    max_score = max(float(s.item() if hasattr(s, 'item') else s) for s in scores[:len(detections)])
                    detection_confidence = float(max_score)
                elif detections and any('score' in d for d in detections):
                    # Fallback: use max score from detections
                    max_score = max(d.get('score', 0.0) for d in detections if 'score' in d)
                    detection_confidence = float(max_score)

                # Store results in frame metadata
                frame_meta = frame.data.setdefault('meta', {})
                frame_meta[self.output_label] = detections

                # Add detection_confidence to meta
                if detection_confidence is not None:
                    frame_meta['detection_confidence'] = detection_confidence

                # Build protege-compatible output format
                # This allows SAM3 to work with downstream filters like sweetgreen aggregator
                self._add_protege_compatible_output(frame_meta, detections)

                # Process temporal intervals if enabled
                if self.enable_temporal_intervals:
                    self._process_temporal_intervals(frame, detections)

                # Get frame metadata for JSONL and filename
                frame_meta = frame.data.get('meta', {})

                # Get timestamp from frame (OpenFilter provides this)
                # Try multiple sources: frame.timestamp, meta.ts, meta.timestamp, data.timestamp
                frame_ts = (
                    getattr(frame, 'timestamp', None)
                    or frame_meta.get('ts', None)
                    or frame_meta.get('timestamp', None)
                    or frame.data.get('timestamp', None)
                )

                # Get frame ID - priority: _filter topic > meta['id'] > frame_counter
                # The _filter topic (TI-130) is the idiomatic way to get frame IDs in openfilter
                frame_id_num = filter_frame_id if filter_frame_id is not None else frame_meta.get('id', None)

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
                filename_base = f"frame_{frame_id_str}_ts{timestamp_str}_count{frame_counter:06d}"
                frame_filename_str = f"{filename_base}.jpg"
                
                # Generate full paths for original and annotated frames
                frame_filename = None
                annotated_frame_filename = None
                
                if self.frames_dir is not None:
                    frame_filename = self.frames_dir / frame_filename_str
                
                if self.annotated_frames_dir is not None:
                    annotated_frame_filename = self.annotated_frames_dir / frame_filename_str


                # Save frames if output directories are configured
                try:
                    import cv2
                    # Get original image from frame
                    image_bgr_original = frame.rw_bgr.image.copy()
                    
                    # Save original frame (always save if frames_output_dir is configured)
                    if self.frames_dir is not None and frame_filename:
                        cv2.imwrite(str(frame_filename), image_bgr_original)
                    
                    # Save annotated frame (if annotated_frames_dir is configured and there are detections)
                    if self.annotated_frames_dir is not None and annotated_frame_filename and detections:
                        # Create annotated version
                        image_bgr_annotated = image_bgr_original.copy()
                        image_bgr_annotated = self._visualize_detections_on_image(image_bgr_annotated, detections)
                        cv2.imwrite(str(annotated_frame_filename), image_bgr_annotated)
                    
                except Exception as e:
                    logger.warning(f"Failed to save frame: {e}")

                # Save to JSONL file if output_path is configured (save ALL frames, even without detections)
                if hasattr(self, 'jsonl_file') and self.jsonl_file is not None:
                    try:
                        # Build protege-compatible meta for JSONL output
                        jsonl_meta = {}
                        self._add_protege_compatible_output(jsonl_meta, detections)

                        # Include frame_id in meta (from _filter topic or VideoIn's meta['id'])
                        output_frame_id = frame_id_num if frame_id_num is not None else frame_counter
                        jsonl_meta['frame_id'] = output_frame_id

                        # Event sink format: {'filter_name': ..., 'topic': ..., 'data': {'id': ..., 'meta': ...}}
                        # This matches what filter-event-sink outputs - frame id is merged into data
                        event_record = {
                            "filter_name": self.output_filter_name,
                            "topic": "main",
                            "data": {
                                "id": output_frame_id,
                                "meta": jsonl_meta
                            }
                        }
                        self.jsonl_file.write(json.dumps(event_record) + '\n')
                        self.jsonl_file.flush()  # Ensure immediate write
                    except Exception as e:
                        logger.warning(f"Failed to save annotation to JSONL: {e}")

                # Optional visualization (for output frame)
                if self.visualize and detections:
                    frame = self._visualize_detections(frame, detections)

            except Exception as e:
                logger.error(f"Error processing frame from {topic}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            output_frames[topic] = frame

        return output_frames

    def _process_multi_output(self, frame: Frame, filter_frame_id: Optional[int]) -> dict[str, Frame]:
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
        import copy

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

        # Get frame metadata for ID tracking
        frame_meta_orig = frame.data.get('meta', {})
        frame_id_num = filter_frame_id if filter_frame_id is not None else frame_meta_orig.get('id', None)
        frame_counter = self.frame_counter
        self.frame_counter += 1

        # Process each prompt set
        for prompt_set in self.prompt_sets:
            ps_name = prompt_set['name']
            ps_prompts = prompt_set['prompts']
            ps_topic = prompt_set.get('topic', 'main')
            ps_threshold = prompt_set.get('confidence_threshold', self.confidence_threshold)
            ps_max_detections = prompt_set.get('max_detections', self.max_detections)
            ps_filter_name = prompt_set.get('filter_name', f"SAM3_{ps_name}")
            # Optional label aliases: maps prompt -> output label (e.g., "printed order ticket" -> "chit")
            ps_label_aliases = prompt_set.get('label_aliases', {})

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
                    prompt_state, output_label, img_width, img_height, self.global_detection_id,
                    confidence_threshold=ps_threshold,
                    max_detections=ps_max_detections
                )
                ps_detections.extend(prompt_detections)
                self.global_detection_id += len(prompt_detections)

            # Apply max_detections limit to entire prompt set (sort by score, take top N)
            if len(ps_detections) > ps_max_detections:
                ps_detections.sort(key=lambda d: d.get('score', 0), reverse=True)
                ps_detections = ps_detections[:ps_max_detections]

            # Create output frame for this topic (deep copy to avoid shared state)
            output_frame = copy.deepcopy(frame)
            output_meta = output_frame.data.setdefault('meta', {})

            # Store detections in frame metadata
            output_meta[self.output_label] = ps_detections

            # Add protege-compatible output
            self._add_protege_compatible_output(output_meta, ps_detections)

            # Also add detections at top-level for aggregator compatibility
            # (aggregator looks for frame.data['detections'] not frame.data['meta']['detections'])
            output_frame.data['detections'] = output_meta.get('detections', [])

            # Add to output frames with the prompt set's topic
            output_frames[ps_topic] = output_frame

            # Write to JSONL if configured (one record per prompt set per frame)
            if hasattr(self, 'jsonl_file') and self.jsonl_file is not None:
                try:
                    jsonl_meta = {}
                    self._add_protege_compatible_output(jsonl_meta, ps_detections)
                    output_frame_id = frame_id_num if frame_id_num is not None else frame_counter
                    jsonl_meta['frame_id'] = output_frame_id

                    event_record = {
                        "filter_name": ps_filter_name,
                        "topic": ps_topic,
                        "data": {
                            "id": output_frame_id,
                            "meta": jsonl_meta
                        }
                    }
                    self.jsonl_file.write(json.dumps(event_record) + '\n')
                    self.jsonl_file.flush()
                except Exception as e:
                    logger.warning(f"Failed to save JSONL for prompt set {ps_name}: {e}")

        return output_frames

    def _add_protege_compatible_output(self, frame_meta: dict, detections: list) -> None:
        """
        Add protege-compatible output format to frame metadata.

        This enables SAM3 to work with downstream filters that expect protege-model format,
        such as the sweetgreen subject data aggregator and the golden truth comparison engine
        (PR 355 frame-level format).

        Output format (per detection):
        - meta.detections: list of {"label": str, "confidence": float, "bbox": {x, y, width, height}}
        - meta.classification: {"classes": [...], "confidences": [...], "architecture": str}

        Each detection has ONE box with ONE confidence score.
        Uses "label" and "bbox" format for PR 355 comparator compatibility.

        Args:
            frame_meta: Frame metadata dict to update
            detections: List of SAM3 detection dicts with box, score, class fields
        """
        if not detections:
            # Even with no detections, add empty structures for consistency
            frame_meta['detections'] = []
            frame_meta['classification'] = {
                'classes': [],
                'confidences': [],
                'architecture': 'sam3',
            }
            return

        # Build detections list - one per object with individual confidence
        # Uses "label" and "bbox" format for PR 355 comparator compatibility
        output_detections = []
        class_max_scores: dict[str, float] = {}  # Track max score per class for classification block

        for det in detections:
            label = det.get('class') or det.get('class_name') or 'object'
            box = det.get('box')  # [x1, y1, x2, y2] pixel coordinates
            score = det.get('score', 0.0)

            if box is None:
                continue

            # Track max score per class for classification block
            if label not in class_max_scores:
                class_max_scores[label] = 0.0
            class_max_scores[label] = max(class_max_scores[label], score)

            # Convert [x1, y1, x2, y2] to {x, y, width, height} format
            x1, y1, x2, y2 = box
            bbox = {
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1,
            }

            # Include both PR 355 format (label, confidence, bbox) AND
            # aggregator-compatible format (class, rois) in same detection
            output_detections.append({
                'label': label,
                'confidence': score,
                'bbox': bbox,
                # Aggregator-compatible fields:
                'class': label,
                'rois': [[int(x1), int(y1), int(x2), int(y2)]],
            })

        frame_meta['detections'] = output_detections

        # Build classification block (classes with their confidence scores)
        # Sort by score descending for consistency
        sorted_classes = sorted(class_max_scores.items(), key=lambda x: x[1], reverse=True)
        classes = [cls for cls, _ in sorted_classes]
        confidences = [score for _, score in sorted_classes]

        frame_meta['classification'] = {
            'classes': classes,
            'confidences': confidences,
            'architecture': 'sam3',
        }

    def _extract_detections_from_state(
        self, state: dict, class_name: str, img_width: int, img_height: int, id_offset: int = 0,
        confidence_threshold: Optional[float] = None, max_detections: Optional[int] = None
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
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        max_dets = max_detections if max_detections is not None else self.max_detections

        if "boxes" not in state or "scores" not in state:
            return detections

        boxes = state["boxes"]
        scores = state["scores"]
        masks = state.get("masks", None)

        # Apply NMS if enabled to suppress overlapping detections
        if self.nms_enabled and len(boxes) > 0:
            boxes, scores, masks = self._apply_nms(boxes, scores, masks)

        num_detections = min(len(boxes), max_dets)

        actual_id = 0  # Track actual number of detections added
        for i in range(len(boxes)):
            # Apply confidence threshold filter
            score = scores[i]
            if hasattr(score, 'item'):
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

            if self.output_boxes:
                box = boxes[i]
                if hasattr(box, 'tolist'):
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
                detection['box'] = [int(x1), int(y1), int(x2), int(y2)]

                # Convert to COCO bbox format [x, y, width, height]
                coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                detection['bbox'] = coco_bbox  # COCO format

            if self.output_scores:
                score = scores[i]
                if hasattr(score, 'item'):
                    score = score.item()
                detection['score'] = float(score)
                # Also add 'confidence' for protege compatibility
                detection['confidence'] = float(score)

            if self.output_masks and masks is not None and i < len(masks):
                mask = masks[i]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if hasattr(mask, 'squeeze'):
                    mask = mask.squeeze()

                # Convert to binary mask
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Convert mask to COCO format (polygons)
                segmentation = self._mask_to_coco_polygons(binary_mask)

                if segmentation:
                    detection['segmentation'] = segmentation

                    # Calculate area (number of pixels in mask)
                    area = int(np.sum(binary_mask))
                    detection['area'] = area

                    # Category ID (1 = object, since we don't have specific categories)
                    detection['category_id'] = 1

                    # iscrowd (0 = single object, 1 = crowd)
                    detection['iscrowd'] = 0

            if detection:
                detection['id'] = detection_id

                # Add class name from the prompt
                detection['class'] = class_name
                detection['class_name'] = class_name
                detection['category_name'] = class_name

                # Add rois with pixel coordinates [x1, y1, x2, y2]
                # This matches protege detection model output format for aggregator compatibility
                if 'box' in detection:
                    x1, y1, x2, y2 = detection['box']
                    detection['rois'] = [[int(x1), int(y1), int(x2), int(y2)]]

                # Add category_id if not already set (for COCO compatibility)
                if 'category_id' not in detection:
                    detection['category_id'] = 1

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
                for prompt in ps.get('prompts', []):
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

        logger.info(f"Pre-caching text embeddings for {len(all_prompts)} unique prompts...")

        with torch.no_grad():
            for prompt in all_prompts:
                try:
                    # Encode the text prompt using the model's backbone
                    text_outputs = self.model.backbone.forward_text([prompt], device=str(self.device))

                    # Store the relevant tensors (detach to avoid memory leaks)
                    self.cached_text_embeddings[prompt] = {
                        'language_features': text_outputs.get('language_features'),
                        'language_mask': text_outputs.get('language_mask'),
                        'language_embeds': text_outputs.get('language_embeds'),
                    }
                    logger.debug(f"Cached text embedding for prompt: '{prompt}'")

                except Exception as e:
                    logger.warning(f"Failed to cache text embedding for '{prompt}': {e}")

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
            logger.warning(f"Text embedding not cached for '{prompt}', encoding on-the-fly")
            return self.processor.set_text_prompt_no_grounding(prompt, state)

        cached = self.cached_text_embeddings[prompt]

        # Inject cached language features into backbone_out
        if cached.get('language_features') is not None:
            state['backbone_out']['language_features'] = cached['language_features']
        if cached.get('language_mask') is not None:
            state['backbone_out']['language_mask'] = cached['language_mask']
        if cached.get('language_embeds') is not None:
            state['backbone_out']['language_embeds'] = cached['language_embeds']

        # Initialize geometric prompt if not present (same as set_text_prompt_no_grounding)
        if 'geometric_prompt' not in state:
            state['geometric_prompt'] = self.model._get_dummy_prompt()

        return state

    def _load_model(self):
        """
        Load the SAM3 model from HuggingFace.
        """
        if not HAS_SAM3:
            logger.error("SAM3 not available. Install from: https://github.com/facebookresearch/sam3")
            return

        try:
            logger.info(f"Loading SAM3 model on device: {self.device}")

            # Find BPE path - check vendorized sam3 first, then installed package
            bpe_path = None
            # Try vendorized sam3 (in project root/sam3/assets/)
            vendorized_bpe = Path(__file__).parent.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if vendorized_bpe.exists():
                bpe_path = str(vendorized_bpe)
                logger.info(f"Using vendorized BPE file: {bpe_path}")
            else:
                # Try vendorized sam3/sam3/assets/ (alternative location)
                vendorized_bpe2 = Path(__file__).parent.parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
                if vendorized_bpe2.exists():
                    bpe_path = str(vendorized_bpe2)
                    logger.info(f"Using vendorized BPE file: {bpe_path}")
                else:
                    # Try to find in installed package
                    try:
                        import sam3
                        sam3_path = Path(sam3.__file__).parent
                        installed_bpe = sam3_path / "assets" / "bpe_simple_vocab_16e6.txt.gz"
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
                confidence_threshold=self.confidence_threshold
            )

            logger.info(f"SAM3 model loaded successfully on {self.device}")

        except Exception as e:
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
            logger.error("SAM3 not available. Install from: https://github.com/facebookresearch/sam3")
            return

        try:
            logger.info(f"Loading SAM3 VIDEO model on device: {self.device}")
            logger.info(f"  detection_interval={self.video_detection_interval}")
            logger.info(f"  min_tracking_confidence={self.video_min_tracking_confidence}")

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
                    for prompt in ps.get('prompts', []):
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
        vendorized_bpe = Path(__file__).parent.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if vendorized_bpe.exists():
            return str(vendorized_bpe)

        # Try vendorized sam3/sam3/assets/ (alternative location)
        vendorized_bpe2 = Path(__file__).parent.parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
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
                    f for f in exemplars_path.iterdir()
                    if f.suffix.lower() in REF_IMAGE_EXTENSIONS
                ]

            if not image_files:
                logger.warning(f"No image files found in {self.exemplars_path}")
                return

            logger.info(f"Loading {len(image_files)} exemplar images from {self.exemplars_path}")

            # Encode each exemplar image and collect features
            all_embeddings = []

            for img_path in image_files:
                try:
                    # Load image
                    pil_image = Image.open(img_path).convert('RGB')

                    # Encode with SAM3 backbone
                    with torch.no_grad():
                        # Use the processor's transform
                        image_tensor = self.processor.transform(
                            torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).to(self.device)
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
                        elif "backbone_fpn" in backbone_out and backbone_out["backbone_fpn"]:
                            # Use direct backbone_fpn features (SAM3 format)
                            feats = backbone_out["backbone_fpn"][-1]
                        elif "vision_features" in backbone_out:
                            # Fallback to vision_features
                            feats = backbone_out["vision_features"]
                        else:
                            raise ValueError(f"Cannot extract features from backbone output. Keys: {backbone_out.keys()}")

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

            logger.info(f"Created visual prompt with {num_exemplars} exemplar tokens (shape: {self.visual_prompt_embed.shape})")

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
                    f for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in REF_IMAGE_EXTENSIONS
                )
                if not files:
                    logger.warning(f"No image files found in ref directory: {p}")
                result.extend(files)
            else:
                result.append(p)
        return result if result else None

    def _build_composite_with_refs(self, pil_image: Image.Image):
        """
        Paste reference images onto the frame (positive bottom-left, negative bottom-right)
        and return the composite image plus normalized box labels for geometric prompts.

        Returns:
            (composite_pil, all_norm_labels) where all_norm_labels is a list of
            (norm_box, label) with norm_box as [cx, cy, w, h] normalized in [0, 1].
        """
        ref_paths_pos = self.ref_images or []
        ref_paths_neg = self.ref_images_negative or []
        if not ref_paths_pos and not ref_paths_neg:
            return pil_image, []

        composite = pil_image.copy()
        width, height = composite.size
        margin = self.ref_margin
        gap = self.ref_gap
        all_norm_labels = []

        def paste_refs(ref_paths: list, is_positive: bool):
            y_next = height - margin
            paste_x_base = margin if is_positive else None
            for ref_path in ref_paths:
                ref_path = Path(ref_path)
                try:
                    ref_img = Image.open(ref_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to open ref image {ref_path}: {e}")
                    continue
                sw, sh = ref_img.size
                paste_y = y_next - sh
                if is_positive:
                    paste_x_cur = paste_x_base
                else:
                    paste_x_cur = width - margin - sw
                composite.paste(ref_img, (int(paste_x_cur), int(paste_y)))
                box_xywh = [float(paste_x_cur), float(paste_y), float(sw), float(sh)]
                box_t = torch.tensor(box_xywh, device=self.device, dtype=torch.float32).view(1, 4)
                box_cxcywh = box_xywh_to_cxcywh(box_t)
                norm = normalize_bbox(box_cxcywh, width, height)
                norm_list = norm.flatten().tolist()
                all_norm_labels.append((norm_list, is_positive))
                y_next = paste_y - gap

        paste_refs(ref_paths_pos, is_positive=True)
        paste_refs(ref_paths_neg, is_positive=False)

        return composite, all_norm_labels

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
        target_device = backbone_out.get("vision_features", self.visual_prompt_embed).device
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

    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, masks: torch.Tensor = None) -> tuple:
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
            logger.debug(f"NMS: {num_before} -> {num_after} detections (IoU threshold={self.nms_threshold})")

        return filtered_boxes, filtered_scores, filtered_masks

    def _visualize_detections(self, frame: Frame, detections: list) -> Frame:
        """
        Draw detection results on the frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            New Frame with visualizations drawn
        """
        try:
            import cv2
            image = frame.rw_bgr.image.copy()

            boxes_drawn = 0
            for i, det in enumerate(detections):
                # Draw bounding box
                if 'box' in det:
                    x1, y1, x2, y2 = det['box']
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    boxes_drawn += 1

                    # Draw label with class name and score
                    label_parts = []
                    if 'label' in det:
                        label_parts.append(det['label'])
                    if 'score' in det:
                        label_parts.append(f"{det['score']:.2f}")
                    label = " ".join(label_parts) if label_parts else ""
                    if label:
                        cv2.putText(image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw mask overlay (semi-transparent)
                if 'mask' in det:
                    mask = np.array(det['mask'], dtype=np.uint8)
                    if mask.shape == image.shape[:2]:
                        color_mask = np.zeros_like(image)
                        color_mask[mask > 0] = [0, 255, 0]  # Green mask
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

    def _visualize_detections_on_image(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw detection results on an image array (not Frame).
        
        Args:
            image: BGR image array
            detections: List of detection dictionaries
            
        Returns:
            Image array with visualizations drawn
        """
        try:
            import cv2
            image = image.copy()
            
            for i, det in enumerate(detections):
                # Draw bounding box
                if 'box' in det:
                    x1, y1, x2, y2 = det['box']
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw score
                    if 'score' in det:
                        score = det['score']
                        label = f"{score:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw mask overlay (semi-transparent)
                if 'mask' in det:
                    mask = np.array(det['mask'], dtype=np.uint8)
                    if mask.shape == image.shape[:2]:
                        color_mask = np.zeros_like(image)
                        color_mask[mask > 0] = [0, 255, 0]  # Green mask
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
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
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
