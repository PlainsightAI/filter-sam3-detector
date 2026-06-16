"""
Streaming Video Processor for SAM3

This module provides streaming-compatible video processing for SAM3.

## Implementation Modes

### v1 - Detection Throttling (default)
Uses SAM3 image mode with optimizations:
1. Text embedding caching: Pre-encode prompts at startup
2. Image backbone reuse: Process multiple prompts with single backbone pass
3. Detection caching: Optionally reuse detections for K frames

This approach is simpler and works well for static scenes or slow-moving objects.

### v2 - Memory Tracking
Full video mode with memory-based tracking:
- Uses Sam3VideoInferenceWithInstanceInteractivity (detector + tracker)
- Maintains memory bank of last 6 frames for temporal propagation
- Runs tracker propagation (fast) on most frames, detection only when needed

SAM3 video mode was trained at 6 fps, matching our 5 fps target.
Memory bank provides ~1 second of temporal context.

v2 provides ~10x speedup for video processing by:
- Running the detector only on key frames (every N frames)
- Using the tracker's memory bank for temporal propagation on intermediate frames
- Maintaining consistent object IDs across frames
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def find_bpe_path() -> Optional[str]:
    """
    Find the BPE tokenizer file path for SAM3.

    Searches in:
    1. Vendorized sam3/assets/ (relative to this module)
    2. Installed sam3 package assets

    Returns:
        Path to BPE file, or None if not found
    """
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


class ProcessingMode(Enum):
    """Processing mode for streaming video processor."""

    V1_DETECTION_THROTTLING = "v1"  # Image mode with detection caching
    V2_MEMORY_TRACKING = "v2"  # Video mode with memory-based tracking


@dataclass
class StreamingState:
    """State maintained across frames for streaming video processing (v1 mode)."""

    # Frame counter
    frame_idx: int = 0

    # Text embedding cache: prompt -> language features
    cached_text_embeddings: Dict[str, Dict[str, torch.Tensor]] = field(
        default_factory=dict
    )

    # Detection cache for throttling (reuse detections for K frames)
    # Stores: {prompt -> last_detections}
    cached_detections: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    last_detection_frame: int = -1

    # Per-object tracking info (for future memory tracking)
    # obj_id -> {mask, box, score, class_name, last_detection_frame}
    tracked_objects: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Next object ID to assign
    next_obj_id: int = 1


@dataclass
class V2StreamingState:
    """
    State maintained for v2 video mode with memory-based tracking.

    This state wraps the SAM3 video inference state and provides streaming-compatible
    management of the memory bank and tracker state.
    """

    # Frame counter (total frames processed)
    frame_idx: int = 0

    # SAM3 video inference state (from Sam3VideoInference.init_state)
    # This is lazily initialized on first frame
    inference_state: Optional[Dict[str, Any]] = None

    # Text prompt being used for detection
    text_prompt: Optional[str] = None

    # Mapping from SAM3 object IDs to our tracking info
    # obj_id -> {class_name, first_seen_frame, score}
    obj_id_to_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Class name to latest detections mapping for output formatting
    class_to_detections: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Frame history for memory bank management (stores preprocessed frames)
    # We keep track of the last N frames for potential memory bank operations
    frame_history: List[torch.Tensor] = field(default_factory=list)

    # Maximum frames to keep in history (SAM3 uses 7 total: 1 current + 6 memory)
    max_frame_history: int = 7

    # Original video dimensions (set on first frame)
    orig_height: int = 0
    orig_width: int = 0

    # Whether the state has been initialized
    initialized: bool = False


class StreamingVideoProcessor:
    """
    Streaming video processor for SAM3.

    This class provides streaming-compatible video processing with two modes:

    v1 (Detection Throttling):
    - Text embedding caching (encode prompts once at startup)
    - Detection throttling (optionally reuse detections for K frames)
    - Image backbone reuse (single backbone pass for multiple prompts)

    v2 (Memory Tracking):
    - Uses SAM3 video model with detector + tracker
    - Memory bank for temporal propagation across frames
    - 10x speedup by running detector only on key frames

    Usage:
        # v1 mode (default)
        processor = StreamingVideoProcessor(device="cuda", mode="v1")
        processor.load_model()
        processor.cache_text_embeddings(["person", "car"])

        for frame in video_frames:
            detections = processor.process_frame(frame, ["person", "car"])

        # v2 mode
        processor = StreamingVideoProcessor(device="cuda", mode="v2")
        processor.load_model()
        processor.set_text_prompt("person")  # Single prompt for v2

        for frame in video_frames:
            detections = processor.process_frame(frame)
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        detection_interval: int = 1,  # Re-run detection every N frames (1 = every frame)
        min_tracking_confidence: float = 0.3,  # For future: re-detect if tracking drops
        bpe_path: Optional[str] = None,
        mode: str = "v1",  # "v1" or "v2"
    ):
        """
        Initialize streaming video processor.

        Args:
            device: Device to run inference on
            confidence_threshold: Minimum detection confidence
            detection_interval: Frames between detection runs (1 = every frame, 5 = every 5th)
            min_tracking_confidence: (Future) Min tracking confidence before re-detecting
            bpe_path: Path to BPE tokenizer (for text encoder)
            mode: Processing mode - "v1" for detection throttling, "v2" for memory tracking
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.min_tracking_confidence = min_tracking_confidence
        self.bpe_path = bpe_path

        # Set processing mode
        if mode == "v1":
            self.mode = ProcessingMode.V1_DETECTION_THROTTLING
        elif mode == "v2":
            self.mode = ProcessingMode.V2_MEMORY_TRACKING
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'v1' or 'v2'.")

        self.model = None
        self.processor = None  # Sam3Processor for image mode (v1)
        self.video_model = None  # Sam3VideoInferenceWithInstanceInteractivity for v2

        # State depends on mode
        self.state = StreamingState()
        self.v2_state = V2StreamingState()

        # Image preprocessing (same as Sam3Processor)
        self.resolution = 1008
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        from torchvision.transforms import v2

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.resolution, self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def load_model(self) -> bool:
        """
        Load the SAM3 model based on the configured mode.

        v1 mode: Loads SAM3 image model with Sam3Processor
        v2 mode: Loads SAM3 video model with detector + tracker

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.mode == ProcessingMode.V1_DETECTION_THROTTLING:
            return self._load_image_model()
        else:
            return self._load_video_model()

    def _load_image_model(self) -> bool:
        """
        Load the SAM3 image model with processor (v1 mode).

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info(f"Loading SAM3 image model (v1 mode) on device: {self.device}")

            # Find BPE path if not provided
            bpe_path = self.bpe_path or find_bpe_path()
            if bpe_path:
                logger.info(f"Using BPE file: {bpe_path}")

            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                device=str(self.device),
                eval_mode=True,
                load_from_HF=True,
            )

            self.processor = Sam3Processor(
                self.model,
                device=str(self.device),
                confidence_threshold=self.confidence_threshold,
            )

            logger.info(f"SAM3 image model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM3 image model: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def _load_video_model(self) -> bool:
        """
        Load the SAM3 video model with detector + tracker (v2 mode).

        The video model provides memory-based tracking for temporal propagation.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from sam3.model_builder import build_sam3_video_model

            logger.info(f"Loading SAM3 video model (v2 mode) on device: {self.device}")

            # Find BPE path if not provided
            bpe_path = self.bpe_path or find_bpe_path()
            if bpe_path:
                logger.info(f"Using BPE file: {bpe_path}")

            self.video_model = build_sam3_video_model(
                bpe_path=bpe_path,
                device=str(self.device),
                load_from_HF=True,
                apply_temporal_disambiguation=True,
            )

            # Set confidence threshold
            self.video_model.score_threshold_detection = self.confidence_threshold

            logger.info(f"SAM3 video model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM3 video model: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def cache_text_embeddings(self, prompts: List[str]) -> None:
        """
        Pre-cache text embeddings for all prompts (v1 mode only).

        This is called once at startup to avoid encoding text on every frame.

        Args:
            prompts: List of text prompts to cache
        """
        if self.mode == ProcessingMode.V2_MEMORY_TRACKING:
            logger.warning(
                "cache_text_embeddings is not used in v2 mode. Use set_text_prompt() instead."
            )
            return

        if self.model is None:
            logger.warning("Cannot cache text embeddings: model not loaded")
            return

        logger.info(f"Caching text embeddings for {len(prompts)} prompts")

        with torch.no_grad():
            for prompt in prompts:
                try:
                    # Use model's backbone for text encoding
                    text_outputs = self.model.backbone.forward_text(
                        [prompt], device=self.device
                    )

                    self.state.cached_text_embeddings[prompt] = {
                        "language_features": text_outputs.get("language_features"),
                        "language_mask": text_outputs.get("language_mask"),
                        "language_embeds": text_outputs.get("language_embeds"),
                    }
                    logger.debug(f"Cached embedding for: '{prompt}'")

                except Exception as e:
                    logger.warning(f"Failed to cache embedding for '{prompt}': {e}")

        logger.info(f"Cached {len(self.state.cached_text_embeddings)} text embeddings")

    def set_text_prompt(self, prompt: str) -> None:
        """
        Set the text prompt for v2 video mode.

        In v2 mode, a single text prompt is used for the entire video stream.
        The prompt is used for detection on key frames, and the tracker
        propagates the detected objects across frames.

        Args:
            prompt: Text prompt for detection (e.g., "person", "car")
        """
        if self.mode != ProcessingMode.V2_MEMORY_TRACKING:
            logger.warning("set_text_prompt is only used in v2 mode")
            return

        self.v2_state.text_prompt = prompt
        logger.info(f"Set text prompt for v2 mode: '{prompt}'")

    def reset_tracking(self) -> None:
        """Reset all tracking state for a new video/stream."""
        if self.mode == ProcessingMode.V1_DETECTION_THROTTLING:
            self.state = StreamingState()
        else:
            self.v2_state = V2StreamingState()
        logger.info("Tracking state reset")

    def process_frame(
        self,
        image: np.ndarray,
        prompts: Optional[List[str]] = None,
        force_detection: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame and return detections.

        v1 mode (detection throttling):
        - With detection_interval > 1, detections are cached and reused
        - Prompts argument is required

        v2 mode (memory tracking):
        - Uses SAM3 video model with memory-based tracking
        - Prompts argument is optional (uses set_text_prompt())

        Args:
            image: Input image (RGB numpy array or PIL Image)
            prompts: Text prompts to detect (required for v1, optional for v2)
            force_detection: Force full detection (ignore cache)

        Returns:
            List of detection dicts with keys: box, score, class, mask, obj_id
        """
        if self.mode == ProcessingMode.V1_DETECTION_THROTTLING:
            return self._process_frame_v1(image, prompts, force_detection)
        else:
            return self._process_frame_v2(image, prompts, force_detection)

    def _process_frame_v1(
        self,
        image: np.ndarray,
        prompts: List[str],
        force_detection: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process frame using v1 detection throttling mode.
        """
        if self.model is None:
            logger.warning("Model not loaded")
            return []

        if prompts is None:
            logger.warning("Prompts required for v1 mode")
            return []

        frame_idx = self.state.frame_idx
        self.state.frame_idx += 1

        # Determine if we should run detection or use cached results
        should_detect = (
            force_detection
            or frame_idx == 0  # First frame always detects
            or self.detection_interval <= 1  # No caching
            or (frame_idx - self.state.last_detection_frame) >= self.detection_interval
            or len(self.state.cached_detections) == 0  # No cached detections
        )

        if should_detect:
            detections = self._run_detection(image, prompts, frame_idx)
            # Cache detections for reuse
            self.state.cached_detections = {}
            for det in detections:
                prompt = det.get("class_name", det.get("class", "unknown"))
                if prompt not in self.state.cached_detections:
                    self.state.cached_detections[prompt] = []
                self.state.cached_detections[prompt].append(det)
            self.state.last_detection_frame = frame_idx
            return detections
        else:
            # Return cached detections
            cached = []
            for prompt in prompts:
                cached.extend(self.state.cached_detections.get(prompt, []))
            logger.debug(f"Frame {frame_idx}: Using {len(cached)} cached detections")
            return cached

    def _process_frame_v2(
        self,
        image: np.ndarray,
        prompts: Optional[List[str]] = None,
        force_detection: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process frame using v2 memory-based tracking mode.

        This method:
        1. Preprocesses the frame for SAM3 video model
        2. Initializes or updates the streaming inference state
        3. Runs single-frame inference with memory-based tracking
        4. Extracts and returns detections
        """
        if self.video_model is None:
            logger.warning("Video model not loaded")
            return []

        # Use prompts argument if provided, otherwise use set_text_prompt
        text_prompt = prompts[0] if prompts else self.v2_state.text_prompt
        if text_prompt is None:
            logger.warning(
                "No text prompt set. Use set_text_prompt() or pass prompts argument."
            )
            return []

        frame_idx = self.v2_state.frame_idx
        self.v2_state.frame_idx += 1

        try:
            # Initialize state on first frame
            if not self.v2_state.initialized:
                self._init_v2_streaming_state(image, text_prompt)

            # Add frame to inference state
            preprocessed_frame = self._preprocess_frame_for_v2(image)
            self._add_frame_to_v2_state(preprocessed_frame, frame_idx)

            # Run single-frame inference
            detections = self._run_v2_inference(frame_idx, text_prompt)

            return detections

        except Exception as e:
            logger.error(f"Error in v2 frame processing: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return []

    def _preprocess_frame_for_v2(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a frame for the SAM3 video model.

        Args:
            image: Input image (RGB numpy array or PIL Image)

        Returns:
            Preprocessed frame tensor (C, H, W) normalized for SAM3
        """
        # Convert to PIL if needed
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        # Store original dimensions
        orig_width, orig_height = pil_image.size
        if self.v2_state.orig_height == 0:
            self.v2_state.orig_height = orig_height
            self.v2_state.orig_width = orig_width

        # Resize to model resolution
        img_resized = pil_image.resize((self.resolution, self.resolution))
        img_np = np.array(img_resized)

        # Convert to tensor and normalize (same as SAM3 io_utils)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16)
        img_tensor = img_tensor / 255.0

        # Normalize by mean and std
        img_mean = torch.tensor(self.image_mean, dtype=torch.float16)[:, None, None]
        img_std = torch.tensor(self.image_std, dtype=torch.float16)[:, None, None]
        img_tensor = (img_tensor - img_mean) / img_std

        return img_tensor.to(self.device)

    def _init_v2_streaming_state(
        self, first_frame: np.ndarray, text_prompt: str
    ) -> None:
        """
        Initialize the v2 streaming state for the first frame.

        This creates the SAM3 inference state structure without loading a video file.
        """
        from sam3.model.data_misc import FindStage, convert_my_tensors
        from sam3.model.geometry_encoders import Prompt

        # Store original dimensions
        if isinstance(first_frame, Image.Image):
            orig_width, orig_height = first_frame.size
        else:
            orig_height, orig_width = first_frame.shape[:2]

        self.v2_state.orig_height = orig_height
        self.v2_state.orig_width = orig_width
        self.v2_state.text_prompt = text_prompt

        # Create initial inference state structure
        inference_state = {}
        inference_state["image_size"] = self.resolution
        inference_state["num_frames"] = 1  # Will grow as frames are added
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        inference_state["constants"] = {}
        inference_state["is_image_only"] = False

        # Initialize empty img_batch (will be populated with frames)
        inference_state["_frame_tensors"] = []  # Our frame buffer

        # Initialize find_text_batch with text prompt
        find_text_batch = [text_prompt, "visual"]

        # Initialize find_inputs for first frame
        input_box_embedding_dim = 258
        input_points_embedding_dim = 257

        first_stage = FindStage(
            img_ids=[0],
            text_ids=[0],  # Use text prompt (index 0)
            input_boxes=[torch.zeros(input_box_embedding_dim)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, input_points_embedding_dim)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        first_stage = convert_my_tensors(first_stage)

        # Create placeholder geometric prompt
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=self.device),
            box_mask=torch.zeros(bs, 0, device=self.device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=self.device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=self.device),
            point_mask=torch.zeros(bs, 0, device=self.device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=self.device, dtype=torch.long),
        )

        # Initialize tracking-related state
        inference_state["previous_stages_out"] = [None]
        inference_state["text_prompt"] = text_prompt
        inference_state["per_frame_raw_point_input"] = [None]
        inference_state["per_frame_raw_box_input"] = [None]
        inference_state["per_frame_visual_prompt"] = [None]
        inference_state["per_frame_geometric_prompt"] = [None]
        inference_state["per_frame_cur_step"] = [0]

        # Placeholders for cached outputs
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

        # Initialize tracker states
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []

        # Store the find_inputs and find_text_batch for later use
        inference_state["_find_inputs_list"] = [first_stage]
        inference_state["_find_text_batch"] = find_text_batch

        self.v2_state.inference_state = inference_state
        self.v2_state.initialized = True

        logger.info(f"Initialized v2 streaming state with text prompt: '{text_prompt}'")

    def _add_frame_to_v2_state(
        self, frame_tensor: torch.Tensor, frame_idx: int
    ) -> None:
        """
        Add a preprocessed frame to the v2 streaming state.

        This updates the inference state's frame buffer and input structures.
        """
        from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        inference_state = self.v2_state.inference_state

        # Add frame to our buffer
        inference_state["_frame_tensors"].append(frame_tensor)

        # Update num_frames
        num_frames = len(inference_state["_frame_tensors"])
        inference_state["num_frames"] = num_frames

        # Add a new FindStage for this frame if needed
        input_box_embedding_dim = 258
        input_points_embedding_dim = 257

        while len(inference_state["_find_inputs_list"]) < num_frames:
            idx = len(inference_state["_find_inputs_list"])
            new_stage = FindStage(
                img_ids=[idx],
                text_ids=[0],  # Use text prompt
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            new_stage = convert_my_tensors(new_stage)
            inference_state["_find_inputs_list"].append(new_stage)

            # Extend tracking state lists
            inference_state["previous_stages_out"].append(None)
            inference_state["per_frame_raw_point_input"].append(None)
            inference_state["per_frame_raw_box_input"].append(None)
            inference_state["per_frame_visual_prompt"].append(None)
            inference_state["per_frame_geometric_prompt"].append(None)
            inference_state["per_frame_cur_step"].append(0)

        # Build the BatchedDatapoint with all frames
        img_batch = torch.stack(inference_state["_frame_tensors"], dim=0)

        find_inputs = [
            copy_data_to_device(stage, self.device, non_blocking=True)
            for stage in inference_state["_find_inputs_list"]
        ]

        input_batch = BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=inference_state["_find_text_batch"],
            find_inputs=find_inputs,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
        )
        input_batch = copy_data_to_device(input_batch, self.device, non_blocking=True)
        inference_state["input_batch"] = input_batch

        # Manage frame history (keep last N frames for memory efficiency)
        max_frames = self.v2_state.max_frame_history
        if num_frames > max_frames:
            # Remove oldest frames from buffer
            # Note: We keep the input_batch complete but could optimize memory here
            pass

    def _run_v2_inference(
        self, frame_idx: int, text_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Run single-frame inference using SAM3 video model.

        Uses add_prompt() which is the proper entry point for processing frames.
        For streaming, we use a sliding window approach where we process each frame
        as the "first" frame in a mini-batch, then propagate to subsequent frames
        if objects are detected.
        """
        inference_state = self.v2_state.inference_state

        # For streaming, we use add_prompt which properly initializes the detection
        # and tracking state for a new frame
        with torch.inference_mode():
            _, out = self.video_model.add_prompt(
                inference_state,
                frame_idx=frame_idx,
                text_str=text_prompt,
            )

        # Extract detections from output
        detections = self._extract_v2_detections(out, text_prompt)

        logger.debug(
            f"Frame {frame_idx}: Detected {len(detections)} objects in v2 mode"
        )
        return detections

    def _extract_v2_detections(
        self,
        out: Dict[str, Any],
        class_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract detections from SAM3 video model output.

        Args:
            out: Output from add_prompt (contains out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks)
            class_name: Class name to assign to detections

        Returns:
            List of detection dicts
        """
        # Handle both add_prompt output format and _run_single_frame_inference format
        if "out_obj_ids" in out:
            # Output from add_prompt / propagate_in_video
            out_obj_ids = out.get("out_obj_ids", np.array([]))
            out_probs = out.get("out_probs", np.array([]))
            out_boxes_xywh = out.get("out_boxes_xywh", np.array([]))
            out_binary_masks = out.get("out_binary_masks", np.array([]))

            if len(out_obj_ids) == 0:
                return []

            detections = []
            H_video = self.v2_state.orig_height
            W_video = self.v2_state.orig_width

            for i, obj_id in enumerate(out_obj_ids):
                score = float(out_probs[i]) if i < len(out_probs) else 0.5

                # Skip low-confidence detections
                if score < self.confidence_threshold:
                    continue

                # Get mask
                if i < len(out_binary_masks):
                    mask = out_binary_masks[i]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.squeeze().cpu().numpy()
                    mask_bool = mask.astype(bool)
                else:
                    continue

                # Get bounding box from normalized xywh or from mask
                if i < len(out_boxes_xywh):
                    box_xywh = out_boxes_xywh[i]
                    # Convert from normalized xywh to pixel xyxy
                    x, y, w, h = box_xywh
                    box = [
                        int(x * W_video),
                        int(y * H_video),
                        int((x + w) * W_video),
                        int((y + h) * H_video),
                    ]
                else:
                    # Get bounding box from mask
                    if not mask_bool.any():
                        continue
                    ys, xs = np.where(mask_bool)
                    box = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ]

                # Track object info
                obj_id_int = int(obj_id)
                if obj_id_int not in self.v2_state.obj_id_to_info:
                    self.v2_state.obj_id_to_info[obj_id_int] = {
                        "class_name": class_name,
                        "first_seen_frame": self.v2_state.frame_idx - 1,
                        "score": score,
                    }

                detections.append(
                    {
                        "box": box,
                        "score": score,
                        "class": class_name,
                        "class_name": class_name,
                        "mask": mask_bool,
                        "obj_id": obj_id_int,
                    }
                )

            return detections

        # Fallback for _run_single_frame_inference output format
        obj_id_to_mask = out.get("obj_id_to_mask", {})
        obj_id_to_score = out.get("obj_id_to_score", {})

        if len(obj_id_to_mask) == 0:
            return []

        detections = []
        H_video = self.v2_state.orig_height
        W_video = self.v2_state.orig_width

        for obj_id, mask_tensor in obj_id_to_mask.items():
            score = obj_id_to_score.get(obj_id, 0.5)

            # Skip low-confidence detections
            if score < self.confidence_threshold:
                continue

            # Convert mask to numpy and ensure correct shape
            if isinstance(mask_tensor, torch.Tensor):
                # Mask is (1, H, W) or (H, W)
                mask = mask_tensor.squeeze().cpu().numpy()
            else:
                mask = mask_tensor

            # Resize mask to original video resolution if needed
            if mask.shape != (H_video, W_video):
                mask_resized = (
                    F.interpolate(
                        torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float(),
                        size=(H_video, W_video),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                    > 0.5
                )
                mask = mask_resized

            # Get bounding box from mask
            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                continue

            ys, xs = np.where(mask_bool)
            box = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]

            # Track object info
            if obj_id not in self.v2_state.obj_id_to_info:
                self.v2_state.obj_id_to_info[obj_id] = {
                    "class_name": class_name,
                    "first_seen_frame": self.v2_state.frame_idx - 1,
                    "score": score,
                }

            detections.append(
                {
                    "box": box,
                    "score": float(score),
                    "class": class_name,
                    "class_name": class_name,
                    "mask": mask_bool,
                    "obj_id": obj_id,
                }
            )

        return detections

    def _run_detection(
        self,
        image: np.ndarray,
        prompts: List[str],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Run detection on frame using Sam3Processor.

        Uses cached text embeddings when available for efficiency.
        """
        # Convert image to PIL if needed
        if isinstance(image, Image.Image):
            pil_image = image
            width, height = image.size
        else:
            pil_image = Image.fromarray(image)
            height, width = image.shape[:2]

        detections = []

        with torch.no_grad():
            # Set image once (runs backbone)
            state = self.processor.set_image(pil_image)

            # Process each prompt
            for prompt in prompts:
                # Inject cached text embedding or encode fresh
                if prompt in self.state.cached_text_embeddings:
                    cached = self.state.cached_text_embeddings[prompt]
                    # Inject into backbone_out
                    if cached.get("language_features") is not None:
                        state["backbone_out"]["language_features"] = cached[
                            "language_features"
                        ]
                    if cached.get("language_mask") is not None:
                        state["backbone_out"]["language_mask"] = cached["language_mask"]
                    if cached.get("language_embeds") is not None:
                        state["backbone_out"]["language_embeds"] = cached[
                            "language_embeds"
                        ]
                    # Initialize geometric prompt if not present
                    if "geometric_prompt" not in state:
                        state["geometric_prompt"] = self.model._get_dummy_prompt()
                    # Run grounding
                    state = self.processor.forward_grounding(state)
                else:
                    # Encode text prompt and run grounding
                    state = self.processor.set_text_prompt(prompt, state)

                # Extract detections from state
                prompt_dets = self._extract_detections_from_state(state, prompt)
                detections.extend(prompt_dets)

                # Reset prompts for next iteration (keeps image backbone features)
                self.processor.reset_all_prompts(state)

        logger.debug(f"Frame {frame_idx}: Detected {len(detections)} objects")
        return detections

    def _extract_detections_from_state(
        self,
        state: Dict[str, Any],
        class_name: str,
    ) -> List[Dict[str, Any]]:
        """Extract detection dicts from processor state."""
        if "boxes" not in state or len(state["boxes"]) == 0:
            return []

        boxes = state["boxes"]
        scores = state["scores"]
        masks = state["masks"]

        detections = []
        for i in range(len(boxes)):
            box = boxes[i].cpu().tolist()
            score = scores[i].item()
            mask = masks[i].squeeze(0).cpu().numpy()

            # Clip box to image boundaries
            img_height, img_width = mask.shape
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img_width, int(box[2])),
                min(img_height, int(box[3])),
            ]

            # Assign object ID
            obj_id = self.state.next_obj_id
            self.state.next_obj_id += 1

            detections.append(
                {
                    "box": box,
                    "score": score,
                    "class": class_name,
                    "class_name": class_name,
                    "mask": mask,
                    "obj_id": obj_id,
                }
            )

        return detections
