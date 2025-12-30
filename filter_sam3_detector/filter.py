import logging
import os
import json
import multiprocessing
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

__all__ = ["FilterSAM3DetectorConfig", "FilterSAM3Detector"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import SAM3 from facebookresearch/sam3
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
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
            "model_id": "facebook/sam2-hiera-large",
            "device": "cuda",
            "text_prompt": None,
            "exemplars_path": None,
            "exemplar_embeddings_cache": None,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "max_detections": 100,
            "output_masks": True,
            "output_boxes": True,
            "output_scores": True,
            "output_label": "sam3_detections",
            "output_path": None,  # Path to save JSONL annotations
            "frames_output_dir": None,  # Directory to save frames with detections
            "debug": False,
            "visualize": False,
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Load from environment variables (override config values)
        env_mapping = {
            "model_id": str,
            "device": str,
            "text_prompt": str,
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
            "frames_output_dir": str,
            "debug": bool,
            "visualize": bool,
        }

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
        self.debug = config.get("debug", False)
        
        # Initialize jsonl_file to None (will be set if output_path is provided)
        self.jsonl_file = None

        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Store configuration (access as dict since FilterConfig is dict-like)
        self.model_id = config.get("model_id", "facebook/sam2-hiera-large")
        self.text_prompt = config.get("text_prompt")
        self.exemplars_path = config.get("exemplars_path")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.max_detections = config.get("max_detections", 100)
        self.output_masks = config.get("output_masks", True)
        self.output_boxes = config.get("output_boxes", True)
        self.output_scores = config.get("output_scores", True)
        self.output_label = config.get("output_label", "sam3_detections")
        self.output_path = config.get("output_path", None)
        self.frames_output_dir = config.get("frames_output_dir", None)
        self.visualize = config.get("visualize", False)
        
        # Initialize JSONL output file if path is provided
        self.jsonl_file = None
        if self.output_path:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = open(output_path, 'w')
            logger.info(f"Saving annotations to: {output_path}")
        
        # Initialize frames output directory if provided
        self.frames_dir = None
        if self.frames_output_dir:
            self.frames_dir = Path(self.frames_output_dir)
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            if self.debug:
                logger.info(f"Saving frames with detections (annotated) to: {self.frames_dir}")
            else:
                logger.info(f"Saving original frames to: {self.frames_dir}")

        # Determine device
        device_str = config.get("device", "cuda")
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # Load SAM3 model
        self.model = None
        self.processor = None
        self._load_model()

        # Load exemplar images if provided
        self.visual_prompt_embed = None
        self.visual_prompt_mask = None
        if self.exemplars_path:
            self._load_exemplar_images()

        logger.info("FilterSAM3Detector setup complete")

    def shutdown(self):
        """
        Clean up resources when the filter is stopped.

        This method should release any held resources like:
        - GPU memory
        - File handles
        - Cached data
        """
        logger.info("FilterSAM3Detector shutdown")

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

    def process(self, frames: dict[str, Frame]) -> dict[str, Frame]:
        """
        Process input frames and detect objects.

        Args:
            frames: Dictionary of input frames keyed by topic name

        Returns:
            Dictionary of output frames with detection results
        """
        output_frames = {}

        for topic, frame in frames.items():
            if frame is None:
                continue

            if not frame.has_image:
                # Forward non-image frames unchanged
                output_frames[topic] = frame
                continue

            if self.debug:
                logger.debug(f"Processing frame from topic: {topic}")

            # Check if model is loaded
            if self.model is None or self.processor is None:
                logger.warning("SAM3 model not loaded, forwarding frame unchanged")
                output_frames[topic] = frame
                continue

            # Need either text prompt or visual embeddings from exemplars
            if self.text_prompt is None and self.visual_prompt_embed is None:
                logger.warning("No text prompt or exemplars configured, forwarding frame unchanged")
                output_frames[topic] = frame
                continue

            try:
                # Extract image from frame (convert BGR to RGB PIL)
                image_bgr = frame.rw_bgr.image
                image_rgb = image_bgr[:, :, ::-1]  # BGR to RGB
                pil_image = Image.fromarray(image_rgb)

                # Set image in processor
                state = self.processor.set_image(pil_image)

                # Run inference with text prompt and/or visual embeddings
                if self.text_prompt is not None:
                    state = self.processor.set_text_prompt(self.text_prompt, state)

                # If we have visual embeddings from exemplar images, run grounding with them
                if self.visual_prompt_embed is not None:
                    # Ensure we have language features (use "visual" as placeholder if no text prompt)
                    if "language_features" not in state["backbone_out"]:
                        dummy_text_outputs = self.model.backbone.forward_text(
                            ["visual"], device=str(self.device)
                        )
                        state["backbone_out"].update(dummy_text_outputs)

                    # Initialize geometric prompt if not present
                    if "geometric_prompt" not in state:
                        state["geometric_prompt"] = self.model._get_dummy_prompt()

                    # Run grounding with visual prompt embeddings
                    state = self._forward_grounding_with_visual_prompt(state)
                elif self.text_prompt is None:
                    # No prompts at all
                    output_frames[topic] = frame
                    continue

                # Extract detections from state
                detections = []

                if "boxes" in state and "scores" in state:
                    boxes = state["boxes"]
                    scores = state["scores"]
                    masks = state.get("masks", None)

                    num_detections = min(len(boxes), self.max_detections)

                    for i in range(num_detections):
                        detection = {}

                        if self.output_boxes:
                            box = boxes[i]
                            if hasattr(box, 'tolist'):
                                box = box.tolist()
                            detection['box'] = [int(x) for x in box]

                        if self.output_scores:
                            score = scores[i]
                            if hasattr(score, 'item'):
                                score = score.item()
                            detection['score'] = float(score)

                        if self.output_masks and masks is not None and i < len(masks):
                            mask = masks[i]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            if hasattr(mask, 'squeeze'):
                                mask = mask.squeeze()
                            # Store mask as binary array
                            detection['mask'] = (mask > 0).astype(np.uint8).tolist()

                        if detection:
                            detections.append(detection)

                # Store results in frame metadata
                frame.data.setdefault('meta', {})[self.output_label] = detections

                # Get frame metadata for JSONL and filename
                frame_meta = frame.data.get('meta', {})
                frame_id = frame_meta.get('frame_id', len(output_frames))
                # Get timestamp from frame (OpenFilter provides this)
                timestamp = getattr(frame, 'timestamp', None) or frame_meta.get('timestamp', None) or frame.data.get('timestamp', None)
                if timestamp:
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds precision
                else:
                    # Fallback: use frame_id if timestamp not available
                    timestamp_str = f"frame{frame_id:06d}"
                
                # Generate unique filename using timestamp and frame_id
                frame_filename = None
                if self.frames_dir is not None:
                    frame_filename = f"{timestamp_str}_frame{frame_id:06d}.jpg"
                    frame_filename = self.frames_dir / frame_filename

                # Save to JSONL file if output_path is configured
                if hasattr(self, 'jsonl_file') and self.jsonl_file is not None:
                    try:
                        annotation = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "timestamp_str": timestamp_str,
                            "filename": frame_filename.name if frame_filename else None,
                            "num_detections": len(detections),
                            "meta": {
                                self.output_label: detections
                            }
                        }
                        self.jsonl_file.write(json.dumps(annotation) + '\n')
                        self.jsonl_file.flush()  # Ensure immediate write
                    except Exception as e:
                        logger.warning(f"Failed to save annotation to JSONL: {e}")

                if self.debug:
                    logger.debug(f"[{topic}] Found {len(detections)} detections")

                # Save frame if detections found and frames_output_dir is configured
                if detections and self.frames_dir is not None and frame_filename:
                    try:
                        import cv2
                        # Get image from frame
                        image_bgr = frame.rw_bgr.image.copy()
                        
                        # Only visualize detections if debug mode is enabled
                        if self.debug:
                            image_bgr = self._visualize_detections_on_image(image_bgr, detections)
                        
                        # Save frame
                        cv2.imwrite(str(frame_filename), image_bgr)
                        
                        if self.debug:
                            logger.debug(f"Saved frame {'with annotations' if self.debug else ''} to: {frame_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to save frame: {e}")

                # Optional visualization (for output frame)
                if self.visualize and detections:
                    frame = self._visualize_detections(frame, detections)

            except Exception as e:
                logger.error(f"Error processing frame from {topic}: {e}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())

            output_frames[topic] = frame

        return output_frames

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
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if exemplars_path.is_file():
                # Single image file
                image_files = [exemplars_path]
            else:
                # Directory of images
                image_files = [
                    f for f in exemplars_path.iterdir()
                    if f.suffix.lower() in image_extensions
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
                        if "sam2_backbone_out" in backbone_out:
                            # Use SAM2 backbone features
                            feats = backbone_out["sam2_backbone_out"]["backbone_fpn"][-1]
                        else:
                            # Fallback to other feature format
                            feats = backbone_out.get("backbone_fpn", [backbone_out.get("image_embed")])[- 1]

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

            # Average all embeddings to create the visual prompt
            # Stack and average: [N, 1, C] -> [1, C]
            stacked = torch.cat(all_embeddings, dim=0)  # [N, C]
            averaged = stacked.mean(dim=0, keepdim=True)  # [1, C]

            # Format for SAM3's visual prompt: [seq_len, batch, hidden_dim]
            # We treat the averaged embedding as a single visual token
            self.visual_prompt_embed = averaged.unsqueeze(0)  # [1, 1, C]
            self.visual_prompt_mask = torch.zeros(
                (1, 1), device=self.device, dtype=torch.bool
            )  # No masking

            logger.info(f"Created visual prompt embedding from {len(all_embeddings)} exemplar images")

        except Exception as e:
            logger.error(f"Failed to load exemplar images: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.visual_prompt_embed = None
            self.visual_prompt_mask = None

    def _forward_grounding_with_visual_prompt(self, state):
        """
        Run SAM3 grounding with visual prompt embeddings from exemplar images.

        This is similar to the processor's _forward_grounding but includes
        visual prompt embeddings.
        """
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        # Run the model's forward_grounding with visual prompt
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
            visual_prompt_embed=self.visual_prompt_embed,
            visual_prompt_mask=self.visual_prompt_mask,
        )

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
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

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

    def _visualize_detections(self, frame: Frame, detections: list) -> Frame:
        """
        Draw detection results on the frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            Frame with visualizations drawn
        """
        try:
            import cv2
            image = frame.rw_bgr.image.copy()

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

            # Update frame with visualized image
            frame.rw_bgr.image = image

        except Exception as e:
            logger.warning(f"Failed to visualize detections: {e}")

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


if __name__ == "__main__":
    FilterSAM3Detector.run()
