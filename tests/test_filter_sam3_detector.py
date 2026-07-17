#!/usr/bin/env python

import logging
import os
import sys
import unittest

from filter_sam3_detector.filter import FilterSAM3Detector
from openfilter.filter_runtime.filter import FilterConfig

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper())))

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class TestFilterSAM3Detector(unittest.TestCase):
    def test_config_defaults(self):
        """Test that default configuration values are set correctly."""
        config = FilterSAM3Detector.normalize_config(FilterConfig({}))

        self.assertEqual(config["model_id"], "facebook/sam3")
        # Default is "cuda" but we don't test actual GPU usage in CI
        self.assertIn(config["device"], ["cuda", "cpu", "mps"])
        self.assertIsNone(config["text_prompt"])
        self.assertIsNone(config["exemplars_path"])
        self.assertEqual(config["confidence_threshold"], 0.5)
        self.assertEqual(config["mask_threshold"], 0.5)
        self.assertEqual(config["max_detections"], 100)
        self.assertFalse(config["output_masks"])
        self.assertTrue(config["output_boxes"])
        self.assertTrue(config["output_scores"])
        self.assertEqual(config["output_label"], "sam3_detections")
        self.assertFalse(config["debug"])
        self.assertFalse(config["visualize"])

    def test_schema_compliance_and_serialization(self):
        """Test FilterSAM3DetectorOutput instantiates and serializes correctly."""
        from filter_sam3_detector.filter import FilterSAM3DetectorOutput

        # Valid canonical detection dictionary
        valid_items = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "mask": {
                    "polygons": [
                        {"points": [(10.0, 20.0), (30.0, 20.0), (30.0, 45.0)]}
                    ],
                    "area": 150,
                },
            }
        ]

        output = FilterSAM3DetectorOutput(items=valid_items)
        dumped = output.model_dump(mode="json")

        self.assertIn("items", dumped)
        self.assertEqual(len(dumped["items"]), 1)
        det = dumped["items"][0]
        self.assertEqual(det["label"], "car")
        self.assertEqual(det["score"], 0.85)
        self.assertEqual(det["bbox"]["x1"], 10.0)
        self.assertEqual(det["mask"]["area"], 150)

    def test_schema_coordinate_validation_error(self):
        """Test that invalid coordinates (e.g. x2 < x1 or y2 < y1) trigger a ValidationError."""
        from filter_sam3_detector.filter import FilterSAM3DetectorOutput
        from pydantic import ValidationError

        # Invalid coordinates where x2 < x1
        invalid_items = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 5.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
            }
        ]

        with self.assertRaises(ValidationError):
            FilterSAM3DetectorOutput(items=invalid_items)

    def test_schema_extra_fields_ignored_during_serialization(self):
        """Test that any non-schema legacy fields (e.g. 'box' list, 'mask_np') are ignored during serialization."""
        from filter_sam3_detector.filter import FilterSAM3DetectorOutput
        import numpy as np

        mixed_items = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                # Extra legacy / internal fields
                "box": [10, 20, 30, 45],
                "confidence": 0.85,
                "mask_np": np.zeros((10, 10)),
                "category_id": 1,
            }
        ]

        output = FilterSAM3DetectorOutput(items=mixed_items)
        dumped = output.model_dump(mode="json")

        det = dumped["items"][0]
        # Should contain ONLY standard schema fields
        self.assertIn("bbox", det)
        self.assertIn("score", det)
        self.assertIn("label", det)
        self.assertIn("label_id", det)
        self.assertIn("mask", det)

        # Should NOT contain extra non-schema legacy fields
        self.assertNotIn("box", det)
        self.assertNotIn("confidence", det)
        self.assertNotIn("mask_np", det)
        self.assertNotIn("category_id", det)

    def test_serialize_detections_gating_and_pruning(self):
        """Test that setup() raises a ValueError when output_boxes or output_scores are False."""
        config = FilterSAM3Detector.normalize_config(
            FilterConfig(
                {
                    "output_boxes": False,
                    "output_scores": True,
                }
            )
        )
        detector = FilterSAM3Detector(config)

        with self.assertRaises(ValueError) as context:
            detector.setup(config)

        self.assertIn(
            "output_boxes and output_scores must both be True", str(context.exception)
        )

    def test_temporal_intervals_fallback(self):
        """Test that _aggregate_detections correctly falls back to standard schema keys."""
        from filter_sam3_detector.temporal_intervals import TemporalIntervalFilter

        config = TemporalIntervalFilter.normalize_config(
            {
                "label_field": "class",
                "score_field": "confidence",
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Schema-compliant detections: missing "class" and "confidence", but has "label" and "score"
        detections = [
            {"label": "person", "score": 0.9},
            {"label": "car", "score": 0.8},
        ]

        aggregated = filter_instance._aggregate_detections(detections)
        self.assertIn("person", aggregated)
        self.assertEqual(aggregated["person"], 0.9)
        self.assertIn("car", aggregated)
        self.assertEqual(aggregated["car"], 0.8)

    def test_normalize_detections_degenerate_polygons_filtered(self):
        """Test that _normalize_detections filters out polygons with <3 points to prevent validation errors."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "mask": {
                    "polygons": [
                        {
                            "points": [(10.0, 20.0), (30.0, 20.0)]
                        },  # Degenerate: only 2 points
                        {
                            "points": [(10.0, 20.0), (30.0, 20.0), (30.0, 45.0)]
                        },  # Valid: 3 points
                    ],
                    "area": 150,
                },
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)

        # Valid polygon should be kept
        self.assertEqual(len(canonical["items"][0]["mask"]["polygons"]), 1)
        self.assertEqual(len(canonical["items"][0]["mask"]["polygons"][0]["points"]), 3)

    def test_normalize_detections_with_none_label_or_prompt(self):
        """Test that _normalize_detections safely extracts string labels/prompts even when values are None."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "box": [10.0, 20.0, 30.0, 45.0],
                "score": 0.85,
                "label": None,
                "class": "truck",
                "prompt": None,
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, protege, _ = detector._normalize_detections(detections)

        self.assertEqual(canonical["items"][0]["label"], "truck")
        self.assertEqual(protege[0]["prompt"], "truck")

    def test_normalize_detections_with_non_dict_mask(self):
        """Test that _normalize_detections safely ignores non-dictionary masks instead of crashing."""
        import numpy as np
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "mask": np.zeros((100, 100), dtype=np.uint8),  # Binary array mask
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)

        self.assertIsNone(canonical["items"][0]["mask"])

    def test_normalize_detections_with_direct_list_polygons(self):
        """Test that _normalize_detections correctly processes direct list/tuple of points for polygons."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "mask": {
                    "polygons": [
                        [
                            (10.0, 20.0),
                            (30.0, 20.0),
                            (30.0, 45.0),
                        ]  # Direct list of points
                    ],
                    "area": 150,
                },
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)

        self.assertEqual(len(canonical["items"][0]["mask"]["polygons"]), 1)
        self.assertEqual(len(canonical["items"][0]["mask"]["polygons"][0]["points"]), 3)

    def test_normalize_detections_invalid_label_id(self):
        """Test that _normalize_detections safely ignores non-integer label_id instead of crashing."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "label_id": "not-an-integer",
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)

        self.assertIsNone(canonical["items"][0]["label_id"])

    def test_normalize_detections_float_string_label_id(self):
        """Test that _normalize_detections robustly handles float-formatted label_id strings."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detections = [
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
                "label_id": "12.0",
            }
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)

        self.assertEqual(canonical["items"][0]["label_id"], 12)

    def test_config_dictionary_protocol_and_idempotence(self):
        """Test that the dictionary protocol includes extra fields and normalize_config is idempotent."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import FilterConfig

        # 1. Verify custom attributes are captured in dictionary conversion
        custom_input = {
            "text_prompt": "electric post",
            "confidence_threshold": 0.85,
            "device": "cpu",
        }
        config = FilterSAM3Detector.normalize_config(FilterConfig(custom_input))

        config_dict = dict(config)
        self.assertIn("text_prompt", config_dict)
        self.assertIn("confidence_threshold", config_dict)
        self.assertEqual(config_dict["text_prompt"], "electric post")
        self.assertEqual(config_dict["confidence_threshold"], 0.85)

        # 2. Verify idempotence: normalizing already-normalized config retains custom values
        normalized_twice = FilterSAM3Detector.normalize_config(config)
        self.assertEqual(normalized_twice.get("text_prompt"), "electric post")
        self.assertEqual(normalized_twice.get("confidence_threshold"), 0.85)

    def test_deprecated_video_env_vars_warn(self):
        """Test that deprecated video throttling env vars emit warnings instead of failing silently."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import FilterConfig

        original_env = {
            "FILTER_VIDEO_DETECTION_INTERVAL": os.environ.get(
                "FILTER_VIDEO_DETECTION_INTERVAL"
            ),
            "FILTER_VIDEO_MIN_TRACKING_CONFIDENCE": os.environ.get(
                "FILTER_VIDEO_MIN_TRACKING_CONFIDENCE"
            ),
        }
        try:
            os.environ["FILTER_VIDEO_DETECTION_INTERVAL"] = "10"
            os.environ["FILTER_VIDEO_MIN_TRACKING_CONFIDENCE"] = "0.7"
            with self.assertLogs(
                "filter_sam3_detector.filter", level="WARNING"
            ) as logs:
                FilterSAM3Detector.normalize_config(FilterConfig({}))

            warning_text = "\n".join(logs.output)
            self.assertIn("FILTER_VIDEO_DETECTION_INTERVAL", warning_text)
            self.assertIn("FILTER_VIDEO_MIN_TRACKING_CONFIDENCE", warning_text)
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_single_output_handles_exception_gracefully(self):
        """Verify that single-output process handles exceptions gracefully by yielding a degraded frame."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import FilterConfig, Frame
        from unittest.mock import MagicMock
        import numpy as np

        # Create a config
        config = FilterSAM3Detector.normalize_config(
            FilterConfig({"text_prompt": "car"})
        )

        # Create a stub detector using __new__
        detector = FilterSAM3Detector.__new__(FilterSAM3Detector)
        detector.config = config
        detector.model = MagicMock()
        detector.processor = MagicMock()
        detector.prompt_sets = None
        detector.text_prompts = None
        detector.text_prompt = "car"
        detector.positive_boxes = []
        detector.negative_boxes = []
        detector.ref_images_paths = None
        detector.ref_images_negative_paths = None
        detector.enable_temporal_intervals = False
        detector.enable_video_mode = False
        detector.output_filter_name = "SAM3Detector"
        detector.output_label = "sam3_detections"
        detector.viz_topic = ""
        detector.visualize = False
        detector.jsonl_file = None
        detector.frame_counter = 0

        # Force an exception during image preprocessing or inference
        # We mock Image.fromarray to raise an error
        from unittest.mock import patch

        with patch(
            "PIL.Image.fromarray", side_effect=RuntimeError("Simulated inference error")
        ):
            # Create a mock Frame with image data using MagicMock to avoid shapes import issues
            bgr_data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = MagicMock(spec=Frame)
            frame.has_image = True
            rw_bgr = MagicMock()
            rw_bgr.image = bgr_data
            frame.rw_bgr = rw_bgr
            frame.data = {}

            # Run process
            result_frames = detector.process({"main": frame})

            # Verify that the frame was returned, but gracefully populated as a degraded frame
            self.assertIn("main", result_frames)
            out_frame = result_frames["main"]

            self.assertIn("detections", out_frame.data)
            self.assertEqual(out_frame.data["detections"]["items"], [])

            frame_meta = out_frame.data["meta"]
            self.assertEqual(frame_meta["width"], 640)
            self.assertEqual(frame_meta["height"], 480)
            self.assertEqual(frame_meta["detections"], [])
            self.assertEqual(
                frame_meta["classification"],
                {"classes": [], "confidences": [], "architecture": "sam3"},
            )
            self.assertEqual(frame_meta[detector.output_label], [])

    def test_video_mode_failure_forwards_frame_unchanged(self):
        """Verify that video mode returns the original frame unchanged on preprocessing failure."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import FilterConfig, Frame
        from unittest.mock import MagicMock, patch
        import numpy as np

        config = FilterSAM3Detector.normalize_config(
            FilterConfig({"enable_video_mode": True, "text_prompt": "car"})
        )

        detector = FilterSAM3Detector.__new__(FilterSAM3Detector)
        detector.config = config
        detector.video_model = MagicMock()
        detector.video_processor = MagicMock()
        detector.video_inference_session = MagicMock()
        detector.prune_video_memory = False
        detector.frame_counter = 0
        detector._video_mode_viz_frame = None
        detector.output_filter_name = "SAM3Detector"
        detector.output_label = "sam3_detections"
        detector.viz_topic = ""
        detector.visualize = False
        detector.jsonl_file = None
        detector.enable_temporal_intervals = False
        detector.text_prompt = "car"
        detector.text_prompts = None
        detector.prompt_sets = None
        detector.positive_boxes = []
        detector.negative_boxes = []
        detector.ref_images_paths = None
        detector.ref_images_negative_paths = None
        detector.frames_dir = None
        detector.annotated_frames_dir = None
        detector.mixed_precision = False
        detector.device = MagicMock()

        bgr_data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = MagicMock(spec=Frame)
        frame.has_image = True
        rw_bgr = MagicMock()
        rw_bgr.image = bgr_data
        frame.rw_bgr = rw_bgr
        frame.data = {}

        with patch(
            "PIL.Image.fromarray",
            side_effect=RuntimeError("Simulated video preprocessing error"),
        ):
            result = detector._process_video_mode_frame(frame, None)

        self.assertIs(result, frame)
        self.assertEqual(frame.data, {})

    def test_video_mode_applies_max_detections(self):
        """Verify that video-mode detections are truncated to max_detections by score."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        detector = FilterSAM3Detector.__new__(FilterSAM3Detector)
        detector.config = {}
        detector.output_masks = False
        detector.max_detections = 2

        processed_outputs = {
            "object_ids": [1, 2, 3],
            "scores": [0.25, 0.9, 0.5],
            "boxes": [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            "prompt_to_obj_ids": {"car": [1, 2, 3]},
        }

        detections = detector._video_outputs_to_detections(processed_outputs)

        self.assertEqual(len(detections), 2)
        self.assertGreaterEqual(detections[0]["score"], detections[1]["score"])
        self.assertEqual([det["id"] for det in detections], [2, 3])

    def test_video_mode_postprocess_failure_advances_counter(self):
        """Verify that postprocess failures still advance the video frame counter."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import Frame
        from unittest.mock import MagicMock
        import numpy as np

        detector = FilterSAM3Detector.__new__(FilterSAM3Detector)
        detector.video_model = MagicMock()
        detector.video_processor = MagicMock()
        detector.video_inference_session = MagicMock()
        detector.prune_video_memory = False
        detector.frame_counter = 7
        detector._video_mode_viz_frame = None
        detector.output_filter_name = "SAM3Detector"
        detector.output_label = "sam3_detections"
        detector.viz_topic = ""
        detector.visualize = False
        detector.jsonl_file = None
        detector.enable_temporal_intervals = False
        detector.text_prompt = "car"
        detector.text_prompts = None
        detector.prompt_sets = None
        detector.positive_boxes = []
        detector.negative_boxes = []
        detector.ref_images_paths = None
        detector.ref_images_negative_paths = None
        detector.frames_dir = None
        detector.annotated_frames_dir = None
        detector.mixed_precision = False
        detector.device = MagicMock()
        detector.max_detections = 100
        detector.output_masks = False
        detector._extract_filter_frame_id = MagicMock(return_value=None)

        pixel_values = MagicMock(name="pixel_values_0")
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.pixel_values = [pixel_values]
        inputs.original_sizes = [(480, 640)]
        detector.video_processor.return_value = inputs
        detector.video_processor.postprocess_outputs.side_effect = RuntimeError(
            "Simulated postprocess error"
        )

        bgr_data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = MagicMock(spec=Frame)
        frame.has_image = True
        rw_bgr = MagicMock()
        rw_bgr.image = bgr_data
        frame.rw_bgr = rw_bgr
        frame.data = {}

        result = detector._process_video_mode_frame(frame, None)

        self.assertIs(result, frame)
        self.assertEqual(detector.frame_counter, 8)
        inputs.to.assert_called_once_with(detector.device)
        detector.video_model.assert_called_once()
        self.assertEqual(detector.video_model.call_args.kwargs["frame_idx"], 7)
        self.assertIs(detector.video_model.call_args.kwargs["frame"], pixel_values)

    def test_normalize_detections_validate_per_item(self):
        """Test that _normalize_detections validates detections per-item, dropping invalid ones but preserving valid ones."""
        from filter_sam3_detector.filter import FilterSAM3Detector
        from openfilter.filter_runtime.filter import FilterConfig

        detections = [
            # Valid item
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0},
                "score": 0.85,
                "label": "car",
            },
            # Invalid item (x2 < x1)
            {
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 5.0, "y2": 45.0},
                "score": 0.90,
                "label": "person",
            },
            # Valid item 2
            {
                "bbox": {"x1": 100.0, "y1": 200.0, "x2": 150.0, "y2": 250.0},
                "score": 0.80,
                "label": "truck",
            },
        ]

        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, protege, classification = detector._normalize_detections(detections)

        # 1. The valid detections should be preserved in the canonical list
        self.assertEqual(len(canonical["items"]), 2)
        self.assertEqual(canonical["items"][0]["label"], "car")
        self.assertEqual(canonical["items"][1]["label"], "truck")

        # 2. The valid detections should be preserved in the protege legacy list
        self.assertEqual(len(protege), 2)
        self.assertEqual(protege[0]["label"], "car")
        self.assertEqual(protege[1]["label"], "truck")

        # 3. Only the valid labels should be in the classification dict
        self.assertEqual(classification["classes"], ["car", "truck"])
        self.assertEqual(classification["confidences"], [0.85, 0.80])


if __name__ == "__main__":
    unittest.main()
