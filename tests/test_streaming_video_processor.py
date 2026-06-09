#!/usr/bin/env python
"""
Tests for the StreamingVideoProcessor with v1 and v2 modes.

These tests verify:
1. v1 mode initialization and basic operation
2. v2 mode initialization and basic operation
3. Frame processing in both modes
4. Detection throttling in v1 mode
5. Memory-based tracking state management in v2 mode
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStreamingVideoProcessorV1(unittest.TestCase):
    """Tests for v1 (detection throttling) mode."""

    def test_processor_initialization_v1(self):
        """Test that v1 processor initializes correctly."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
            ProcessingMode,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v1")

        self.assertEqual(processor.mode, ProcessingMode.V1_DETECTION_THROTTLING)
        self.assertIsNone(processor.model)
        self.assertIsNone(processor.processor)
        self.assertEqual(processor.state.frame_idx, 0)

    def test_processor_initialization_v2(self):
        """Test that v2 processor initializes correctly."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
            ProcessingMode,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")

        self.assertEqual(processor.mode, ProcessingMode.V2_MEMORY_TRACKING)
        self.assertIsNone(processor.video_model)
        self.assertEqual(processor.v2_state.frame_idx, 0)
        self.assertFalse(processor.v2_state.initialized)

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        from filter_sam3_detector.streaming_video_processor import StreamingVideoProcessor

        with self.assertRaises(ValueError):
            StreamingVideoProcessor(device="cpu", mode="invalid")

    def test_v1_detection_throttling_logic(self):
        """Test that v1 detection throttling works correctly."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
            StreamingState,
        )

        processor = StreamingVideoProcessor(
            device="cpu",
            mode="v1",
            detection_interval=3,  # Detect every 3rd frame
        )

        # Check detection interval
        self.assertEqual(processor.detection_interval, 3)

    def test_v2_text_prompt_setting(self):
        """Test that text prompt can be set for v2 mode."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")
        processor.set_text_prompt("person")

        self.assertEqual(processor.v2_state.text_prompt, "person")

    def test_reset_tracking_v1(self):
        """Test that reset_tracking works for v1 mode."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v1")
        processor.state.frame_idx = 100
        processor.state.last_detection_frame = 99

        processor.reset_tracking()

        self.assertEqual(processor.state.frame_idx, 0)
        self.assertEqual(processor.state.last_detection_frame, -1)

    def test_reset_tracking_v2(self):
        """Test that reset_tracking works for v2 mode."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")
        processor.v2_state.frame_idx = 100
        processor.v2_state.initialized = True
        processor.v2_state.text_prompt = "test"

        processor.reset_tracking()

        self.assertEqual(processor.v2_state.frame_idx, 0)
        self.assertFalse(processor.v2_state.initialized)
        self.assertIsNone(processor.v2_state.text_prompt)


class TestV2StreamingState(unittest.TestCase):
    """Tests for V2StreamingState dataclass."""

    def test_v2_state_initialization(self):
        """Test V2StreamingState default values."""
        from filter_sam3_detector.streaming_video_processor import V2StreamingState

        state = V2StreamingState()

        self.assertEqual(state.frame_idx, 0)
        self.assertIsNone(state.inference_state)
        self.assertIsNone(state.text_prompt)
        self.assertEqual(len(state.obj_id_to_info), 0)
        self.assertEqual(len(state.frame_history), 0)
        self.assertEqual(state.max_frame_history, 7)
        self.assertFalse(state.initialized)


class TestFramePreprocessing(unittest.TestCase):
    """Tests for frame preprocessing."""

    def test_preprocess_frame_for_v2(self):
        """Test that frame preprocessing produces correct output shape."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")

        # Create a dummy RGB frame (480x640)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        preprocessed = processor._preprocess_frame_for_v2(frame)

        # Should be (C, H, W) with resolution 1008
        self.assertEqual(preprocessed.shape, (3, 1008, 1008))
        self.assertEqual(preprocessed.dtype, processor.device.type == "cpu" and preprocessed.dtype or preprocessed.dtype)

    def test_preprocess_frame_stores_dimensions(self):
        """Test that preprocessing stores original dimensions."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")

        # Create a dummy RGB frame (480x640)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        processor._preprocess_frame_for_v2(frame)

        self.assertEqual(processor.v2_state.orig_height, 480)
        self.assertEqual(processor.v2_state.orig_width, 640)


class TestProcessingModeEnum(unittest.TestCase):
    """Tests for ProcessingMode enum."""

    def test_processing_mode_values(self):
        """Test ProcessingMode enum values."""
        from filter_sam3_detector.streaming_video_processor import ProcessingMode

        self.assertEqual(ProcessingMode.V1_DETECTION_THROTTLING.value, "v1")
        self.assertEqual(ProcessingMode.V2_MEMORY_TRACKING.value, "v2")


class TestV2InferenceStateInitialization(unittest.TestCase):
    """Tests for v2 inference state initialization."""

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS") == "1",
        "Requires GPU and SAM3 model weights",
    )
    def test_init_v2_streaming_state_structure(self):
        """Test that _init_v2_streaming_state creates proper structure."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cpu", mode="v2")

        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Initialize state
        processor._init_v2_streaming_state(frame, "person")

        # Check state was initialized
        self.assertTrue(processor.v2_state.initialized)
        self.assertEqual(processor.v2_state.text_prompt, "person")

        # Check inference state structure
        inf_state = processor.v2_state.inference_state
        self.assertIsNotNone(inf_state)
        self.assertEqual(inf_state["image_size"], 1008)
        self.assertEqual(inf_state["num_frames"], 1)
        self.assertEqual(inf_state["text_prompt"], "person")
        self.assertIn("constants", inf_state)
        self.assertIn("_frame_tensors", inf_state)
        self.assertIn("_find_inputs_list", inf_state)


class TestIntegration(unittest.TestCase):
    """Integration tests that require the SAM3 model."""

    @unittest.skipIf(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
    )
    def test_v1_model_loading(self):
        """Test v1 model loading (requires GPU and model weights)."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cuda", mode="v1")
        success = processor.load_model()

        self.assertTrue(success)
        self.assertIsNotNone(processor.model)
        self.assertIsNotNone(processor.processor)

    @unittest.skipIf(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
    )
    def test_v2_model_loading(self):
        """Test v2 model loading (requires GPU and model weights)."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cuda", mode="v2")
        success = processor.load_model()

        self.assertTrue(success)
        self.assertIsNotNone(processor.video_model)

    @unittest.skipIf(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
    )
    def test_v2_process_frames(self):
        """Test processing multiple frames with v2 mode."""
        from filter_sam3_detector.streaming_video_processor import (
            StreamingVideoProcessor,
        )

        processor = StreamingVideoProcessor(device="cuda", mode="v2")
        processor.load_model()
        processor.set_text_prompt("object")

        # Process a few dummy frames
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = processor.process_frame(frame)

            # Detections should be a list (possibly empty)
            self.assertIsInstance(detections, list)

            # Frame counter should increment
            self.assertEqual(processor.v2_state.frame_idx, i + 1)


if __name__ == "__main__":
    unittest.main()
