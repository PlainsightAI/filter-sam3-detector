#!/usr/bin/env python
"""FILTER-349: Verify prompt_sets (multi-output) mode saves original and annotated frames.

This is a targeted regression test. It patches the heavy SAM3 machinery and
validates that _process_multi_output actually calls cv2.imwrite for both
original frames (once) and annotated frames (per prompt set).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestPromptSetsFrameSaving(unittest.TestCase):
    """Regression: prompt_sets mode must save original + annotated frames."""

    def _make_detector_stub(self, frames_dir=None, annotated_frames_dir=None):
        """Build a minimal object with the attributes _process_multi_output reads."""
        det = MagicMock()
        # Instance attributes read by _process_multi_output
        det.frames_dir = Path(frames_dir) if frames_dir else None
        det.annotated_frames_dir = (
            Path(annotated_frames_dir) if annotated_frames_dir else None
        )
        det.positive_boxes = []
        det.negative_boxes = []
        det.output_label = "sam3_detections"
        det.confidence_threshold = 0.5
        det.max_detections = 100
        det.frame_counter = 0
        det.global_detection_id = 0
        det.prompt_sets = [
            {"name": "people", "prompts": ["person"], "topic": "people_topic"},
            {"name": "vehicles", "prompts": ["car"], "topic": "vehicles_topic"},
        ]
        det.cached_text_embeddings = {"person": "emb_person", "car": "emb_car"}
        det.jsonl_file = None  # no JSONL for this test

        # _inject_cached_text_embedding: return whatever state was passed in
        det._inject_cached_text_embedding = MagicMock(
            side_effect=lambda state, prompt: state
        )

        # _extract_detections_from_state: return one fake detection per prompt
        def fake_extract(
            state, label, w, h, gid, confidence_threshold=0.5, max_detections=100
        ):
            return [
                {
                    "label": label,
                    "box": [10, 20, 30, 40],
                    "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                    "score": 0.9,
                }
            ]

        det._extract_detections_from_state = MagicMock(side_effect=fake_extract)
        det._normalize_detections = MagicMock(return_value=({"items": []}, [], {}))

        # _visualize_detections_on_image: return the image unchanged
        det._visualize_detections_on_image = MagicMock(
            side_effect=lambda img, *a, **kw: img
        )

        # processor.set_image / forward_grounding: return a dummy state
        det.processor = MagicMock()
        det.processor.set_image = MagicMock(return_value="state")
        det.processor.forward_grounding = MagicMock(return_value="state")

        return det

    def _make_frame(self):
        """Create a minimal frame-like object with rw_bgr.image and data dict."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = MagicMock()
        frame.rw_bgr.image = image
        frame.data = {"meta": {}}
        frame.timestamp = 1234.567
        return frame

    @patch("cv2.imwrite", return_value=True)
    def test_saves_original_frame_once(self, mock_imwrite):
        """Original frame is saved exactly once (before the prompt set loop)."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir)

            det = self._make_detector_stub(frames_dir=frames_dir)
            frame = self._make_frame()

            # Call the real method on our stub
            _ = FilterSAM3Detector._process_multi_output(det, frame, filter_frame_id=42)

            # Filter imwrite calls to the frames_dir (original frames)
            original_writes = [
                c for c in mock_imwrite.call_args_list if frames_dir in str(c)
            ]
            self.assertEqual(
                len(original_writes),
                1,
                f"Expected 1 original frame write, got {len(original_writes)}: {original_writes}",
            )

    @patch("cv2.imwrite", return_value=True)
    def test_saves_annotated_frame_per_prompt_set(self, mock_imwrite):
        """Annotated frame is saved once per prompt set with detections."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        with tempfile.TemporaryDirectory() as tmpdir:
            annotated_dir = os.path.join(tmpdir, "annotated")
            os.makedirs(annotated_dir)

            det = self._make_detector_stub(annotated_frames_dir=annotated_dir)
            frame = self._make_frame()

            _ = FilterSAM3Detector._process_multi_output(det, frame, filter_frame_id=42)

            # Filter imwrite calls to the annotated_dir
            annotated_writes = [
                c for c in mock_imwrite.call_args_list if annotated_dir in str(c)
            ]
            # 2 prompt sets, each with detections -> 2 annotated writes
            self.assertEqual(
                len(annotated_writes),
                2,
                f"Expected 2 annotated frame writes, got {len(annotated_writes)}: {annotated_writes}",
            )

            # Filenames must include the prompt set name for disambiguation
            written_paths = [str(c[0][0]) for c in annotated_writes]
            self.assertTrue(
                any("_people" in p for p in written_paths),
                f"Missing 'people' set in filenames: {written_paths}",
            )
            self.assertTrue(
                any("_vehicles" in p for p in written_paths),
                f"Missing 'vehicles' set in filenames: {written_paths}",
            )

    @patch("cv2.imwrite", return_value=True)
    def test_no_writes_when_dirs_not_configured(self, mock_imwrite):
        """No frame writes when neither frames_dir nor annotated_frames_dir is set."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        det = self._make_detector_stub(frames_dir=None, annotated_frames_dir=None)
        frame = self._make_frame()

        result = FilterSAM3Detector._process_multi_output(
            det, frame, filter_frame_id=42
        )

        mock_imwrite.assert_not_called()
        # Should still produce output frames for both topics
        self.assertIn("people_topic", result)
        self.assertIn("vehicles_topic", result)

    @patch("cv2.imwrite", return_value=True)
    def test_annotated_not_saved_when_no_detections(self, mock_imwrite):
        """Annotated frame is skipped for a prompt set with zero detections."""
        from filter_sam3_detector.filter import FilterSAM3Detector

        with tempfile.TemporaryDirectory() as tmpdir:
            annotated_dir = os.path.join(tmpdir, "annotated")
            os.makedirs(annotated_dir)

            det = self._make_detector_stub(annotated_frames_dir=annotated_dir)
            # Make extract return empty for the second prompt set ("car")
            call_count = [0]

            def selective_extract(
                state, label, w, h, gid, confidence_threshold=0.5, max_detections=100
            ):
                call_count[0] += 1
                if label == "car":
                    return []
                return [
                    {
                        "label": label,
                        "box": [10, 20, 30, 40],
                        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                        "score": 0.9,
                    }
                ]

            det._extract_detections_from_state = MagicMock(
                side_effect=selective_extract
            )

            frame = self._make_frame()
            _ = FilterSAM3Detector._process_multi_output(det, frame, filter_frame_id=42)

            annotated_writes = [
                c for c in mock_imwrite.call_args_list if annotated_dir in str(c)
            ]
            # Only 1 annotated write (for "people" set; "vehicles" had no detections)
            self.assertEqual(
                len(annotated_writes),
                1,
                f"Expected 1 annotated write, got {len(annotated_writes)}",
            )
            self.assertIn("_people", str(annotated_writes[0]))

    def test_process_multi_output_handles_exception_gracefully(self):
        """Verify that _process_multi_output handles prompt set exceptions gracefully,
        yielding degraded frames with appropriate empty structures for both canonical
        and legacy outputs, and preserving topic cardinality.
        """
        from filter_sam3_detector.filter import FilterSAM3Detector

        det = self._make_detector_stub()

        # Mock _extract_detections_from_state to raise an error for the 'car' label
        def selective_error_extract(
            state, label, w, h, gid, confidence_threshold=0.5, max_detections=100
        ):
            if label == "car":
                raise ValueError("Simulated processing error for car")
            return [
                {
                    "label": label,
                    "box": [10, 20, 30, 40],
                    "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                    "score": 0.9,
                }
            ]

        det._extract_detections_from_state = MagicMock(
            side_effect=selective_error_extract
        )
        # Mock the real _normalize_detections so it actually processes [] and [d] properly
        # inside our test instead of being a hardcoded stub
        det._normalize_detections = FilterSAM3Detector._normalize_detections.__get__(
            det, FilterSAM3Detector
        )

        frame = self._make_frame()
        result = FilterSAM3Detector._process_multi_output(
            det, frame, filter_frame_id=42
        )

        # Confirm we still have both topics returned (cardinality preserved)
        self.assertIn("people_topic", result)
        self.assertIn("vehicles_topic", result)

        # "people" should have succeeded and got its detections (due to our mocked success return)
        people_frame = result["people_topic"]
        self.assertIn("detections", people_frame.data)
        self.assertEqual(len(people_frame.data["detections"]["items"]), 1)
        self.assertEqual(people_frame.data["meta"]["detections"][0]["confidence"], 0.9)

        # "vehicles" should have hit the exception handler and generated a degraded frame
        vehicles_frame = result["vehicles_topic"]
        self.assertIn("detections", vehicles_frame.data)
        self.assertEqual(vehicles_frame.data["detections"]["items"], [])
        self.assertEqual(vehicles_frame.data["meta"]["detections"], [])
        self.assertEqual(
            vehicles_frame.data["meta"]["classification"],
            {"classes": [], "confidences": [], "architecture": "sam3"},
        )
        self.assertEqual(vehicles_frame.data["meta"][det.output_label], [])


if __name__ == "__main__":
    unittest.main()
