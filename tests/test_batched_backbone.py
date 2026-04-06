import multiprocessing
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

try:
    multiprocessing.set_start_method('spawn')  # Required for CUDA compatibility
except RuntimeError:
    pass

from filter_sam3_detector.filter import FilterSAM3Detector
from openfilter.filter_runtime.filter import FilterConfig
from openfilter.filter_runtime.frame import Frame


def _make_frame(width=640, height=480):
    image_bgr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    frame = MagicMock(spec=Frame)
    frame.has_image = True
    rw_bgr = MagicMock()
    rw_bgr.image = image_bgr
    frame.rw_bgr = rw_bgr
    frame.data = {}
    return frame


def _make_frames_dict(width=640, height=480, topic="main"):
    return {topic: _make_frame(width, height)}


class TestCanBatch(unittest.TestCase):
    def _make_detector(self, **overrides):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        d.model = MagicMock()
        d.processor = MagicMock()
        d.prompt_sets = None
        d.positive_boxes = []
        d.negative_boxes = []
        d.ref_images_paths = None
        d.ref_images_negative_paths = None
        d.enable_video_mode = False
        d.text_prompt = "person"
        d.text_prompts = None
        d.visual_prompt_embed = None
        d._cached_backbone_state = None
        for k, v in overrides.items():
            setattr(d, k, v)
        return d

    def test_standard_mode_can_batch(self):
        d = self._make_detector()
        self.assertTrue(d._can_batch())

    def test_no_model_cannot_batch(self):
        d = self._make_detector(model=None)
        self.assertFalse(d._can_batch())

    def test_prompt_sets_cannot_batch(self):
        d = self._make_detector(prompt_sets=[{"name": "a", "prompts": ["x"]}])
        self.assertFalse(d._can_batch())

    def test_ref_boxes_cannot_batch(self):
        d = self._make_detector(positive_boxes=[[0, 0, 100, 100]])
        self.assertFalse(d._can_batch())

    def test_ref_images_cannot_batch(self):
        d = self._make_detector(ref_images_paths=["/some/path.jpg"])
        self.assertFalse(d._can_batch())

    def test_video_mode_cannot_batch(self):
        d = self._make_detector(enable_video_mode=True)
        self.assertFalse(d._can_batch())

    def test_no_prompts_or_visual_cannot_batch(self):
        d = self._make_detector(
            text_prompt=None, text_prompts=None, visual_prompt_embed=None
        )
        self.assertFalse(d._can_batch())

    def test_visual_embed_only_can_batch(self):
        d = self._make_detector(
            text_prompt=None, visual_prompt_embed=torch.randn(1, 256)
        )
        self.assertTrue(d._can_batch())


class TestSplitBackboneStates(unittest.TestCase):
    def _make_detector(self):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        return d

    def _make_batched_state(self, batch_size=4):
        return {
            "original_heights": [480] * batch_size,
            "original_widths": [640] * batch_size,
            "backbone_out": {
                "vision_features": torch.randn(batch_size, 256, 72, 72),
                "backbone_fpn": [
                    torch.randn(batch_size, 256, 288, 288),
                    torch.randn(batch_size, 256, 144, 144),
                    torch.randn(batch_size, 256, 72, 72),
                    torch.randn(batch_size, 256, 36, 36),
                ],
                "vision_pos_enc": [
                    torch.randn(batch_size, 256, 288, 288),
                ],
            },
        }

    def test_returns_n_states(self):
        d = self._make_detector()
        batched = self._make_batched_state(4)
        states = d._split_backbone_states(batched)
        self.assertEqual(len(states), 4)

    def test_each_state_has_batch_dim_1(self):
        d = self._make_detector()
        batched = self._make_batched_state(4)
        states = d._split_backbone_states(batched)
        for state in states:
            self.assertEqual(state["original_height"], 480)
            self.assertEqual(state["original_width"], 640)
            self.assertEqual(state["backbone_out"]["vision_features"].shape[0], 1)

    def test_fpn_list_split(self):
        d = self._make_detector()
        batched = self._make_batched_state(4)
        states = d._split_backbone_states(batched)
        for state in states:
            fpn = state["backbone_out"]["backbone_fpn"]
            self.assertEqual(len(fpn), 4)
            for tensor in fpn:
                self.assertEqual(tensor.shape[0], 1)

    def test_split_content_matches_original(self):
        d = self._make_detector()
        batched = self._make_batched_state(3)
        states = d._split_backbone_states(batched)
        for idx, state in enumerate(states):
            torch.testing.assert_close(
                state["backbone_out"]["vision_features"],
                batched["backbone_out"]["vision_features"][idx : idx + 1],
            )

    def test_handles_nested_dict(self):
        d = self._make_detector()
        batched = self._make_batched_state(2)
        batched["backbone_out"]["sam2_backbone_out"] = {
            "backbone_fpn": [torch.randn(2, 256, 64, 64)],
        }
        states = d._split_backbone_states(batched)
        for state in states:
            nested_fpn = state["backbone_out"]["sam2_backbone_out"]["backbone_fpn"]
            self.assertEqual(nested_fpn[0].shape[0], 1)


class TestExtractPilImage(unittest.TestCase):
    def _make_detector(self):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        return d

    def test_extracts_from_valid_frame(self):
        d = self._make_detector()
        frames = _make_frames_dict(640, 480)
        pil = d._extract_pil_image(frames)
        self.assertIsInstance(pil, Image.Image)
        self.assertEqual(pil.size, (640, 480))

    def test_returns_none_for_empty_frames(self):
        d = self._make_detector()
        self.assertIsNone(d._extract_pil_image({}))

    def test_returns_none_for_no_image(self):
        d = self._make_detector()
        frame = MagicMock(spec=Frame)
        frame.has_image = False
        self.assertIsNone(d._extract_pil_image({"main": frame}))


class TestProcessBatch(unittest.TestCase):
    def _make_detector(self):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        d.model = MagicMock()
        d.processor = MagicMock()
        d.prompt_sets = None
        d.positive_boxes = []
        d.negative_boxes = []
        d.ref_images_paths = None
        d.ref_images_negative_paths = None
        d.enable_video_mode = False
        d.text_prompt = "person"
        d.text_prompts = None
        d.visual_prompt_embed = None
        d._cached_backbone_state = None
        return d

    def test_fallback_when_cannot_batch(self):
        d = self._make_detector()
        d.model = None
        batch = [_make_frames_dict(), _make_frames_dict()]
        d.process = MagicMock(side_effect=[{"main": "r1"}, {"main": "r2"}])
        results = d.process_batch(batch)
        self.assertEqual(len(results), 2)
        self.assertEqual(d.process.call_count, 2)

    def test_calls_set_image_batch(self):
        d = self._make_detector()
        batched_state = {
            "original_heights": [480, 480],
            "original_widths": [640, 640],
            "backbone_out": {
                "vision_features": torch.randn(2, 256, 72, 72),
                "backbone_fpn": [torch.randn(2, 256, 72, 72)],
                "vision_pos_enc": [torch.randn(2, 256, 72, 72)],
            },
        }
        d.processor.set_image_batch = MagicMock(return_value=batched_state)
        d.process = MagicMock(return_value={"main": MagicMock()})

        batch = [_make_frames_dict(), _make_frames_dict()]
        results = d.process_batch(batch)

        d.processor.set_image_batch.assert_called_once()
        pil_images = d.processor.set_image_batch.call_args[0][0]
        self.assertEqual(len(pil_images), 2)
        self.assertIsInstance(pil_images[0], Image.Image)

    def test_injects_pre_split_state_per_frame(self):
        d = self._make_detector()
        vision = torch.randn(2, 256, 72, 72)
        batched_state = {
            "original_heights": [480, 480],
            "original_widths": [640, 640],
            "backbone_out": {
                "vision_features": vision,
                "backbone_fpn": [torch.randn(2, 256, 72, 72)],
                "vision_pos_enc": [torch.randn(2, 256, 72, 72)],
            },
        }
        d.processor.set_image_batch = MagicMock(return_value=batched_state)

        captured_states = []

        def capture_process(frames):
            captured_states.append(d._cached_backbone_state)
            return {"main": MagicMock()}

        d.process = MagicMock(side_effect=capture_process)

        batch = [_make_frames_dict(), _make_frames_dict()]
        d.process_batch(batch)

        self.assertEqual(len(captured_states), 2)
        for i, state in enumerate(captured_states):
            self.assertIsNotNone(state)
            self.assertIn("backbone_out", state)
            self.assertEqual(state["backbone_out"]["vision_features"].shape[0], 1)
            torch.testing.assert_close(
                state["backbone_out"]["vision_features"],
                vision[i : i + 1],
            )

        self.assertIsNone(d._cached_backbone_state)

    def test_clears_cached_state_on_per_frame_exception(self):
        d = self._make_detector()
        batched_state = {
            "original_heights": [480],
            "original_widths": [640],
            "backbone_out": {
                "vision_features": torch.randn(1, 256, 72, 72),
                "backbone_fpn": [torch.randn(1, 256, 72, 72)],
                "vision_pos_enc": [torch.randn(1, 256, 72, 72)],
            },
        }
        d.processor.set_image_batch = MagicMock(return_value=batched_state)
        d.process = MagicMock(side_effect=RuntimeError("boom"))

        batch = [_make_frames_dict()]
        results = d.process_batch(batch)

        self.assertIsNone(d._cached_backbone_state)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], batch[0])

    def test_fallback_on_set_image_batch_failure(self):
        d = self._make_detector()
        d.processor.set_image_batch = MagicMock(side_effect=RuntimeError("OOM"))
        d.process = MagicMock(return_value={"main": MagicMock()})

        batch = [_make_frames_dict(), _make_frames_dict()]
        results = d.process_batch(batch)

        self.assertEqual(len(results), 2)
        self.assertEqual(d.process.call_count, 2)
        self.assertIsNone(d._cached_backbone_state)

    def test_result_count_matches_batch_size(self):
        d = self._make_detector()
        batch_size = 5
        batched_state = {
            "original_heights": [480] * batch_size,
            "original_widths": [640] * batch_size,
            "backbone_out": {
                "vision_features": torch.randn(batch_size, 256, 72, 72),
                "backbone_fpn": [torch.randn(batch_size, 256, 72, 72)],
                "vision_pos_enc": [torch.randn(batch_size, 256, 72, 72)],
            },
        }
        d.processor.set_image_batch = MagicMock(return_value=batched_state)
        d.process = MagicMock(return_value={"main": MagicMock()})

        batch = [_make_frames_dict() for _ in range(batch_size)]
        results = d.process_batch(batch)

        self.assertEqual(len(results), batch_size)


if __name__ == "__main__":
    unittest.main()
