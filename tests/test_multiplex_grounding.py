"""FILTER-374: multiplexed multi-prompt grounding.

Two layers of testing:

1. A pure-unit test (mocked model) asserting the new _process_multi_output path fires
   EXACTLY ONE model.forward_grounding call per frame regardless of the number of prompts.

2. A GPU/model integration test (skipped when CUDA or HF weights are unavailable) that
   asserts detection EQUIVALENCE between the legacy per-prompt loop and the new
   multiplexed pass: for every kept baseline detection there is a matched multiplexed
   detection with IoU >= 0.95 and score diff <= 1e-2 (mirrors scripts/spike_multiplex.py).
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from filter_sam3_detector.filter import FilterSAM3Detector


REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_PATH = REPO_ROOT / "data" / "electrical_post.png"
PROMPTS = ["electrical post", "wire", "pole"]


# --------------------------------------------------------------------------- #
# Layer 1: call-count (mocked, no GPU needed)
# --------------------------------------------------------------------------- #
class TestMultiplexCallCount(unittest.TestCase):
    """One forward_grounding per frame regardless of N prompts."""

    def _make_detector(self, prompts):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        d.model = MagicMock()
        d.processor = MagicMock()
        d.device = "cpu"
        d.confidence_threshold = 0.5
        # cache one embedding per prompt (single-prompt-shaped tensors)
        d.cached_text_embeddings = {
            p: {
                "language_features": torch.zeros(4, 1, 8),
                "language_mask": torch.zeros(1, 4),
                "language_embeds": torch.zeros(4, 1, 8),
            }
            for p in prompts
        }
        return d

    def _fake_grounding_output(self, n, q=5):
        # raw decoder outputs for n slots, q queries each
        return {
            "pred_boxes": torch.rand(n, q, 4) * 0.5,
            "pred_logits": torch.full((n, q, 1), -10.0),  # all below threshold
            "pred_masks": torch.zeros(n, q, 8, 8),
            "presence_logit_dec": torch.zeros(n, 1),
        }

    def _run(self, n_prompts):
        prompts = PROMPTS[:n_prompts]
        d = self._make_detector(prompts)
        calls = {"n": 0}

        def grounding(*args, **kwargs):
            calls["n"] += 1
            n = len(kwargs["find_input"].text_ids)
            return self._fake_grounding_output(n)

        d.model.forward_grounding = MagicMock(side_effect=grounding)
        d.model._get_dummy_prompt = MagicMock(return_value=None)

        state = {
            "original_height": 480,
            "original_width": 640,
            "backbone_out": {"vision_features": torch.zeros(1, 8)},
        }
        out = d._forward_grounding_multi(state, prompts)
        self.assertEqual(len(out), n_prompts)
        return calls["n"]

    def test_one_call_for_n1(self):
        self.assertEqual(self._run(1), 1)

    def test_one_call_for_n2(self):
        self.assertEqual(self._run(2), 1)

    def test_one_call_for_n3(self):
        self.assertEqual(self._run(3), 1)


# --------------------------------------------------------------------------- #
# Layer 2: equivalence (real model, GPU)
# --------------------------------------------------------------------------- #
def _model_available():
    return torch.cuda.is_available() and IMG_PATH.exists()


def _box_iou_xyxy(a, b):
    inter_lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    inter_rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (inter_rb - inter_lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


@unittest.skipUnless(_model_available(), "CUDA + data/electrical_post.png required")
class TestMultiplexEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        cls.device = "cuda"
        bpe = REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        cls.model = build_sam3_image_model(
            bpe_path=str(bpe) if bpe.exists() else None,
            device=cls.device,
            eval_mode=True,
            load_from_HF=True,
        )
        cls.processor = Sam3Processor(cls.model, device=cls.device, confidence_threshold=0.5)
        cls.image = Image.open(IMG_PATH).convert("RGB")

    def _make_detector(self):
        d = FilterSAM3Detector.__new__(FilterSAM3Detector)
        d.model = self.model
        d.processor = self.processor
        d.device = self.device
        d.confidence_threshold = 0.5
        d.cached_text_embeddings = {}
        with torch.no_grad():
            for p in PROMPTS:
                t = self.model.backbone.forward_text([p], device=self.device)
                d.cached_text_embeddings[p] = {
                    "language_features": t.get("language_features"),
                    "language_mask": t.get("language_mask"),
                    "language_embeds": t.get("language_embeds"),
                }
        return d

    def _baseline(self, d):
        """Legacy per-prompt path: inject cached embedding + processor.forward_grounding."""
        results = []
        with torch.no_grad():
            for p in PROMPTS:
                state = self.processor.set_image(self.image)
                d_state = d._inject_cached_text_embedding(state, p)
                d_state = self.processor.forward_grounding(d_state)
                results.append((d_state["scores"].cpu(), d_state["boxes"].cpu()))
        return results

    def _multiplex(self, d):
        with torch.no_grad():
            state = self.processor.set_image(self.image)
            per = d._forward_grounding_multi(state, PROMPTS)
        return [(s["scores"].cpu(), s["boxes"].cpu()) for s in per]

    def test_equivalence(self):
        d = self._make_detector()
        base = self._baseline(d)
        mux = self._multiplex(d)
        self.assertEqual(len(base), len(mux))
        for i, ((sa, ba), (sb, bb)) in enumerate(zip(base, mux)):
            self.assertEqual(len(sa), len(sb), f"prompt {i} count mismatch")
            if len(sa) == 0:
                continue
            ious = _box_iou_xyxy(ba, bb)
            best_iou, best_j = ious.max(dim=1)
            score_diff = (sa - sb[best_j]).abs()
            self.assertGreater(best_iou.min().item(), 0.95, f"prompt {i} IoU")
            self.assertLess(score_diff.max().item(), 1e-2, f"prompt {i} score")


if __name__ == "__main__":
    unittest.main()
