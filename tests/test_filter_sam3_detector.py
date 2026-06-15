#!/usr/bin/env python

import logging
import os
import sys
import unittest

from filter_sam3_detector.filter import FilterSAM3Detector
from openfilter.filter_runtime.filter import FilterConfig

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv('LOG_LEVEL') or 'INFO').upper())))

VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv
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
                    "polygons": [{"points": [(10.0, 20.0), (30.0, 20.0), (30.0, 45.0)]}],
                    "area": 150
                }
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
        config = FilterSAM3Detector.normalize_config(FilterConfig({
            "output_boxes": False,
            "output_scores": True,
        }))
        detector = FilterSAM3Detector(config)
        
        with self.assertRaises(ValueError) as context:
            detector.setup(config)
            
        self.assertIn("output_boxes and output_scores must both be True", str(context.exception))

    def test_temporal_intervals_fallback(self):
        """Test that _aggregate_detections correctly falls back to standard schema keys."""
        from filter_sam3_detector.temporal_intervals import TemporalIntervalFilter
        
        config = TemporalIntervalFilter.normalize_config({
            'label_field': 'class',
            'score_field': 'confidence',
        })
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
                        {"points": [(10.0, 20.0), (30.0, 20.0)]},  # Degenerate: only 2 points
                        {"points": [(10.0, 20.0), (30.0, 20.0), (30.0, 45.0)]}  # Valid: 3 points
                    ],
                    "area": 150
                }
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
                "prompt": None
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
                "mask": np.zeros((100, 100), dtype=np.uint8)  # Binary array mask
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
                        [(10.0, 20.0), (30.0, 20.0), (30.0, 45.0)]  # Direct list of points
                    ],
                    "area": 150
                }
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
                "label_id": "not-an-integer"
            }
        ]
        
        detector = FilterSAM3Detector(FilterConfig({}))
        canonical, _, _ = detector._normalize_detections(detections)
        
        self.assertIsNone(canonical["items"][0]["label_id"])


if __name__ == '__main__':
    unittest.main()
