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
        
        self.assertEqual(config["model_id"], "facebook/sam2-hiera-large")
        # Default is "cuda" but we don't test actual GPU usage in CI
        self.assertIn(config["device"], ["cuda", "cpu", "mps"])
        self.assertIsNone(config["text_prompt"])
        self.assertIsNone(config["exemplars_path"])
        self.assertEqual(config["confidence_threshold"], 0.5)
        self.assertEqual(config["mask_threshold"], 0.5)
        self.assertEqual(config["max_detections"], 100)
        self.assertTrue(config["output_masks"])
        self.assertTrue(config["output_boxes"])
        self.assertTrue(config["output_scores"])
        self.assertEqual(config["output_label"], "sam3_detections")
        self.assertFalse(config["debug"])
        self.assertFalse(config["visualize"])


if __name__ == '__main__':
    unittest.main()
