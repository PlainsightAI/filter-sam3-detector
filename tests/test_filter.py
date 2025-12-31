"""Tests for FilterSAM3Detector."""

import pytest
import numpy as np

from filter_sam3_detector.filter import FilterSAM3Detector, FilterSAM3DetectorConfig


class TestFilterSAM3DetectorConfig:
    """Tests for configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FilterSAM3DetectorConfig()
        assert config.model_id == "facebook/sam2-hiera-large"
        assert config.device == "cuda"
        assert config.confidence_threshold == 0.5
        assert config.mask_threshold == 0.5
        assert config.max_detections == 100
        assert config.output_masks is True
        assert config.output_boxes is True
        assert config.output_scores is True
        assert config.output_label == "sam3_detections"
        assert config.debug is False
        assert config.visualize is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FilterSAM3DetectorConfig(
            text_prompt="person",
            confidence_threshold=0.7,
            device="cpu",
            max_detections=50,
        )
        assert config.text_prompt == "person"
        assert config.confidence_threshold == 0.7
        assert config.device == "cpu"
        assert config.max_detections == 50


class TestNormalizeConfig:
    """Tests for normalize_config validation."""

    def test_normalize_config_valid(self):
        """Test valid configuration passes normalization."""
        config = {
            'text_prompt': 'person',
            'confidence_threshold': 0.7,
            'device': 'cpu',
        }
        result = FilterSAM3Detector.normalize_config(config)
        assert isinstance(result, FilterSAM3DetectorConfig)
        assert result.text_prompt == 'person'
        assert result.confidence_threshold == 0.7
        assert result.device == 'cpu'

    def test_normalize_config_device_lowercase(self):
        """Test device is normalized to lowercase."""
        config = {'device': 'CUDA'}
        result = FilterSAM3Detector.normalize_config(config)
        assert result.device == 'cuda'

    def test_normalize_config_invalid_device(self):
        """Test invalid device raises ValueError."""
        config = {'device': 'tpu'}
        with pytest.raises(ValueError, match="Invalid device"):
            FilterSAM3Detector.normalize_config(config)

    def test_normalize_config_confidence_threshold_out_of_range(self):
        """Test confidence_threshold out of range raises ValueError."""
        config = {'confidence_threshold': 1.5}
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            FilterSAM3Detector.normalize_config(config)

        config = {'confidence_threshold': -0.1}
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            FilterSAM3Detector.normalize_config(config)

    def test_normalize_config_mask_threshold_out_of_range(self):
        """Test mask_threshold out of range raises ValueError."""
        config = {'mask_threshold': 2.0}
        with pytest.raises(ValueError, match="mask_threshold must be between 0 and 1"):
            FilterSAM3Detector.normalize_config(config)

    def test_normalize_config_max_detections_invalid(self):
        """Test max_detections < 1 raises ValueError."""
        config = {'max_detections': 0}
        with pytest.raises(ValueError, match="max_detections must be >= 1"):
            FilterSAM3Detector.normalize_config(config)


class TestProcessFrames:
    """Tests for frame processing (without model loaded)."""

    def test_process_no_prompt_warning(self):
        """Test that process logs warning when no prompt configured."""
        # This test verifies the filter handles missing prompts gracefully
        # Without SAM3 model loaded, it should forward frames unchanged
        pass  # Integration test - requires openfilter runtime


class TestVisualization:
    """Tests for detection visualization."""

    def test_visualize_detections_with_boxes(self):
        """Test visualization draws bounding boxes."""
        # Integration test - requires cv2 and actual frame
        pass
