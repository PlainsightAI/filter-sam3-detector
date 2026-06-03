from .filter import FilterSAM3Detector, FilterSAM3DetectorConfig, FilterSAM3DetectorOutput
from .confusion_detector import ConfusionDetector
from .temporal_intervals import (
    TemporalIntervalFilter,
    TemporalIntervalConfig,
    EMATracker,
    DetectionInterval,
    IntervalTracker,
)

__all__ = [
    "FilterSAM3Detector",
    "FilterSAM3DetectorConfig",
    "FilterSAM3DetectorOutput",
    # Cross-prompt overlap / confusion detection
    "ConfusionDetector",
    # Temporal interval support (standalone filter and reusable components)
    "TemporalIntervalFilter",
    "TemporalIntervalConfig",
    "EMATracker",
    "DetectionInterval",
    "IntervalTracker",
]
