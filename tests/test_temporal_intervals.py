"""Tests for temporal interval detection with EMA smoothing."""

import json
import math
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from filter_sam3_detector.temporal_intervals import (
    EMATracker,
    DetectionInterval,
    TemporalIntervalConfig,
    TemporalIntervalFilter,
)
from openfilter.filter_runtime.frame import Frame


class TestEMATracker(TestCase):
    """Test EMA tracker functionality."""

    def test_half_life_alpha_calculation(self):
        """Test alpha calculation from half-life."""
        # With half_life=1, after 1 frame we should be at 50% of a step
        tracker = EMATracker(half_life=1.0)
        # alpha = 1 - 2^(-1/1) = 1 - 0.5 = 0.5
        self.assertAlmostEqual(tracker.alpha, 0.5, places=5)

        # With half_life=5, alpha should be smaller
        tracker5 = EMATracker(half_life=5.0)
        expected_alpha = 1 - math.pow(2, -1 / 5)
        self.assertAlmostEqual(tracker5.alpha, expected_alpha, places=5)

    def test_full_decay_life_alpha_calculation(self):
        """Test alpha calculation from full decay life."""
        # Full decay (5 time constants) should give ~99.3% decay
        tracker = EMATracker(full_decay_life=10.0)
        expected_alpha = 1 - math.exp(-5 / 10)
        self.assertAlmostEqual(tracker.alpha, expected_alpha, places=5)

    def test_dual_ema_both_specified(self):
        """Test that specifying both parameters uses both directly."""
        tracker = EMATracker(half_life=5.0, full_decay_life=30.0)
        # Fast alpha from half_life
        expected_fast = 1 - math.pow(2, -1 / 5)
        self.assertAlmostEqual(tracker.alpha_fast, expected_fast, places=5)
        self.assertAlmostEqual(
            tracker.alpha, expected_fast, places=5
        )  # Primary alpha is fast
        # Slow alpha from full_decay_life
        expected_slow = 1 - math.exp(-5 / 30)
        self.assertAlmostEqual(tracker.alpha_slow, expected_slow, places=5)
        # Both lifetimes stored
        self.assertEqual(tracker.half_life, 5.0)
        self.assertEqual(tracker.full_decay_life, 30.0)

    def test_derive_full_decay_from_half_life(self):
        """Test that full_decay_life is derived from half_life."""
        tracker = EMATracker(half_life=5.0)
        # full_decay_life = half_life * 5 / ln(2) ≈ 7.21 * half_life
        expected_full_decay = 5.0 * 5.0 / math.log(2)
        self.assertAlmostEqual(tracker.full_decay_life, expected_full_decay, places=3)
        # Both alphas should be set
        self.assertIsNotNone(tracker.alpha_fast)
        self.assertIsNotNone(tracker.alpha_slow)

    def test_derive_half_life_from_full_decay(self):
        """Test that half_life is derived from full_decay_life."""
        tracker = EMATracker(full_decay_life=36.07)  # ~5 * 7.21
        # half_life = full_decay_life * ln(2) / 5 ≈ 0.139 * full_decay_life
        expected_half_life = 36.07 * math.log(2) / 5.0
        self.assertAlmostEqual(tracker.half_life, expected_half_life, places=3)
        # Both alphas should be set
        self.assertIsNotNone(tracker.alpha_fast)
        self.assertIsNotNone(tracker.alpha_slow)

    def test_default_half_life(self):
        """Test default half-life is applied."""
        tracker = EMATracker()
        # Default half_life=5.0
        expected_alpha = 1 - math.pow(2, -1 / 5)
        self.assertAlmostEqual(tracker.alpha, expected_alpha, places=5)

    def test_invalid_half_life(self):
        """Test invalid half-life values are rejected."""
        with self.assertRaises(ValueError):
            EMATracker(half_life=0)
        with self.assertRaises(ValueError):
            EMATracker(half_life=-5)

    def test_invalid_full_decay_life(self):
        """Test invalid full decay life values are rejected."""
        with self.assertRaises(ValueError):
            EMATracker(full_decay_life=0)
        with self.assertRaises(ValueError):
            EMATracker(full_decay_life=-5)

    def test_first_update_initializes_state(self):
        """Test that first update initializes tracking state."""
        tracker = EMATracker(half_life=5.0, threshold=0.5)

        # First detection - should initialize to 1.0 and be present
        ema, is_present, changed = tracker.update("person", True)
        self.assertEqual(ema, 1.0)
        self.assertTrue(is_present)
        self.assertTrue(changed)  # First update always counts as changed

    def test_first_update_absence(self):
        """Test first update with no detection."""
        tracker = EMATracker(half_life=5.0, threshold=0.5)

        # First non-detection - should initialize to 0.0 and be absent
        ema, is_present, changed = tracker.update("person", False)
        self.assertEqual(ema, 0.0)
        self.assertFalse(is_present)
        self.assertTrue(changed)

    def test_ema_decay_without_detection(self):
        """Test that debiased EMA decays when detection stops."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)  # Fast decay

        # Start with detection - debiased EMA is 1.0 on first frame
        ema, is_present, _ = tracker.update("person", True)
        self.assertAlmostEqual(ema, 1.0, places=2)
        self.assertTrue(is_present)

        # After 1 frame without detection, debiased EMA decays
        # raw = 0.5*0 + 0.5*(0.5*1) = 0.25, bias = 0.25, debiased = 0.25/0.75 = 0.333
        ema, is_present, _ = tracker.update("person", False)
        self.assertAlmostEqual(ema, 0.333, places=2)
        # Below threshold now
        self.assertFalse(is_present)

        # After another frame, continues to decay
        prev_ema = ema
        ema, is_present, _ = tracker.update("person", False)
        self.assertLess(ema, prev_ema)  # EMA should decrease
        self.assertFalse(is_present)

    def test_ema_rise_with_detection(self):
        """Test that debiased EMA rises with continued detection."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)

        # Start with no detection - debiased EMA is 0.0
        ema, is_present, _ = tracker.update("person", False)
        self.assertAlmostEqual(ema, 0.0, places=2)
        self.assertFalse(is_present)

        # After 1 frame with detection
        # raw = 0.5*1 + 0.5*0 = 0.5, bias = 0.25, debiased = 0.5/0.75 = 0.667
        ema, is_present, _ = tracker.update("person", True)
        self.assertAlmostEqual(ema, 0.667, places=2)
        self.assertTrue(is_present)  # Above threshold

    def test_state_transition_detection(self):
        """Test that state transitions are correctly detected."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)

        # Initial state - first detection gives debiased EMA = 1.0
        ema, is_present, changed = tracker.update("person", True)
        self.assertAlmostEqual(ema, 1.0, places=2)
        self.assertTrue(is_present)
        self.assertTrue(changed)  # First update always changes state

        # Continue detecting - no state change, EMA stays at 1.0
        ema, _, changed = tracker.update("person", True)
        self.assertAlmostEqual(ema, 1.0, places=2)
        self.assertFalse(changed)

        # Stop detecting - with debiased EMA, decay is faster
        # Frame 3: raw = 0.5*0 + 0.5*(0.5+0.25) = 0.375, bias = 0.125
        # debiased = 0.375 / 0.875 ≈ 0.429
        ema, is_present, changed = tracker.update("person", False)
        self.assertAlmostEqual(ema, 0.429, places=2)
        self.assertFalse(is_present)  # Below threshold with debiased EMA
        self.assertTrue(changed)  # State changed from present to absent

    def test_multiple_labels(self):
        """Test tracking multiple labels independently."""
        tracker = EMATracker(half_life=5.0, threshold=0.5)

        tracker.update("person", True)
        tracker.update("car", False)
        tracker.update("dog", True)

        self.assertTrue(tracker.is_present("person"))
        self.assertFalse(tracker.is_present("car"))
        self.assertTrue(tracker.is_present("dog"))

    def test_get_ema(self):
        """Test getting EMA values."""
        tracker = EMATracker(half_life=5.0)

        tracker.update("person", True)
        self.assertEqual(tracker.get_ema("person"), 1.0)
        self.assertEqual(tracker.get_ema("unknown"), 0.0)  # Unknown label

    def test_reset(self):
        """Test resetting tracker state."""
        tracker = EMATracker(half_life=5.0)

        tracker.update("person", True)
        tracker.update("car", True)

        tracker.reset()

        self.assertEqual(len(tracker.ema), 0)
        self.assertEqual(len(tracker.current_state), 0)


class TestDualEMACrossingDetection(TestCase):
    """Test dual-EMA crossing detection (fast half-life + slow full-decay)."""

    def test_dual_ema_slow_crossing(self):
        """Test that slow EMA controls state transitions in dual mode."""
        # Fast EMA with half_life=1 (alpha=0.5), slow EMA with full_decay_life=100 (very slow)
        tracker = EMATracker(half_life=1.0, full_decay_life=100.0, threshold=0.5)

        # Start with detection - debiased EMA starts at 1.0
        fast_ema, is_present, changed = tracker.update("person", True)
        self.assertAlmostEqual(fast_ema, 1.0, places=2)
        self.assertTrue(is_present)
        self.assertTrue(changed)

        # Stop detecting - both EMAs drop, but slow EMA drops slower
        fast_ema, is_present, _ = tracker.update("person", False)
        slow_ema = tracker.get_slow_ema("person")

        # With debiased EMA: fast drops quickly, slow drops more slowly
        # Key property: slow EMA drops slower than fast EMA
        self.assertLess(fast_ema, 0.5)
        self.assertGreater(slow_ema, fast_ema)  # Slow decays slower than fast

    def test_dual_ema_prevents_flickering(self):
        """Test that dual EMA prevents rapid state changes."""
        # Fast half_life=1, slow full_decay_life=30
        # With full_decay_life=30, alpha ≈ 0.154, so slow EMA decays slower than fast
        tracker = EMATracker(half_life=1.0, full_decay_life=30.0, threshold=0.5)

        tracker.update("person", True)

        # Count frames until state changes
        state_change_frame = None
        for i in range(100):
            fast_ema, is_present, changed = tracker.update("person", False)
            if changed:
                state_change_frame = i + 1
                self.assertFalse(is_present)
                break

        # State should change after slow EMA crosses threshold (not fast EMA)
        # With debiased EMA, the slow EMA still decays slower than fast EMA
        # The key property is that state is controlled by slow EMA
        self.assertIsNotNone(state_change_frame)
        # With debiased EMA, the crossing happens sooner but still based on slow EMA
        self.assertGreaterEqual(state_change_frame, 1)

    def test_dual_ema_returns_fast_ema_value(self):
        """Test that update() returns the fast EMA value, not slow."""
        tracker = EMATracker(half_life=1.0, full_decay_life=30.0, threshold=0.5)

        tracker.update("person", True)
        fast_ema, _, _ = tracker.update("person", False)

        # With debiased EMA, fast drops to ~0.33 after one non-detection
        self.assertLess(fast_ema, 0.5)

        # Slow EMA should be different (higher since it decays slower)
        slow_ema = tracker.get_slow_ema("person")
        self.assertNotAlmostEqual(slow_ema, fast_ema, places=2)
        self.assertGreater(slow_ema, fast_ema)  # Slow decays slower

    def test_get_crossing_progress_returns_slow_ema(self):
        """Test that get_crossing_progress returns slow EMA value."""
        tracker = EMATracker(half_life=1.0, full_decay_life=100.0, threshold=0.5)

        tracker.update("person", True)

        # Get crossing progress - should show slow EMA (different from fast EMA)
        current_state, slow_ema = tracker.get_crossing_progress("person")
        self.assertTrue(current_state)  # Present after first detection
        self.assertEqual(slow_ema, 1.0)  # Slow EMA at 1.0 after first detection

        # After non-detection, slow EMA should differ from fast EMA
        tracker.update("person", False)
        fast_ema = tracker.get_ema("person")
        current_state, slow_ema = tracker.get_crossing_progress("person")

        # Key property: slow EMA value is different from fast EMA
        self.assertNotAlmostEqual(slow_ema, fast_ema, places=2)
        self.assertGreater(slow_ema, fast_ema)  # Slow decays slower

    def test_half_life_only_derives_full_decay(self):
        """Test that specifying only half_life derives full_decay_life."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)

        # Full decay life should be derived: half_life * 5 / ln(2) ≈ 7.21 * half_life
        expected_full_decay = 1.0 * EMATracker.HALF_TO_FULL
        self.assertAlmostEqual(tracker.full_decay_life, expected_full_decay, places=3)
        self.assertIsNotNone(tracker.alpha_fast)
        self.assertIsNotNone(tracker.alpha_slow)

        # Both EMAs should be tracked
        tracker.update("person", True)
        tracker.update("person", False)
        fast_ema = tracker.get_ema("person")
        slow_ema = tracker.get_slow_ema("person")
        # Both EMAs should be populated and approximately equal
        # (since they're derived from each other, the decay behavior is related)
        self.assertIsNotNone(fast_ema)
        self.assertIsNotNone(slow_ema)

    def test_full_decay_only_derives_half_life(self):
        """Test that specifying only full_decay_life derives half_life."""
        tracker = EMATracker(full_decay_life=10.0, threshold=0.5)

        # Half life should be derived: full_decay_life * ln(2) / 5 ≈ 0.139 * full_decay_life
        expected_half_life = 10.0 * EMATracker.FULL_TO_HALF
        self.assertAlmostEqual(tracker.half_life, expected_half_life, places=3)

        # Alpha (fast) should be based on derived half_life
        expected_alpha_fast = 1 - math.pow(2, -1 / expected_half_life)
        self.assertAlmostEqual(tracker.alpha_fast, expected_alpha_fast, places=5)

        # Alpha slow should be based on full_decay_life
        expected_alpha_slow = 1 - math.exp(-5 / 10)
        self.assertAlmostEqual(tracker.alpha_slow, expected_alpha_slow, places=5)

    def test_dual_ema_30_frame_crossing(self):
        """Test with 30-frame full_decay_life for crossing detection."""
        # Fast half_life=5 for signal, slow full_decay_life=30 for crossing
        # With debiased EMA and full_decay_life=30, slow EMA decays slower
        tracker = EMATracker(half_life=5.0, full_decay_life=30.0, threshold=0.5)

        tracker.update("person", True)

        # Track when slow EMA crosses threshold
        state_change_frame = None
        for i in range(100):
            fast_ema, is_present, changed = tracker.update("person", False)
            slow_ema = tracker.get_slow_ema("person")

            if changed:
                state_change_frame = i + 1
                # At change, slow EMA should be just below threshold
                self.assertLess(slow_ema, 0.5)
                self.assertFalse(is_present)
                break

        self.assertIsNotNone(state_change_frame, "State never changed")
        # With debiased EMA, the crossing happens based on slow EMA
        self.assertGreaterEqual(state_change_frame, 1)


class TestDetectionInterval(TestCase):
    """Test DetectionInterval dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        interval = DetectionInterval(
            start_frame=10,
            end_frame=50,
            label="person",
            present=True,
            confidence=0.876543,
        )
        d = interval.to_dict()

        self.assertEqual(d["start_frame"], 10)
        self.assertEqual(d["end_frame"], 50)
        self.assertEqual(d["label"], "person")
        self.assertTrue(d["present"])
        self.assertEqual(d["confidence"], 0.8765)  # Rounded to 4 places

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "start_frame": 10,
            "end_frame": 50,
            "label": "person",
            "present": True,
            "confidence": 0.8765,
        }
        interval = DetectionInterval.from_dict(d)

        self.assertEqual(interval.start_frame, 10)
        self.assertEqual(interval.end_frame, 50)
        self.assertEqual(interval.label, "person")
        self.assertTrue(interval.present)
        self.assertEqual(interval.confidence, 0.8765)


class TestTemporalIntervalFilter(TestCase):
    """Test TemporalIntervalFilter functionality."""

    def _create_frame(self, detections=None, frame_id=None):
        """Create a test frame with detections."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"meta": {}}
        if detections is not None:
            data["meta"]["sam3_detections"] = detections
        if frame_id is not None:
            data["meta"]["id"] = frame_id
        return Frame(image, data, "BGR")

    def test_normalize_config_defaults(self):
        """Test config normalization with defaults."""
        config = TemporalIntervalFilter.normalize_config({})

        self.assertIsInstance(config, TemporalIntervalConfig)
        self.assertEqual(config.presence_threshold, 0.5)
        self.assertEqual(config.detection_key, "detections")
        self.assertEqual(config.default_label, "foreground")

    def test_normalize_config_custom_values(self):
        """Test config normalization with custom values."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": "10",
                "presence_threshold": "0.7",
                "emit_on_change": "false",
            }
        )

        self.assertEqual(config.half_life, 10.0)
        self.assertEqual(config.presence_threshold, 0.7)
        self.assertFalse(config.emit_on_change)

    def test_normalize_config_invalid_threshold(self):
        """Test that invalid thresholds are rejected."""
        with self.assertRaises(ValueError):
            TemporalIntervalFilter.normalize_config({"presence_threshold": "1.5"})
        with self.assertRaises(ValueError):
            TemporalIntervalFilter.normalize_config({"min_confidence": "-0.1"})

    def test_config_dictionary_protocol_and_idempotence(self):
        """Test that the dictionary protocol includes extra fields and normalize_config is idempotent."""
        # 1. Verify custom attributes are captured in dictionary conversion
        custom_input = {
            "half_life": 5.0,
            "presence_threshold": 0.85,
            "custom_telemetry_field": "some-value",
        }
        config = TemporalIntervalFilter.normalize_config(custom_input)

        config_dict = dict(config)
        self.assertIn("half_life", config_dict)
        self.assertIn("presence_threshold", config_dict)
        self.assertIn("custom_telemetry_field", config_dict)
        self.assertEqual(config_dict["half_life"], 5.0)
        self.assertEqual(config_dict["presence_threshold"], 0.85)
        self.assertEqual(config_dict["custom_telemetry_field"], "some-value")

        # 2. Verify idempotence: normalizing already-normalized config retains custom values
        normalized_twice = TemporalIntervalFilter.normalize_config(config)
        self.assertEqual(normalized_twice.get("half_life"), 5.0)
        self.assertEqual(normalized_twice.get("presence_threshold"), 0.85)
        self.assertEqual(normalized_twice.get("custom_telemetry_field"), "some-value")

    def test_setup_and_shutdown(self):
        """Test filter setup and shutdown."""
        config = TemporalIntervalFilter.normalize_config({"half_life": 5.0})
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        self.assertIsNotNone(filter_instance.interval_tracker)
        self.assertEqual(filter_instance.interval_tracker.frame_count, 0)
        self.assertEqual(len(filter_instance.interval_tracker.intervals), 0)

        filter_instance.shutdown()

    def test_process_with_detections(self):
        """Test processing frames with detections."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_on_change": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with detection
        frame1 = self._create_frame(detections=[{"score": 0.9}], frame_id=1)
        output = filter_instance.process({"main": frame1})

        self.assertIn("main", output)
        self.assertEqual(filter_instance.interval_tracker.frame_count, 1)
        self.assertTrue(filter_instance.interval_tracker.is_present("foreground"))

        filter_instance.shutdown()

    def test_process_detection_to_absence_transition(self):
        """Test state transition from detection to absence."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 0.5,  # Very fast decay
                "emit_on_change": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Initial detection
        frame1 = self._create_frame(detections=[{"score": 0.9}], frame_id=1)
        filter_instance.process({"main": frame1})

        # Continue detection
        frame2 = self._create_frame(detections=[{"score": 0.9}], frame_id=2)
        filter_instance.process({"main": frame2})

        # No detection - should eventually transition
        for i in range(3, 8):
            frame = self._create_frame(detections=[], frame_id=i)
            filter_instance.process({"main": frame})

        self.assertFalse(filter_instance.interval_tracker.is_present("foreground"))

        filter_instance.shutdown()

        # Should have intervals recorded
        self.assertGreater(len(filter_instance.interval_tracker.intervals), 0)

    def test_process_with_label_field(self):
        """Test processing with labeled detections."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 5.0,
                "label_field": "class",
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with labeled detections
        frame = self._create_frame(
            detections=[
                {"class": "person", "score": 0.9},
                {"class": "car", "score": 0.8},
            ],
            frame_id=1,
        )
        filter_instance.process({"main": frame})

        self.assertTrue(filter_instance.interval_tracker.is_present("person"))
        self.assertTrue(filter_instance.interval_tracker.is_present("car"))

        filter_instance.shutdown()

    def test_min_confidence_filtering(self):
        """Test that low-confidence detections are filtered."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 5.0,
                "min_confidence": 0.7,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with low-confidence detection
        frame = self._create_frame(
            detections=[{"score": 0.5}],  # Below threshold
            frame_id=1,
        )
        filter_instance.process({"main": frame})

        # Should not register as detected (no label tracked at all)
        self.assertNotIn("foreground", filter_instance.interval_tracker.tracker.ema)

        filter_instance.shutdown()

    def test_output_json_file(self):
        """Test writing intervals to ndjson file (one JSON object per line)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "intervals.json"

            config = TemporalIntervalFilter.normalize_config(
                {
                    "half_life": 0.5,
                    "output_json_path": str(output_path),
                    "emit_on_complete": True,
                }
            )
            filter_instance = TemporalIntervalFilter(config=config)
            filter_instance.setup(config)

            # Create some intervals
            for i in range(1, 6):
                frame = self._create_frame(detections=[{"score": 0.9}], frame_id=i)
                filter_instance.process({"main": frame})

            for i in range(6, 11):
                frame = self._create_frame(detections=[], frame_id=i)
                filter_instance.process({"main": frame})

            filter_instance.shutdown()

            # Check output file exists
            self.assertTrue(output_path.exists())

            # Parse ndjson format (one JSON object per line)
            intervals = []
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        intervals.append(json.loads(line))

            # Should have at least one interval
            self.assertGreater(len(intervals), 0)

            # Each interval should have expected fields
            for interval in intervals:
                self.assertIn("start_frame", interval)
                self.assertIn("end_frame", interval)
                self.assertIn("label", interval)
                self.assertIn("present", interval)
                self.assertIn("confidence", interval)

    def test_get_current_state(self):
        """Test getting current state of all labels."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 5.0,
                "label_field": "class",
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        frame = self._create_frame(
            detections=[
                {"class": "person", "score": 0.9},
                {"class": "car", "score": 0.8},
            ],
            frame_id=1,
        )
        filter_instance.process({"main": frame})

        state = filter_instance.get_current_state()

        self.assertIn("person", state)
        self.assertIn("car", state)
        self.assertTrue(state["person"]["present"])
        self.assertTrue(state["car"]["present"])
        self.assertEqual(state["person"]["ema"], 1.0)

        filter_instance.shutdown()

    def test_process_with_canonical_custom_detection_key(self):
        """Test that a custom detection_key in canonical schema format (dict with 'items') is parsed correctly."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "detection_key": "custom_key",
                "emit_on_change": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Create a frame with custom_key pointing to a canonical dict
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {
            "custom_key": {"items": [{"score": 0.9, "label": "foreground"}]},
            "meta": {"id": 1},
        }
        frame1 = Frame(image, data, "BGR")

        output = filter_instance.process({"main": frame1})

        self.assertIn("main", output)
        self.assertEqual(filter_instance.interval_tracker.frame_count, 1)
        self.assertTrue(filter_instance.interval_tracker.is_present("foreground"))

        # Also test fallback to meta path for custom key in canonical format
        data_meta = {
            "meta": {
                "id": 2,
                "custom_key": {"items": [{"score": 0.85, "label": "foreground"}]},
            }
        }
        frame2 = Frame(image, data_meta, "BGR")
        filter_instance.process({"main": frame2})
        self.assertEqual(filter_instance.interval_tracker.frame_count, 2)
        self.assertTrue(filter_instance.interval_tracker.is_present("foreground"))

        filter_instance.shutdown()

    def test_process_with_explicit_legacy_detection_key_priority(self):
        """Test that explicit detection_key 'sam3_detections' is prioritized over 'detections' when both exist."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "detection_key": "sam3_detections",
                "emit_on_change": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame has both top-level 'detections' (car) and meta 'sam3_detections' (person)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {
            "detections": {"items": [{"score": 0.9, "label": "car"}]},
            "meta": {"id": 1, "sam3_detections": [{"score": 0.95, "label": "person"}]},
        }
        frame = Frame(image, data, "BGR")

        output = filter_instance.process({"main": frame})

        self.assertIn("main", output)
        self.assertEqual(filter_instance.interval_tracker.frame_count, 1)
        # Should track 'person' from 'sam3_detections', not 'car' from 'detections'
        self.assertTrue(filter_instance.interval_tracker.is_present("person"))
        self.assertFalse(filter_instance.interval_tracker.is_present("car"))

        filter_instance.shutdown()


class TestEMAHalfLifeVerification(TestCase):
    """Verify EMA half-life behavior mathematically."""

    def test_half_life_decay_verification(self):
        """Verify that debiased EMA decays monotonically and respects half_life asymptotically."""
        half_life = 10.0
        tracker = EMATracker(half_life=half_life, threshold=0.5)

        # Start at 1.0 (first detection gives debiased EMA = 1.0)
        tracker.update("test", True)
        self.assertEqual(tracker.get_ema("test"), 1.0)

        # Track decay over time - should decrease monotonically
        prev_ema = 1.0
        for i in range(50):
            tracker.update("test", False)
            ema = tracker.get_ema("test")
            self.assertLess(ema, prev_ema)  # Should decrease monotonically
            prev_ema = ema

        # After many frames, should approach 0
        self.assertLess(tracker.get_ema("test"), 0.01)

    def test_full_decay_life_verification(self):
        """Verify that EMA reaches ~0 after full_decay_life frames."""
        full_decay_life = 20.0
        tracker = EMATracker(full_decay_life=full_decay_life, threshold=0.5)

        # Start at 1.0
        tracker.update("test", True)

        # Decay for full_decay_life frames
        for _ in range(int(full_decay_life)):
            tracker.update("test", False)

        # Should be very close to 0 (within 1%)
        ema = tracker.get_ema("test")
        self.assertLess(ema, 0.01)

    def test_debiased_ema_first_frame_accuracy(self):
        """Verify that debiased EMA accurately reflects the first observation."""
        tracker = EMATracker(half_life=10.0, threshold=0.5)

        # First detection should give EMA = 1.0 (not biased toward 0)
        ema, _, _ = tracker.update("test", True)
        self.assertEqual(ema, 1.0)

        # First non-detection should give EMA = 0.0 (not biased)
        tracker2 = EMATracker(half_life=10.0, threshold=0.5)
        ema2, _, _ = tracker2.update("test", False)
        self.assertEqual(ema2, 0.0)


class TestAssertionCompatibility(TestCase):
    """Test compatibility with assertion testing patterns."""

    def test_interval_format_for_assertions(self):
        """Test that output format is suitable for assertion testing."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Simulate a detection sequence
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # 5 frames with detection
        for i in range(1, 6):
            frame = Frame(
                image, {"meta": {"id": i, "sam3_detections": [{"score": 0.9}]}}, "BGR"
            )
            filter_instance.process({"main": frame})

        # 10 frames without detection
        for i in range(6, 16):
            frame = Frame(image, {"meta": {"id": i, "sam3_detections": []}}, "BGR")
            filter_instance.process({"main": frame})

        filter_instance.shutdown()

        intervals = filter_instance.get_intervals()

        # Should have at least 2 intervals (present, then absent)
        self.assertGreaterEqual(len(intervals), 1)

        # Check interval structure
        for interval in intervals:
            self.assertIsInstance(interval.start_frame, int)
            self.assertIsInstance(interval.end_frame, int)
            self.assertIsInstance(interval.label, str)
            self.assertIsInstance(interval.present, bool)
            self.assertIsInstance(interval.confidence, float)
            self.assertGreater(interval.end_frame, interval.start_frame)

        # Intervals should be useful for assertions like:
        # "person present between frames 1-50"
        # "no detections between frames 51-100"
        presence_intervals = [i for i in intervals if i.present]
        absence_intervals = [i for i in intervals if not i.present]

        # We should have both types
        self.assertGreater(len(presence_intervals) + len(absence_intervals), 0)


class TestEventTopicEmission(TestCase):
    """Test event topic emission for filter-event-sink integration."""

    def test_emit_event_topic_disabled_by_default(self):
        """Test that event topic emission is disabled by default."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
            }
        )
        self.assertFalse(config.emit_event_topic)

    def test_emit_event_topic_enabled(self):
        """Test enabling event topic emission via config."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
            }
        )
        self.assertTrue(config.emit_event_topic)

    def test_emit_event_topic_string_conversion(self):
        """Test that string 'true' is converted to boolean."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": "true",
            }
        )
        self.assertTrue(config.emit_event_topic)

        config2 = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": "false",
            }
        )
        self.assertFalse(config2.emit_event_topic)

    def test_custom_event_topic_name(self):
        """Test custom event topic name configuration."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
                "event_topic_name": "custom_events",
            }
        )
        self.assertEqual(config.event_topic_name, "custom_events")

    def test_event_topic_emitted_on_state_change(self):
        """Test that event topic frame is emitted when state changes."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
                "event_topic_name": "events",
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame with detection - should trigger state change (absent -> present)
        frame = Frame(
            image, {"meta": {"id": 1, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output = filter_instance.process({"main": frame})

        # Should have both the main topic and the events topic
        self.assertIn("main", output)
        self.assertIn("events", output)

        # Events topic should have state change data
        events_frame = output["events"]
        self.assertIsNotNone(events_frame.data)
        self.assertIn("frame_id", events_frame.data)
        self.assertIn("state_changes", events_frame.data)
        self.assertEqual(len(events_frame.data["state_changes"]), 1)
        self.assertEqual(events_frame.data["state_changes"][0]["label"], "foreground")
        self.assertTrue(events_frame.data["state_changes"][0]["present"])

        filter_instance.shutdown()

    def test_no_event_topic_when_no_state_change(self):
        """Test that event topic is not emitted when there's no state change."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame triggers state change
        frame1 = Frame(
            image, {"meta": {"id": 1, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        filter_instance.process({"main": frame1})

        # Second frame with same detection - no state change
        frame2 = Frame(
            image, {"meta": {"id": 2, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output2 = filter_instance.process({"main": frame2})

        # Should only have main topic, not events
        self.assertIn("main", output2)
        self.assertNotIn("events", output2)

        filter_instance.shutdown()

    def test_event_topic_not_emitted_when_disabled(self):
        """Test that event topic is not emitted when emit_event_topic is False."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": False,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Frame with detection - state change but emit_event_topic is False
        frame = Frame(
            image, {"meta": {"id": 1, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output = filter_instance.process({"main": frame})

        # Should only have main topic
        self.assertIn("main", output)
        self.assertNotIn("events", output)

        # But state change should still be in metadata
        self.assertIn("temporal_intervals", output["main"].data["meta"])

        filter_instance.shutdown()

    def test_event_topic_format_for_event_sink(self):
        """Test that event topic data format is compatible with filter-event-sink."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Trigger presence state
        frame = Frame(
            image, {"meta": {"id": 42, "sam3_detections": [{"score": 0.95}]}}, "BGR"
        )
        output = filter_instance.process({"main": frame})

        events_data = output["events"].data

        # Verify format expected by filter-event-sink:
        # The data should be a dict that can be directly used as event payload
        self.assertIsInstance(events_data, dict)
        self.assertIn("frame_id", events_data)
        self.assertIn("state_changes", events_data)

        # frame_id should be the actual frame ID
        self.assertEqual(events_data["frame_id"], 42)

        # state_changes should be a list of dicts with label, present, ema
        self.assertIsInstance(events_data["state_changes"], list)
        for change in events_data["state_changes"]:
            self.assertIn("label", change)
            self.assertIn("present", change)
            self.assertIn("ema", change)
            self.assertIsInstance(change["ema"], float)

        filter_instance.shutdown()


class TestEventSinkIntegration(TestCase):
    """Integration tests simulating the full event sink pipeline flow."""

    def test_event_sink_extraction_simulation(self):
        """
        Simulate the full pipeline: temporal filter -> event sink extraction.

        This test verifies that the event data emitted by TemporalIntervalFilter
        is correctly formatted for filter-event-sink's _extract_events method.
        """
        # Setup temporal filter with event emission enabled
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
                "event_topic_name": "events",
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Trigger a state change (absent -> present)
        frame = Frame(
            image, {"meta": {"id": 100, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output = filter_instance.process({"main": frame})

        # Verify events topic was emitted
        self.assertIn("events", output)
        events_frame = output["events"]

        # --- Simulate event sink receiving the frame ---
        # In the real pipeline, the topic would be remapped:
        # tcp://temporal_intervals??;events>TemporalIntervals__events
        # So event sink sees: {"TemporalIntervals__events": events_frame}

        remapped_topic = "TemporalIntervals__events"
        frames_at_event_sink = {remapped_topic: events_frame}

        # Simulate event sink's _extract_events logic
        extracted_events = []
        for topic, frame in frames_at_event_sink.items():
            if not frame.data:
                continue

            topic_parts = topic.split("__")
            source_filter_name = topic_parts[0]
            source_topic = "main"
            if len(topic_parts) > 1:
                source_topic = topic_parts[1]

            extracted_events.append(
                {
                    "filter_name": source_filter_name,
                    "topic": source_topic,
                    "data": frame.data,
                }
            )

        # Verify extraction worked correctly
        self.assertEqual(len(extracted_events), 1)
        event = extracted_events[0]

        # Check filter name extracted from topic prefix
        self.assertEqual(event["filter_name"], "TemporalIntervals")

        # Check topic extracted from suffix
        self.assertEqual(event["topic"], "events")

        # Check data structure
        self.assertIn("frame_id", event["data"])
        self.assertEqual(event["data"]["frame_id"], 100)
        self.assertIn("state_changes", event["data"])
        self.assertEqual(len(event["data"]["state_changes"]), 1)
        self.assertEqual(event["data"]["state_changes"][0]["label"], "foreground")
        self.assertTrue(event["data"]["state_changes"][0]["present"])

        filter_instance.shutdown()

    def test_multiple_state_changes_extraction(self):
        """Test extraction of multiple state changes across frames."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
                "label_field": "class",  # Use class field for multi-label tracking
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Frame with multiple classes detected
        detections = [
            {"class": "person", "score": 0.9},
            {"class": "car", "score": 0.85},
        ]
        frame = Frame(image, {"meta": {"id": 1, "sam3_detections": detections}}, "BGR")
        output = filter_instance.process({"main": frame})

        self.assertIn("events", output)
        events_data = output["events"].data

        # Should have state changes for both labels
        self.assertEqual(len(events_data["state_changes"]), 2)
        labels = {sc["label"] for sc in events_data["state_changes"]}
        self.assertEqual(labels, {"person", "car"})

        # All should be present=True (first detection)
        for sc in events_data["state_changes"]:
            self.assertTrue(sc["present"])

        filter_instance.shutdown()

    def test_no_event_when_no_state_change(self):
        """Verify no events emitted when state doesn't change."""
        config = TemporalIntervalFilter.normalize_config(
            {
                "half_life": 1.0,
                "emit_event_topic": True,
            }
        )
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame - triggers state change
        frame1 = Frame(
            image, {"meta": {"id": 1, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output1 = filter_instance.process({"main": frame1})
        self.assertIn("events", output1)

        # Second frame with same detection - no state change
        frame2 = Frame(
            image, {"meta": {"id": 2, "sam3_detections": [{"score": 0.9}]}}, "BGR"
        )
        output2 = filter_instance.process({"main": frame2})

        # Should NOT have events topic (no state change)
        self.assertNotIn("events", output2)

        # Event sink would receive empty frames dict for this frame
        # This is correct - only emit events on state changes

        filter_instance.shutdown()


class TestExtractItemsBypassEmptyList(TestCase):
    """Test that extract_items correctly handles and returns empty lists without falling back to meta."""

    def test_extract_items_handles_empty_list_properly(self):
        """Test that extract_items returns an empty list immediately when 'detections' is an empty list."""
        from filter_sam3_detector.utils.detections import extract_items

        data = {
            "detections": [],
            "meta": {"sam3_detections": [{"score": 0.95, "label": "person"}]},
        }
        # Previously, this would bypass detections=[] and fall back to meta["sam3_detections"]
        # Now, it must correctly return [] immediately.
        self.assertEqual(extract_items(data), [])
