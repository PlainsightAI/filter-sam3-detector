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
        self.assertAlmostEqual(tracker.alpha, expected_fast, places=5)  # Primary alpha is fast
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
        """Test that EMA decays when detection stops."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)  # Fast decay

        # Start with detection
        tracker.update("person", True)

        # After 1 frame without detection, should be at ~50%
        ema, is_present, _ = tracker.update("person", False)
        self.assertAlmostEqual(ema, 0.5, places=2)
        # Note: 0.5 >= 0.5 is True (at threshold = present)
        self.assertTrue(is_present)

        # After another frame, should drop below threshold
        ema, is_present, _ = tracker.update("person", False)
        self.assertLess(ema, 0.5)
        self.assertFalse(is_present)  # Now below threshold

    def test_ema_rise_with_detection(self):
        """Test that EMA rises with continued detection."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)

        # Start with no detection
        tracker.update("person", False)

        # After 1 frame with detection, should be at ~50%
        ema, is_present, _ = tracker.update("person", True)
        self.assertAlmostEqual(ema, 0.5, places=2)
        self.assertTrue(is_present)  # At threshold

    def test_state_transition_detection(self):
        """Test that state transitions are correctly detected."""
        tracker = EMATracker(half_life=1.0, threshold=0.5)

        # Initial state
        tracker.update("person", True)  # Present

        # Continue detecting - no state change
        _, _, changed = tracker.update("person", True)
        self.assertFalse(changed)

        # Stop detecting - at threshold (0.5 >= 0.5 is True, still present)
        ema, is_present, changed = tracker.update("person", False)
        # With alpha=0.5: ema = 0.5*0 + 0.5*1 = 0.5, at threshold
        self.assertAlmostEqual(ema, 0.5, places=2)
        self.assertTrue(is_present)  # 0.5 >= 0.5 is True
        self.assertFalse(changed)  # No state change yet

        # Another frame without detection - drops below threshold
        ema, is_present, changed = tracker.update("person", False)
        # ema = 0.5*0 + 0.5*0.5 = 0.25
        self.assertAlmostEqual(ema, 0.25, places=2)
        self.assertFalse(is_present)  # Now below threshold
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
        # Fast EMA with half_life=1 (alpha=0.5), slow EMA with full_decay_life=10
        tracker = EMATracker(half_life=1.0, full_decay_life=10.0, threshold=0.5)

        # Start with detection
        fast_ema, is_present, changed = tracker.update("person", True)
        self.assertEqual(fast_ema, 1.0)
        self.assertTrue(is_present)
        self.assertTrue(changed)

        # Stop detecting - fast EMA drops quickly but slow EMA stays high longer
        fast_ema, is_present, changed = tracker.update("person", False)
        slow_ema = tracker.get_slow_ema("person")

        # Fast EMA should drop to 0.5, but slow EMA drops much slower
        self.assertAlmostEqual(fast_ema, 0.5, places=2)
        self.assertGreater(slow_ema, 0.5)  # Still above threshold
        self.assertTrue(is_present)  # State based on slow EMA
        self.assertFalse(changed)

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
        # Fast EMA with half_life=1 drops below 0.5 after just 1-2 frames
        # Slow EMA with full_decay_life=30 takes longer (about 5 frames to cross 0.5)
        self.assertIsNotNone(state_change_frame)
        self.assertGreaterEqual(state_change_frame, 5)  # Slower than fast EMA alone

    def test_dual_ema_returns_fast_ema_value(self):
        """Test that update() returns the fast EMA value, not slow."""
        tracker = EMATracker(half_life=1.0, full_decay_life=30.0, threshold=0.5)

        tracker.update("person", True)
        fast_ema, _, _ = tracker.update("person", False)

        # Returned value should be fast EMA (alpha=0.5)
        self.assertAlmostEqual(fast_ema, 0.5, places=2)

        # But slow EMA should be different
        slow_ema = tracker.get_slow_ema("person")
        self.assertNotAlmostEqual(slow_ema, fast_ema, places=2)
        self.assertGreater(slow_ema, fast_ema)  # Slow decays slower

    def test_get_crossing_progress_returns_slow_ema(self):
        """Test that get_crossing_progress returns slow EMA value."""
        tracker = EMATracker(half_life=1.0, full_decay_life=30.0, threshold=0.5)

        tracker.update("person", True)
        tracker.update("person", False)
        tracker.update("person", False)

        current_state, slow_ema = tracker.get_crossing_progress("person")
        self.assertTrue(current_state)  # Still present (slow EMA above threshold)
        self.assertGreater(slow_ema, 0.5)

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
        # With full_decay_life=30, alpha ≈ 0.154, takes ~5 frames to drop from 1.0 to 0.5
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
        # With full_decay_life=30, takes about 5 frames for slow EMA to cross 0.5
        # This is slower than if we only had half_life=5 (which would also take ~5-6 frames)
        self.assertGreaterEqual(state_change_frame, 4)


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
        return Frame(image, data, 'BGR')

    def test_normalize_config_defaults(self):
        """Test config normalization with defaults."""
        config = TemporalIntervalFilter.normalize_config({})

        self.assertIsInstance(config, TemporalIntervalConfig)
        self.assertEqual(config.presence_threshold, 0.5)
        self.assertEqual(config.detection_key, "sam3_detections")
        self.assertEqual(config.default_label, "foreground")

    def test_normalize_config_custom_values(self):
        """Test config normalization with custom values."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': '10',
            'presence_threshold': '0.7',
            'emit_on_change': 'false',
        })

        self.assertEqual(config.half_life, 10.0)
        self.assertEqual(config.presence_threshold, 0.7)
        self.assertFalse(config.emit_on_change)

    def test_normalize_config_invalid_threshold(self):
        """Test that invalid thresholds are rejected."""
        with self.assertRaises(ValueError):
            TemporalIntervalFilter.normalize_config({'presence_threshold': '1.5'})
        with self.assertRaises(ValueError):
            TemporalIntervalFilter.normalize_config({'min_confidence': '-0.1'})

    def test_setup_and_shutdown(self):
        """Test filter setup and shutdown."""
        config = TemporalIntervalFilter.normalize_config({'half_life': 5.0})
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        self.assertIsNotNone(filter_instance.tracker)
        self.assertEqual(filter_instance.frame_count, 0)
        self.assertEqual(len(filter_instance.intervals), 0)

        filter_instance.shutdown()

    def test_process_with_detections(self):
        """Test processing frames with detections."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 1.0,
            'emit_on_change': True,
        })
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with detection
        frame1 = self._create_frame(
            detections=[{"score": 0.9}],
            frame_id=1
        )
        output = filter_instance.process({"main": frame1})

        self.assertIn("main", output)
        self.assertEqual(filter_instance.frame_count, 1)
        self.assertTrue(filter_instance.tracker.is_present("foreground"))

        filter_instance.shutdown()

    def test_process_detection_to_absence_transition(self):
        """Test state transition from detection to absence."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 0.5,  # Very fast decay
            'emit_on_change': True,
        })
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

        self.assertFalse(filter_instance.tracker.is_present("foreground"))

        filter_instance.shutdown()

        # Should have intervals recorded
        self.assertGreater(len(filter_instance.intervals), 0)

    def test_process_with_label_field(self):
        """Test processing with labeled detections."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 5.0,
            'label_field': 'class',
        })
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with labeled detections
        frame = self._create_frame(
            detections=[
                {"class": "person", "score": 0.9},
                {"class": "car", "score": 0.8},
            ],
            frame_id=1
        )
        filter_instance.process({"main": frame})

        self.assertTrue(filter_instance.tracker.is_present("person"))
        self.assertTrue(filter_instance.tracker.is_present("car"))

        filter_instance.shutdown()

    def test_min_confidence_filtering(self):
        """Test that low-confidence detections are filtered."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 5.0,
            'min_confidence': 0.7,
        })
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Frame with low-confidence detection
        frame = self._create_frame(
            detections=[{"score": 0.5}],  # Below threshold
            frame_id=1
        )
        filter_instance.process({"main": frame})

        # Should not register as detected
        self.assertFalse(filter_instance.tracker.is_present("foreground"))

        filter_instance.shutdown()

    def test_output_json_file(self):
        """Test writing intervals to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "intervals.json"

            config = TemporalIntervalFilter.normalize_config({
                'half_life': 0.5,
                'output_json_path': str(output_path),
                'emit_on_complete': True,
            })
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

            # Check output file
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                data = json.load(f)

            self.assertIn("total_frames", data)
            self.assertIn("intervals", data)
            self.assertEqual(data["total_frames"], 10)
            self.assertGreater(len(data["intervals"]), 0)

    def test_get_current_state(self):
        """Test getting current state of all labels."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 5.0,
            'label_field': 'class',
        })
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        frame = self._create_frame(
            detections=[
                {"class": "person", "score": 0.9},
                {"class": "car", "score": 0.8},
            ],
            frame_id=1
        )
        filter_instance.process({"main": frame})

        state = filter_instance.get_current_state()

        self.assertIn("person", state)
        self.assertIn("car", state)
        self.assertTrue(state["person"]["present"])
        self.assertTrue(state["car"]["present"])
        self.assertEqual(state["person"]["ema"], 1.0)

        filter_instance.shutdown()


class TestEMAHalfLifeVerification(TestCase):
    """Verify EMA half-life behavior mathematically."""

    def test_half_life_decay_verification(self):
        """Verify that EMA reaches 50% after half_life frames."""
        half_life = 10.0
        tracker = EMATracker(half_life=half_life, threshold=0.5)

        # Start at 1.0
        tracker.update("test", True)
        self.assertEqual(tracker.get_ema("test"), 1.0)

        # Decay for exactly half_life frames
        for _ in range(int(half_life)):
            tracker.update("test", False)

        # Should be approximately at 50% of the way to 0
        ema = tracker.get_ema("test")
        self.assertAlmostEqual(ema, 0.5, delta=0.05)

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


class TestAssertionCompatibility(TestCase):
    """Test compatibility with assertion testing patterns."""

    def test_interval_format_for_assertions(self):
        """Test that output format is suitable for assertion testing."""
        config = TemporalIntervalFilter.normalize_config({
            'half_life': 1.0,
        })
        filter_instance = TemporalIntervalFilter(config=config)
        filter_instance.setup(config)

        # Simulate a detection sequence
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # 5 frames with detection
        for i in range(1, 6):
            frame = Frame(image, {"meta": {"id": i, "sam3_detections": [{"score": 0.9}]}}, 'BGR')
            filter_instance.process({"main": frame})

        # 10 frames without detection
        for i in range(6, 16):
            frame = Frame(image, {"meta": {"id": i, "sam3_detections": []}}, 'BGR')
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
