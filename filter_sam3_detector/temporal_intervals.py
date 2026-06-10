"""
Temporal detection interval generator using EMA smoothing.

This module provides an EMA-based temporal interval tracker that converts
per-frame detection results into stable presence/absence intervals suitable
for ground truth generation in assertion testing.

The tracker uses Exponential Moving Average (EMA) with configurable decay
parameters specified in frame counts (half-life or full decay life) to
smooth noisy per-frame detections into stable temporal intervals.

Features:
- Dual EMA smoothing: fast EMA (half-life) for signal, slow EMA (full decay) for crossing detection
- Crossing detection uses the slow EMA to prevent threshold flickering
- Frame ID support compatible with TI-129 (_filter topic)
- JSON output for assertion testing ground truth
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

__all__ = [
    "TemporalIntervalConfig",
    "TemporalIntervalFilter",
    "EMATracker",
    "DetectionInterval",
    "IntervalTracker",
]

logger = logging.getLogger(__name__)


@dataclass
class DetectionInterval:
    """Represents a temporal interval where a detection class is present or absent."""

    start_frame: int
    end_frame: int
    label: str
    present: bool
    confidence: float  # Average EMA confidence during interval

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "label": self.label,
            "present": self.present,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionInterval":
        """Create from dictionary."""
        return cls(
            start_frame=d["start_frame"],
            end_frame=d["end_frame"],
            label=d["label"],
            present=d["present"],
            confidence=d["confidence"],
        )


class EMATracker:
    """
    Dual Exponential Moving Average tracker for detection presence.

    Always uses two EMAs for robust presence detection:
    - Fast EMA (half-life based): Quickly responds to detection changes, used for signal value
    - Slow EMA (full decay based): Used for crossing detection to prevent flickering

    EMA formula: ema[t] = alpha * value + (1 - alpha) * ema[t-1]

    Where alpha is derived from:
    - half_life: alpha = 1 - 2^(-1/half_life)  # 50% decay after half_life frames
    - full_decay_life: alpha = 1 - e^(-5/full_decay_life)  # 99.3% decay after full_decay_life frames

    Mathematical relationship between half_life and full_decay_life:
    - full_decay_life = 5 * half_life / ln(2) ≈ 7.21 * half_life
    - half_life = full_decay_life * ln(2) / 5 ≈ 0.139 * full_decay_life

    If only one parameter is specified, the other is computed automatically.
    The fast EMA (half_life) is used for the reported signal value.
    The slow EMA (full_decay_life) is used for state crossing detection.

    Args:
        half_life: Number of frames for fast EMA to reach 50% of step change.
            If not specified, derived from full_decay_life.
        full_decay_life: Number of frames for slow EMA to reach ~99.3% of step change.
            If not specified, derived from half_life.
        threshold: Threshold for presence detection (default: 0.5)
    """

    # Conversion factors between half_life and full_decay_life
    # full_decay_life = half_life * HALF_TO_FULL
    # half_life = full_decay_life * FULL_TO_HALF
    HALF_TO_FULL = 5.0 / math.log(2)  # ≈ 7.213
    FULL_TO_HALF = math.log(2) / 5.0  # ≈ 0.139

    def __init__(
        self,
        half_life: Optional[float] = None,
        full_decay_life: Optional[float] = None,
        threshold: float = 0.5,
    ):
        # Validate inputs
        if half_life is not None and half_life <= 0:
            raise ValueError(f"half_life must be positive, got {half_life}")
        if full_decay_life is not None and full_decay_life <= 0:
            raise ValueError(f"full_decay_life must be positive, got {full_decay_life}")

        # Apply defaults and derive missing parameter
        if half_life is None and full_decay_life is None:
            half_life = 5.0  # Default: 5 frame half-life

        # Derive missing parameter from the other
        if half_life is not None and full_decay_life is None:
            full_decay_life = half_life * self.HALF_TO_FULL
        elif full_decay_life is not None and half_life is None:
            half_life = full_decay_life * self.FULL_TO_HALF

        # Store both lifetimes
        self.half_life = half_life
        self.full_decay_life = full_decay_life

        # Calculate alphas
        # Fast alpha (half-life based): 50% decay after half_life frames
        self.alpha_fast = 1 - math.pow(2, -1 / half_life)
        # Slow alpha (full decay based): 99.3% decay after full_decay_life frames
        self.alpha_slow = 1 - math.exp(-5 / full_decay_life)

        # Primary alpha is fast EMA (for backwards compatibility with .alpha attribute)
        self.alpha = self.alpha_fast

        self.threshold = threshold

        # Fast EMA state (primary signal) - stores RAW (biased) EMA values
        self._ema_raw: dict[str, float] = {}  # label -> raw fast EMA value
        self.ema: dict[
            str, float
        ] = {}  # label -> debiased fast EMA value (for external access)
        self.current_state: dict[
            str, bool
        ] = {}  # label -> is_present (confirmed state)

        # Slow EMA state (for crossing detection) - stores RAW (biased) EMA values
        self._slow_ema_raw: dict[str, float] = {}  # label -> raw slow EMA value
        self._slow_ema: dict[str, float] = {}  # label -> debiased slow EMA value

        # Bias correction factors: (1 - alpha)^t for each label
        self._fast_bias: dict[str, float] = {}  # label -> (1 - alpha_fast)^t
        self._slow_bias: dict[str, float] = {}  # label -> (1 - alpha_slow)^t

    def update(self, label: str, detected: bool) -> tuple[float, bool, bool]:
        """
        Update EMA for a label and return current state.

        Uses debiased EMA to avoid initial bias toward 0. The debiasing formula is:
            ema_debiased = ema_raw / (1 - (1 - alpha)^t)

        This ensures the first observation is accurately reflected (EMA = 1.0 for
        first detection, EMA = 0.0 for first non-detection).

        Args:
            label: Detection class label
            detected: Whether the class was detected in current frame

        Returns:
            Tuple of (fast_ema_value, is_present, state_changed)
            - fast_ema_value: The debiased fast EMA signal value
            - is_present: State based on debiased slow EMA crossing threshold
            - state_changed: Whether state changed this frame
        """
        value = 1.0 if detected else 0.0

        if label not in self._ema_raw:
            # Initialize both EMAs - first observation gets value directly
            self._ema_raw[label] = self.alpha_fast * value
            self._slow_ema_raw[label] = self.alpha_slow * value

            # Initialize bias correction factors
            self._fast_bias[label] = 1 - self.alpha_fast
            self._slow_bias[label] = 1 - self.alpha_slow

            # Debiased values for first frame
            debiased_fast = value  # First frame: raw / (1 - bias) = alpha*v / alpha = v
            debiased_slow = value

            self.ema[label] = debiased_fast
            self._slow_ema[label] = debiased_slow
            self.current_state[label] = debiased_slow >= self.threshold

            return debiased_fast, self.current_state[label], True

        # Update raw fast EMA
        prev_fast_raw = self._ema_raw[label]
        new_fast_raw = self.alpha_fast * value + (1 - self.alpha_fast) * prev_fast_raw
        self._ema_raw[label] = new_fast_raw

        # Update raw slow EMA
        prev_slow_raw = self._slow_ema_raw[label]
        new_slow_raw = self.alpha_slow * value + (1 - self.alpha_slow) * prev_slow_raw
        self._slow_ema_raw[label] = new_slow_raw

        # Update bias correction factors: bias_t = (1 - alpha)^t
        self._fast_bias[label] *= 1 - self.alpha_fast
        self._slow_bias[label] *= 1 - self.alpha_slow

        # Compute debiased EMAs: debiased = raw / (1 - bias)
        debiased_fast = new_fast_raw / (1 - self._fast_bias[label])
        debiased_slow = new_slow_raw / (1 - self._slow_bias[label])

        # Store debiased values
        self.ema[label] = debiased_fast
        self._slow_ema[label] = debiased_slow

        # State is determined by debiased slow EMA crossing threshold
        prev_state = self.current_state[label]
        new_state = debiased_slow >= self.threshold
        state_changed = prev_state != new_state
        self.current_state[label] = new_state

        return debiased_fast, new_state, state_changed

    def get_ema(self, label: str) -> float:
        """Get current fast EMA value for a label."""
        return self.ema.get(label, 0.0)

    def get_slow_ema(self, label: str) -> float:
        """Get current slow EMA value for a label (for crossing detection)."""
        return self._slow_ema.get(label, 0.0)

    def is_present(self, label: str) -> bool:
        """Check if label is currently considered present."""
        return self.current_state.get(label, False)

    def get_crossing_progress(self, label: str) -> tuple[bool, float]:
        """
        Get crossing detection progress for a label.

        Returns:
            Tuple of (current_state, slow_ema_value)
            In dual-EMA mode, slow_ema_value shows how close we are to threshold crossing.
        """
        return (
            self.current_state.get(label, False),
            self._slow_ema.get(label, 0.0),
        )

    def reset(self):
        """Reset all tracking state."""
        self._ema_raw.clear()
        self.ema.clear()
        self._slow_ema_raw.clear()
        self._slow_ema.clear()
        self._fast_bias.clear()
        self._slow_bias.clear()
        self.current_state.clear()


class IntervalTracker:
    """
    Reusable interval tracking component that can be embedded in any filter.

    This class encapsulates the interval tracking logic (EMA updates, state changes,
    interval management) without being tied to the Filter interface. It can be used
    by both TemporalIntervalFilter (standalone) and FilterSAM3Detector (integrated).

    Features:
    - EMA-based presence detection with dual EMA for robustness
    - Streaming JSON output for real-time interval emission
    - Interval aggregation with confidence averaging
    - Optional callback for state changes

    Args:
        half_life: Frames for 50% EMA decay (fast signal)
        full_decay_life: Frames for ~99.3% EMA decay (slow crossing)
        presence_threshold: EMA threshold for presence detection
        output_json_path: Optional path for JSON output (streaming)
        on_interval_closed: Optional callback(interval) when interval closes
        streaming_mode: If True, emit intervals incrementally to JSON file
    """

    def __init__(
        self,
        half_life: Optional[float] = None,
        full_decay_life: Optional[float] = None,
        presence_threshold: float = 0.5,
        output_json_path: Optional[str] = None,
        on_interval_closed: Optional[callable] = None,
        streaming_mode: bool = False,
    ):
        self.tracker = EMATracker(
            half_life=half_life,
            full_decay_life=full_decay_life,
            threshold=presence_threshold,
        )

        self.output_json_path = Path(output_json_path) if output_json_path else None
        self.on_interval_closed = on_interval_closed
        self.streaming_mode = streaming_mode

        # Interval tracking state
        self.frame_count = 0
        self.intervals: list[DetectionInterval] = []
        self.open_intervals: dict[
            str, dict
        ] = {}  # label -> {start_frame, confidence_sum, frame_count, present}

        # Streaming file handle
        self._json_file = None

        if self.output_json_path and self.streaming_mode:
            self._open_streaming_file()

    def _open_streaming_file(self):
        """Open streaming JSON file (ndjson format - one interval per line)."""
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_file = open(self.output_json_path, "w")
        logger.info(f"Streaming intervals to: {self.output_json_path}")

    def update(
        self, detected_labels: dict[str, float], frame_id: Optional[int] = None
    ) -> list[tuple[str, bool, float]]:
        """
        Update tracking with current frame's detections.

        Args:
            detected_labels: Dict mapping label -> max confidence score for this frame
            frame_id: Optional explicit frame ID (uses internal counter if None)

        Returns:
            List of state changes as (label, is_present, ema) tuples
        """
        self.frame_count += 1
        current_frame = frame_id if frame_id is not None else self.frame_count

        # Update EMA for all known labels
        all_labels = set(detected_labels.keys()) | set(self.tracker.ema.keys())
        state_changes = []

        for label in all_labels:
            detected = label in detected_labels

            ema, is_present, changed = self.tracker.update(label, detected)

            if changed:
                state_changes.append((label, is_present, ema))
                self._handle_state_change(label, is_present, current_frame, ema)
            elif label in self.open_intervals:
                # Update running confidence for open interval
                self.open_intervals[label]["confidence_sum"] += ema
                self.open_intervals[label]["frame_count"] += 1

        return state_changes

    def _handle_state_change(
        self, label: str, is_present: bool, frame_id: int, ema: float
    ):
        """Handle a state transition for a label."""
        if is_present:
            # Start new presence interval
            self.open_intervals[label] = {
                "start_frame": frame_id,
                "confidence_sum": ema,
                "frame_count": 1,
                "present": True,
            }
        else:
            # Close existing interval and start absence interval
            self._close_interval(label)
            self.open_intervals[label] = {
                "start_frame": frame_id,
                "confidence_sum": ema,
                "frame_count": 1,
                "present": False,
            }

    def _close_interval(self, label: str):
        """Close an open interval and add to completed intervals."""
        if label not in self.open_intervals:
            return

        info = self.open_intervals.pop(label)
        avg_confidence = info["confidence_sum"] / max(1, info["frame_count"])

        interval = DetectionInterval(
            start_frame=info["start_frame"],
            end_frame=self.frame_count,
            label=label,
            present=info["present"],
            confidence=avg_confidence,
        )
        self.intervals.append(interval)

        # Stream to file if enabled
        if self._json_file is not None:
            self._stream_interval(interval)

        # Invoke callback if provided
        if self.on_interval_closed:
            self.on_interval_closed(interval)

        logger.debug(f"Closed interval: {interval}")

    def _stream_interval(self, interval: DetectionInterval):
        """Write a single interval to the streaming ndjson file."""
        try:
            interval_json = json.dumps(interval.to_dict())
            self._json_file.write(f"{interval_json}\n")
            self._json_file.flush()
        except Exception as e:
            logger.warning(f"Failed to stream interval: {e}")

    def finalize(self):
        """
        Finalize tracking: close open intervals and close output file.

        Call this when processing is complete (e.g., in filter shutdown).
        Output is ndjson format - each line is a complete JSON object.
        """
        # Close any open intervals
        for label in list(self.open_intervals.keys()):
            self._close_interval(label)

        logger.info(
            f"IntervalTracker: {len(self.intervals)} intervals recorded over {self.frame_count} frames"
        )

        # Close streaming file (no footer needed - ndjson format)
        if self._json_file is not None:
            try:
                self._json_file.close()
                logger.info(f"Closed streaming ndjson: {self.output_json_path}")
            except Exception as e:
                logger.warning(f"Error closing streaming file: {e}")
            self._json_file = None

    def get_intervals(self) -> list[DetectionInterval]:
        """Get all completed intervals."""
        return list(self.intervals)

    def get_current_state(self) -> dict[str, dict]:
        """Get current presence state for all tracked labels."""
        return {
            label: {
                "present": self.tracker.is_present(label),
                "ema": round(self.tracker.get_ema(label), 4),
            }
            for label in self.tracker.ema.keys()
        }

    def is_present(self, label: str) -> bool:
        """Check if a label is currently present."""
        return self.tracker.is_present(label)

    def get_ema(self, label: str) -> float:
        """Get current EMA value for a label."""
        return self.tracker.get_ema(label)


class TemporalIntervalConfig(FilterConfig):
    """Configuration for temporal interval detection filter."""

    # EMA parameters - specify one or both
    # If only one is specified, the other is derived automatically:
    #   full_decay_life ≈ 7.21 * half_life
    #   half_life ≈ 0.139 * full_decay_life
    half_life: Optional[float] = None  # Frames for 50% decay (fast EMA for signal)
    full_decay_life: Optional[float] = (
        None  # Frames for ~99.3% decay (slow EMA for crossing)
    )

    # Detection thresholds
    presence_threshold: float = 0.5  # EMA threshold for presence detection
    min_confidence: float = 0.0  # Minimum detection confidence to consider

    # Input configuration
    detection_key: str = "sam3_detections"  # Key in frame.data['meta'] for detections
    score_field: str = "score"  # Field name for detection confidence
    label_field: Optional[str] = (
        "label"  # Field name for class label (if None, tracks all as default_label)
    )
    default_label: str = "foreground"  # Label to use when label_field is None

    # Output configuration
    output_json_path: Optional[str] = None  # Path to write intervals JSON
    output_key: str = (
        "temporal_intervals"  # Key in frame.data['meta'] for current intervals
    )
    emit_on_change: bool = True  # Emit interval data on state changes
    emit_on_complete: bool = True  # Emit all intervals on filter shutdown

    # Event sink integration
    # When enabled, emits state changes on a dedicated 'events' topic for filter-event-sink
    emit_event_topic: bool = False  # Emit events on dedicated topic for event sink
    event_topic_name: str = "events"  # Topic name for event emission

    # Debug
    debug: bool = False


class TemporalIntervalFilter(Filter):
    """
    Filter that tracks temporal detection intervals using EMA smoothing.

    This filter consumes detection results from upstream filters (like SAM3)
    and tracks presence/absence of detection classes over time. It uses
    Exponential Moving Average (EMA) smoothing to convert noisy per-frame
    detections into stable temporal intervals.

    Use cases:
    - Generating ground truth data for assertion testing
    - Temporal aggregation of detection events
    - Hysteresis-based presence detection

    The filter outputs intervals in a format suitable for assertion testing:
    [
        {"start_frame": 0, "end_frame": 45, "label": "person", "present": false, "confidence": 0.1},
        {"start_frame": 45, "end_frame": 120, "label": "person", "present": true, "confidence": 0.85},
        ...
    ]
    """

    @classmethod
    def normalize_config(cls, config):
        """Normalize and validate configuration."""
        config = super().normalize_config(config)

        # Convert string values
        for key in [
            "half_life",
            "full_decay_life",
            "presence_threshold",
            "min_confidence",
        ]:
            if isinstance(config.get(key), str):
                val = config[key].strip()
                config[key] = float(val) if val else None

        if isinstance(config.get("emit_on_change"), str):
            config["emit_on_change"] = config["emit_on_change"].lower() in (
                "true",
                "1",
                "yes",
            )
        if isinstance(config.get("emit_on_complete"), str):
            config["emit_on_complete"] = config["emit_on_complete"].lower() in (
                "true",
                "1",
                "yes",
            )
        if isinstance(config.get("emit_event_topic"), str):
            config["emit_event_topic"] = config["emit_event_topic"].lower() in (
                "true",
                "1",
                "yes",
            )
        if isinstance(config.get("debug"), str):
            config["debug"] = config["debug"].lower() in ("true", "1", "yes")

        # Validate thresholds
        presence_threshold = config.get("presence_threshold", 0.5)
        if not (0.0 <= presence_threshold <= 1.0):
            raise ValueError(
                f"presence_threshold must be between 0 and 1, got {presence_threshold}"
            )

        min_confidence = config.get("min_confidence", 0.0)
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {min_confidence}"
            )

        return TemporalIntervalConfig(**config)

    def setup(self, config: TemporalIntervalConfig):
        """Initialize the filter using reusable IntervalTracker."""
        logger.info(f"TemporalIntervalFilter setup: {config.clean()}")

        self.cfg = config
        self.debug = config.debug

        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Detection input config
        self.detection_key = config.detection_key
        self.score_field = config.score_field
        self.label_field = config.label_field
        self.default_label = config.default_label
        self.min_confidence = config.min_confidence

        # Output config
        self.output_key = config.output_key
        self.emit_on_change = config.emit_on_change
        self.emit_on_complete = config.emit_on_complete
        self.emit_event_topic = config.emit_event_topic
        self.event_topic_name = config.event_topic_name

        # Auto-enable streaming when an output path is requested and emit_on_complete
        # is set: IntervalTracker only opens the ndjson handle when streaming_mode is
        # True, so leaving it off would silently drop the file the caller asked for.
        self.interval_tracker = IntervalTracker(
            half_life=config.half_life,
            full_decay_life=config.full_decay_life,
            presence_threshold=config.presence_threshold,
            output_json_path=config.output_json_path
            if config.emit_on_complete
            else None,
            streaming_mode=bool(config.output_json_path and config.emit_on_complete),
        )

        logger.info(
            f"TemporalIntervalFilter initialized with alpha={self.interval_tracker.tracker.alpha:.4f}"
        )

    def shutdown(self):
        """Clean up and finalize intervals."""
        logger.info("TemporalIntervalFilter shutdown")
        self.interval_tracker.finalize()

    def process(self, frames: dict[str, Frame]) -> dict[str, Frame]:
        """Process frames and track detection intervals."""
        output_frames = {}

        for topic, frame in frames.items():
            if frame is None:
                continue

            # Get frame ID from metadata if available
            meta = frame.data.get("meta", {})
            frame_id = int(meta["id"]) if "id" in meta else None

            # Get detections from frame metadata
            detections = self._get_detections(frame)

            # Aggregate detections by label
            detected_labels = self._aggregate_detections(detections)

            # Update tracker and get state changes
            state_changes = self.interval_tracker.update(detected_labels, frame_id)

            if self.debug:
                current_frame = frame_id or self.interval_tracker.frame_count
                for label, score in detected_labels.items():
                    ema = self.interval_tracker.get_ema(label)
                    is_present = self.interval_tracker.is_present(label)
                    logger.debug(
                        f"[frame {current_frame}] {label}: detected=True, "
                        f"score={score:.3f}, ema={ema:.3f}, present={is_present}"
                    )

            # Add interval info to frame metadata if state changed
            if state_changes and self.emit_on_change:
                current_frame_id = frame_id or self.interval_tracker.frame_count
                state_change_data = {
                    "frame_id": current_frame_id,
                    "state_changes": [
                        {"label": label, "present": present, "ema": round(ema, 4)}
                        for label, present, ema in state_changes
                    ],
                }

                # Add to frame metadata (for downstream filters)
                meta = frame.data.setdefault("meta", {})
                meta[self.output_key] = state_change_data

                # Emit on dedicated event topic for filter-event-sink
                if self.emit_event_topic:
                    event_frame = Frame(data=state_change_data)
                    output_frames[self.event_topic_name] = event_frame

            output_frames[topic] = frame

        return output_frames

    def _get_detections(self, frame: Frame) -> list[dict]:
        """Extract detections from frame metadata or top-level data."""
        # Try new top-level frame.data["detections"] key first
        if "detections" in frame.data:
            dets_payload = frame.data["detections"]
            if isinstance(dets_payload, dict) and isinstance(
                dets_payload.get("items"), list
            ):
                return dets_payload["items"]
            elif isinstance(dets_payload, list):
                return dets_payload

        # Fallback to metadata keys
        meta = frame.data.get("meta", {})
        detections = meta.get(self.detection_key)
        if not isinstance(detections, list):
            # Check standard "detections" key as fallback in meta
            detections = meta.get("detections")
            if not isinstance(detections, list):
                return []
        return detections

    def _aggregate_detections(self, detections: list[dict]) -> dict[str, float]:
        """
        Aggregate detections by label, returning max confidence per label.

        Returns:
            Dict mapping label -> max confidence score
        """
        label_scores: dict[str, float] = {}

        for det in detections:
            if not isinstance(det, dict):
                continue

            # Get confidence score
            score = det.get(self.score_field)
            if score is None and self.score_field != "score":
                # Fallback to standard schema key if config specified legacy confidence alias
                score = det.get("score")
            if score is None:
                score = 1.0

            if score < self.min_confidence:
                continue

            # Get label
            if self.label_field and self.label_field in det:
                label = str(det[self.label_field])
            elif self.label_field and "label" in det:
                # Fallback to standard schema key if config specified legacy class/class_name alias
                label = str(det["label"])
            else:
                label = self.default_label

            # Track max score per label
            if label not in label_scores or score > label_scores[label]:
                label_scores[label] = score

        return label_scores

    def get_intervals(self) -> list[DetectionInterval]:
        """Get all completed intervals (for programmatic access)."""
        return self.interval_tracker.get_intervals()

    def get_current_state(self) -> dict[str, dict]:
        """Get current presence state for all tracked labels."""
        return self.interval_tracker.get_current_state()


if __name__ == "__main__":
    TemporalIntervalFilter.run()
