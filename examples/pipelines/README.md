# Pipeline Examples

This directory contains docker-compose pipeline examples demonstrating different detection modes for the SAM3 detector filter.

## Pipelines

| Pipeline | Description | Use Case |
|----------|-------------|----------|
| `raw-detections.yaml` | Per-frame detection output | Training data, frame analysis, custom post-processing |
| `temporal-intervals.yaml` | EMA-smoothed presence intervals | Ground truth generation, activity detection, event triggers |

## Raw Detections Pipeline

Outputs per-frame detection results with bounding boxes and confidence scores. Each frame contains all detections found by SAM3.

```bash
# Basic usage
docker compose -f examples/pipelines/raw-detections.yaml up

# Custom prompt and threshold
TEXT_PROMPT="car" CONFIDENCE_THRESHOLD=0.7 \
  docker compose -f examples/pipelines/raw-detections.yaml up

# With custom video
VIDEO_PATH=/path/to/video.mp4 \
  docker compose -f examples/pipelines/raw-detections.yaml up
```

**Output format** (in `frame.data['meta']['sam3_detections']`):
```json
[
  {"box": [100, 50, 300, 400], "score": 0.95, "label": "person"},
  {"box": [500, 100, 650, 350], "score": 0.87, "label": "person"}
]
```

## Temporal Intervals Pipeline

Uses dual-EMA (Exponential Moving Average) smoothing to convert noisy per-frame detections into stable presence/absence intervals. This is ideal for:

- **Ground truth generation**: Create stable intervals for testing assertions
- **Activity detection**: Detect when objects enter/leave the scene
- **Event-based triggers**: Fire events on state changes, not every frame
- **Noise reduction**: Eliminate frame-to-frame detection flickering

```bash
# Basic usage (default: 5-frame half-life)
docker compose -f examples/pipelines/temporal-intervals.yaml up

# Slower response (more smoothing)
HALF_LIFE=15 docker compose -f examples/pipelines/temporal-intervals.yaml up

# Specify full decay instead (99.3% decay time)
FULL_DECAY_LIFE=60 docker compose -f examples/pipelines/temporal-intervals.yaml up

# Both parameters for custom fast/slow EMA ratio
HALF_LIFE=5 FULL_DECAY_LIFE=60 \
  docker compose -f examples/pipelines/temporal-intervals.yaml up
```

**Output format** (in `frame.data['meta']['temporal_intervals']`):
```json
{
  "current_state": {"foreground": true},
  "ema_values": {"foreground": 0.85},
  "intervals": [
    {
      "label": "foreground",
      "present": true,
      "start_frame": 0,
      "end_frame": null,
      "confidence": 0.92
    }
  ]
}
```

### EMA Parameters Explained

The temporal filter uses dual EMAs for robust presence detection:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `HALF_LIFE` | Frames for fast EMA to decay 50% | 5 |
| `FULL_DECAY_LIFE` | Frames for slow EMA to decay 99.3% | Derived: 7.21 × half_life |
| `PRESENCE_THRESHOLD` | EMA value threshold for "present" state | 0.5 |
| `MIN_CONFIDENCE` | Minimum raw detection confidence | 0.3 |

**How it works:**
- **Fast EMA** (half_life): Quickly responds to detection changes, provides the signal value
- **Slow EMA** (full_decay_life): Used for threshold crossing to prevent state flickering

If you only specify one parameter, the other is derived mathematically:
- `full_decay_life ≈ 7.21 × half_life`
- `half_life ≈ 0.139 × full_decay_life`

### JSON Output

The temporal intervals filter can output completed intervals to a JSON file for ground truth:

```bash
OUTPUT_JSON=/path/to/intervals.json \
  docker compose -f examples/pipelines/temporal-intervals.yaml up
```

Output file format:
```json
{
  "intervals": [
    {"label": "foreground", "present": true, "start_frame": 0, "end_frame": 150, "confidence": 0.91},
    {"label": "foreground", "present": false, "start_frame": 151, "end_frame": 200, "confidence": 0.08},
    {"label": "foreground", "present": true, "start_frame": 201, "end_frame": 450, "confidence": 0.88}
  ]
}
```

## Environment Variables

Both pipelines support these common variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_PATH` | Path to input video file | `../../data/sample.mp4` |
| `TEXT_PROMPT` | What to detect | `person` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.5` (raw) / `0.3` (temporal) |
| `EXEMPLARS_PATH` | Path to exemplar images directory | `../../exemplars` |
| `WEBVIS_PORT` | Port for web visualization | `8001` |
| `LOG_LEVEL` | Logging verbosity | (default) |

## Viewing Results

Both pipelines include a web visualization service. Open your browser to:

```
http://localhost:8001
```

(Or the port specified by `WEBVIS_PORT`)
