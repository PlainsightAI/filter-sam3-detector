# API Reference

Complete API documentation for the SAM3 detector filter.

## FilterSAM3DetectorConfig

Configuration class for the SAM3 detector filter.

### Class Definition

```python
class FilterSAM3DetectorConfig(FilterConfig):
    """Configuration for SAM3 object detection filter."""
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `"facebook/sam2-hiera-large"` | HuggingFace model ID or local path to model checkpoint |
| `device` | `str` | `"cuda"` | Device to run inference on: `"cuda"`, `"cpu"`, or `"mps"` |
| `text_prompt` | `str \| None` | `None` | Natural language text prompt for detection (e.g., "person", "car") |
| `exemplars_path` | `str \| None` | `None` | Path to directory containing exemplar images (jpg/png) |
| `exemplar_embeddings_cache` | `str \| None` | `None` | Path to cache file for exemplar embeddings (optional) |
| `positive_boxes` | `list \| None` | `None` | Reference boxes (positive): list of [x, y, w, h] in pixels |
| `negative_boxes` | `list \| None` | `None` | Reference boxes (negative): list of [x, y, w, h] in pixels |
| `confidence_threshold` | `float` | `0.5` | Minimum confidence score for detections (0.0-1.0) |
| `mask_threshold` | `float` | `0.5` | Threshold for mask binarization (0.0-1.0) |
| `max_detections` | `int` | `100` | Maximum number of detections per frame |
| `output_masks` | `bool` | `True` | Whether to output segmentation masks |
| `output_boxes` | `bool` | `True` | Whether to output bounding boxes |
| `output_scores` | `bool` | `True` | Whether to output confidence scores |
| `output_label` | `str` | `"sam3_detections"` | Key for storing results in `frame.data['meta']` |
| `visualize` | `bool` | `False` | Whether to draw detections on output frames |
| `debug` | `bool` | `False` | Enable debug logging |

### Environment Variables

All parameters can be set via environment variables with the `FILTER_` prefix:

```bash
export FILTER_TEXT_PROMPT="person"
export FILTER_CONFIDENCE_THRESHOLD=0.7
export FILTER_DEVICE=cuda
```

## FilterSAM3Detector

Main filter class for SAM3 object detection.

### Class Definition

```python
class FilterSAM3Detector(Filter):
    """
    SAM3 object detection filter.
    
    This filter performs open-set object detection using SAM3 (Segment Anything Model 3).
    It supports two prompting modes:
    - Text prompts: Natural language descriptions (e.g., "person", "car")
    - Image exemplars: Few-shot learning with cropped example images
    """
```

### Methods

#### `normalize_config(config: FilterConfig) -> FilterConfig`

Normalize and validate configuration parameters.

**Parameters:**
- `config` (FilterConfig): Input configuration dictionary

**Returns:**
- `FilterConfig`: Normalized and validated configuration

**Raises:**
- `ValueError`: If configuration parameters are invalid

**Note:** This method is idempotent - calling it multiple times produces the same result.

#### `setup(config: FilterSAM3DetectorConfig) -> None`

Initialize the filter with the given configuration.

**Parameters:**
- `config` (FilterSAM3DetectorConfig): Filter configuration

**Actions:**
- Loads the SAM3 model from HuggingFace or local path
- Loads and processes exemplar images if provided
- Computes exemplar embeddings
- Initializes device (CUDA/CPU/MPS)

**Raises:**
- `ImportError`: If SAM3 dependencies are not available
- `RuntimeError`: If model loading fails

#### `process(frames: dict[str, Frame]) -> dict[str, Frame]`

Process input frames and detect objects.

**Parameters:**
- `frames` (dict[str, Frame]): Dictionary of input frames keyed by topic name

**Returns:**
- `dict[str, Frame]`: Dictionary of output frames with detection results

**Output Format:**
Detections are stored in `frame.data['meta'][output_label]`:

```python
[
  {
    "box": [x1, y1, x2, y2],  # Bounding box coordinates
    "score": 0.95,            # Confidence score (0.0-1.0)
    "mask": [[...]]           # Binary mask as 2D array (if output_masks=True)
  },
  ...
]
```

#### `shutdown() -> None`

Clean up resources when the filter is stopped.

**Actions:**
- Releases model resources
- Clears GPU memory cache
- Closes file handles

#### `run(config: dict[str, Any] | None = None, **kwargs) -> None`

Run the filter standalone until it exits.

**Parameters:**
- `config` (dict | None): Configuration dictionary (if None, uses environment variables)
- `**kwargs`: Additional arguments passed to Filter.run()

**Usage:**
```python
from filter_sam3_detector import FilterSAM3Detector

# Run with default config from environment
FilterSAM3Detector.run()

# Run with explicit config
FilterSAM3Detector.run({
    "text_prompt": "person",
    "confidence_threshold": 0.7,
    "sources": "tcp://127.0.0.1:5555",
    "outputs": ["tcp://127.0.0.1:5556"],
})
```

## Frame Data Structure

### Input Frame

Frames must have image data accessible via `frame.rw_bgr.image`:

```python
frame.rw_bgr.image  # numpy array in BGR format (H, W, 3)
frame.has_image     # bool: whether frame contains image data
```

### Output Frame

Detections are added to frame metadata:

```python
frame.data['meta'][output_label] = [
    {
        "box": [x1, y1, x2, y2],
        "score": 0.95,
        "mask": [[...]]  # Optional, if output_masks=True
    }
]
```

## Error Handling

The filter handles errors gracefully:

- **Missing model**: Forwards frames unchanged with warning
- **No prompts**: Forwards frames unchanged with warning
- **Processing errors**: Logs error and forwards frame unchanged
- **Invalid configuration**: Raises `ValueError` during setup

## Examples

### Basic Usage

```python
from filter_sam3_detector import FilterSAM3Detector
from openfilter.filter_runtime.filter import Filter

filters = [
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",
        "confidence_threshold": 0.5,
    }),
]

Filter.run_multi(filters)
```

### With Exemplars

```python
filters = [
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "exemplars_path": "./cup_examples/",
        "confidence_threshold": 0.3,
    }),
]
```

### With Reference Boxes

```python
filters = [
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",  # optional when using ref boxes
        "positive_boxes": [[480, 290, 110, 360], [370, 280, 115, 375]],
        "negative_boxes": [[100, 100, 50, 200]],
        "visualize": True,  # green=positive ref, red=negative ref, blue=detections
    }),
]
```

### Custom Output Label

```python
filters = [
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "car",
        "output_label": "vehicle_detections",
    }),
]
```

