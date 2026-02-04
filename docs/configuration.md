# Configuration Guide

Complete guide to configuring the SAM3 detector filter.

## Configuration Methods

The filter can be configured in three ways:

1. **Environment Variables** (recommended for standalone usage)
2. **Configuration Dictionary** (for programmatic usage)
3. **Command-line Arguments** (when using scripts)

## Environment Variables

All configuration parameters use the `FILTER_` prefix:

```bash
# Text prompt
export FILTER_TEXT_PROMPT="person"

# Exemplar images (alternative to text prompt)
export FILTER_EXEMPLARS_PATH="./cup_examples/"

# Reference image prompts (comma-separated paths; requires text prompt)
# Positive refs pasted bottom-left, negative bottom-right
# export FILTER_REF_IMAGES="/path/pos1.png,/path/pos2.png"
# export FILTER_REF_IMAGES_NEGATIVE="/path/neg1.png"
# export FILTER_REF_MARGIN=10   # optional, default 10
# export FILTER_REF_GAP=5       # optional, default 5

# Model configuration
export FILTER_MODEL_ID=facebook/sam2-hiera-large
export FILTER_DEVICE=cuda

# Detection parameters
export FILTER_CONFIDENCE_THRESHOLD=0.5
export FILTER_MASK_THRESHOLD=0.5
export FILTER_MAX_DETECTIONS=100

# Output configuration
export FILTER_OUTPUT_MASKS=true
export FILTER_OUTPUT_BOXES=true
export FILTER_OUTPUT_SCORES=true
export FILTER_OUTPUT_LABEL=sam3_detections

# Visualization and debugging
export FILTER_VISUALIZE=false
export FILTER_DEBUG=false
```

See `env.example` for a complete template.

## Configuration Dictionary

When using the filter programmatically:

```python
from filter_sam3_detector import FilterSAM3Detector

config = {
    "sources": "tcp://127.0.0.1:5555",
    "outputs": ["tcp://127.0.0.1:5556"],
    "text_prompt": "person",
    "confidence_threshold": 0.5,
    "device": "cuda",
    "visualize": True,
}

filter_instance = FilterSAM3Detector(config)
```

## Parameter Details

### Model Configuration

#### `model_id`

- **Type**: `str`
- **Default**: `"facebook/sam2-hiera-large"`
- **Description**: HuggingFace model ID or local path to model checkpoint
- **Examples**:
  - `"facebook/sam2-hiera-large"` (default)
  - `"/path/to/local/model.pt"`

#### `device`

- **Type**: `str`
- **Default**: `"cuda"`
- **Options**: `"cuda"`, `"cpu"`, `"mps"`
- **Description**: Device to run inference on
- **Notes**:
  - `"cuda"`: NVIDIA GPU (fastest, requires CUDA)
  - `"cpu"`: CPU inference (slower but universal)
  - `"mps"`: Apple Silicon GPU (macOS only)

### Prompt Configuration

#### `text_prompt`

- **Type**: `str | None`
- **Default**: `None`
- **Description**: Natural language text prompt for detection
- **Examples**:
  - `"person"`
  - `"car"`
  - `"small transparent cup"`
  - `"dog playing in park"`

**Best Practices:**
- Be specific but concise
- Use common object names
- Avoid overly complex descriptions

#### `exemplars_path`

- **Type**: `str | None`
- **Default**: `None`
- **Description**: Path to directory containing exemplar images for few-shot detection
- **Format**: Directory path with JPG/PNG images
- **Status**: ⚠️ **Experimental** - This feature is currently broken due to a bug in backbone output handling

**Requirements:**
- Each image should be a **pre-cropped** image showing exactly one instance of the target object
- Images should be tightly cropped around the object (no annotations needed)
- Supported formats: JPG, JPEG, PNG, BMP, WEBP
- More exemplars (3-5) generally improve accuracy

**How It Works:**
1. Each exemplar image is loaded and encoded through SAM3's backbone
2. The backbone features are globally averaged to create a single embedding per image
3. All exemplar embeddings are averaged together to create a visual prompt embedding
4. This visual prompt guides detection alongside or instead of text prompts

**Example Structure:**
```
cup_examples/
├── cup1.jpg    # Cropped image of a cup
├── cup2.jpg    # Another cropped cup image
├── cup3.png    # Different angle/lighting
└── ...
```

**Preparing Exemplar Images:**
1. Extract frames from a reference video or use reference images
2. Manually crop regions containing the target object
3. Ensure crops are clean (minimal background, object fills most of the image)
4. Use multiple exemplars with different angles/lighting for better generalization

**Note**: Either `text_prompt` or `exemplars_path` must be provided (or both). When using exemplars, a lower `confidence_threshold` (0.2-0.3) is recommended.

### Detection Parameters

#### `confidence_threshold`

- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0` to `1.0`
- **Description**: Minimum confidence score for detections
- **Recommendations**:
  - Text prompts: `0.5` (default)
  - Exemplar-based: `0.3` (lower recommended)
  - High precision: `0.7` or higher
  - High recall: `0.3` or lower

#### `mask_threshold`

- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0` to `1.0`
- **Description**: Threshold for mask binarization
- **Note**: Only used when `output_masks=True`

#### `max_detections`

- **Type**: `int`
- **Default**: `100`
- **Description**: Maximum number of detections per frame
- **Recommendations**:
  - Single object scenes: `10-20`
  - Crowded scenes: `50-100`
  - Performance optimization: Lower values process faster

### Non-Maximum Suppression (NMS)

NMS is used to suppress overlapping bounding boxes, keeping only the highest-confidence detection for each object.

#### `nms_enabled`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable Non-Maximum Suppression to filter overlapping detections
- **Note**: Highly recommended to keep enabled; without NMS, SAM3 may return ~100+ overlapping boxes per frame

#### `nms_threshold`

- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0` to `1.0`
- **Description**: IoU (Intersection over Union) threshold for NMS
- **Behavior**:
  - Lower values = more aggressive suppression (fewer boxes kept)
  - Higher values = less aggressive suppression (more boxes kept)
- **Recommendations**:
  - `0.3`: Very aggressive - use when objects are well-separated
  - `0.5`: Moderate (default) - good balance for most use cases
  - `0.7`: Conservative - use when objects may legitimately overlap

### Output Configuration

#### `output_masks`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to output segmentation masks
- **Note**: Masks are binary 2D arrays, can be memory-intensive

#### `output_boxes`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to output bounding boxes
- **Format**: `[x1, y1, x2, y2]` coordinates

#### `output_scores`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to output confidence scores

#### `output_label`

- **Type**: `str`
- **Default**: `"sam3_detections"`
- **Description**: Key for storing results in `frame.data['meta']`
- **Usage**: Change this to avoid conflicts with other filters

### Visualization and Debugging

#### `visualize`

- **Type**: `bool`
- **Default**: `False`
- **Description**: Draw bounding boxes and masks on output frames
- **Note**: Requires OpenCV, adds processing overhead

#### `debug`

- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable debug logging
- **Output**: Detailed logs including frame processing, detection counts, etc.

## Configuration Examples

### High Precision Detection

```python
config = {
    "text_prompt": "person",
    "confidence_threshold": 0.8,
    "max_detections": 20,
    "output_masks": False,  # Save memory
}
```

### High Recall Detection

```python
config = {
    "text_prompt": "car",
    "confidence_threshold": 0.3,
    "max_detections": 100,
}
```

### Exemplar-Based Detection

```python
config = {
    "exemplars_path": "./custom_objects/",
    "confidence_threshold": 0.3,  # Lower for exemplars
    "max_detections": 50,
}
```

### CPU-Only Configuration

```python
config = {
    "text_prompt": "person",
    "device": "cpu",
    "max_detections": 20,  # Reduce for CPU performance
}
```

### Memory-Optimized Configuration

```python
config = {
    "text_prompt": "person",
    "output_masks": False,  # Disable masks
    "max_detections": 30,   # Limit detections
}
```

## Validation

The filter validates configuration parameters:

- **Device**: Must be one of `"cuda"`, `"cpu"`, `"mps"`
- **Confidence threshold**: Must be between 0.0 and 1.0
- **Mask threshold**: Must be between 0.0 and 1.0
- **NMS threshold**: Must be between 0.0 and 1.0
- **Max detections**: Must be >= 1
- **Prompts**: At least one of `text_prompt` or `exemplars_path` must be provided

Invalid configurations raise `ValueError` during setup.

## Environment Variable Precedence

Environment variables override configuration dictionary values:

1. Environment variables (highest priority)
2. Configuration dictionary
3. Default values (lowest priority)

