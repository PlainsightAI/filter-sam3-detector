# filter-sam3-detector

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/your-org/filter-sam3-detector/blob/main/LICENSE)

OpenFilter implementation for SAM3 (Segment Anything Model 3) object detection with open-set capabilities.

## Features

- **Open-Set Detection**: Detect objects not in standard training datasets
- **Dual Prompting Modes**: Text prompts or exemplar images (few-shot learning)
- **Flexible Output**: Bounding boxes, segmentation masks, and confidence scores
- **GPU Acceleration**: CUDA, CPU, and MPS (Apple Silicon) support
- **Real-time Processing**: Processes video streams in real-time
- **Pipeline Integration**: Works seamlessly with OpenFilter pipeline architecture
- **Environment Configuration**: Full configuration through environment variables
- **Performance Optimized**: Configurable detection limits, resolution control
- **Fault Tolerant**: Handles errors gracefully, forwards frames on failure
- **Cost Efficient**: Local inference, no API costs

## Architecture

The filter follows the OpenFilter pattern with three main stages:

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| `setup()` | Load SAM3 model from HuggingFace; load and process exemplar images; initialize device (CUDA/CPU/MPS) |
| `process()` | Core operation: run SAM3 inference on frames; extract detections; attach results to frame metadata |
| `shutdown()` | Clean up resources (release model, clear GPU memory) when filter stops |

### Data Signature

The filter returns processed frames with the following data structure:

**Frame Metadata:**
- Original frame data preserved
- Detection results added to `frame.data['meta'][output_label]`:
  ```python
  [
    {
      "box": [x1, y1, x2, y2],  # Bounding box coordinates
      "score": 0.95,            # Confidence score (0.0-1.0)
      "mask": [[...]]           # Binary mask as 2D array (optional)
    },
    ...
  ]
  ```

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

**Quick install:**
```bash
# Clone repository
git clone <repository-url>
cd filter-sam3-detector

# Install package
uv pip install -e .

# Or with development dependencies
uv pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` file with your configuration:
```bash
# Prompt configuration (choose one)
FILTER_TEXT_PROMPT=person                    # Text prompt for detection
FILTER_EXEMPLARS_PATH=./exemplars/           # Path to exemplar images directory

# Model configuration
FILTER_MODEL_ID=facebook/sam2-hiera-large    # HuggingFace model ID
FILTER_DEVICE=cuda                           # Device: cuda, cpu, or mps

# Detection parameters
FILTER_CONFIDENCE_THRESHOLD=0.5              # Minimum confidence (0.0-1.0)
FILTER_MASK_THRESHOLD=0.5                    # Mask binarization threshold
FILTER_MAX_DETECTIONS=100                    # Maximum detections per frame

# Output configuration
FILTER_OUTPUT_MASKS=true                     # Output segmentation masks
FILTER_OUTPUT_BOXES=true                     # Output bounding boxes
FILTER_OUTPUT_SCORES=true                    # Output confidence scores
FILTER_OUTPUT_LABEL=sam3_detections          # Key in frame.data['meta']

# Visualization and debugging
FILTER_VISUALIZE=false                       # Draw detections on frames
FILTER_DEBUG=false                           # Enable debug logging
```

### Configuration Matrix

| Variable | Type | Default | Required | Notes |
|----------|------|---------|----------|-------|
| `text_prompt` | string | None | No* | Natural language description (e.g., "person", "car") |
| `exemplars_path` | string | None | No* | Path to directory with exemplar images |
| `model_id` | string | "facebook/sam2-hiera-large" | No | HuggingFace model ID or local path |
| `device` | string | "cuda" | No | Device: "cuda", "cpu", or "mps" |
| `confidence_threshold` | float | 0.5 | No | Minimum confidence (0.0-1.0) |
| `mask_threshold` | float | 0.5 | No | Mask binarization threshold (0.0-1.0) |
| `max_detections` | int | 100 | No | Maximum detections per frame |
| `output_masks` | bool | true | No | Output segmentation masks |
| `output_boxes` | bool | true | No | Output bounding boxes |
| `output_scores` | bool | true | No | Output confidence scores |
| `output_label` | string | "sam3_detections" | No | Key for storing results |
| `visualize` | bool | false | No | Draw detections on output frames |
| `debug` | bool | false | No | Enable debug logging |

\* Either `text_prompt` or `exemplars_path` must be provided.

## Usage

### Method 1: Using Example Scripts (Recommended)

#### Object Detection with Text Prompts

```bash
# Detect people in a video
python scripts/filter_object_detection.py \
    --video input.mp4 \
    --prompt "person" \
    --output-dir ./results \
    --confidence 0.5

# Detect cars with visualization
python scripts/filter_object_detection.py \
    --video traffic.mp4 \
    --prompt "car" \
    --confidence 0.6 \
    --visualize \
    --output-dir ./cars

# Process multiple videos
python scripts/filter_object_detection.py \
    --video video1.mp4 video2.mp4 video3.mp4 \
    --prompt "dog" \
    --output-dir ./detections
```

#### Exemplar-Based Detection (Few-Shot Learning)

```bash
# Prepare exemplar images first
mkdir -p cup_examples
# Add cropped images of cups to cup_examples/

# Run detection
python scripts/filter_exemplar_detection.py \
    --video input.mp4 \
    --exemplars ./cup_examples/ \
    --output-dir ./results \
    --confidence 0.3
```

**Exemplar Directory Structure:**
```
cup_examples/
├── cup1.jpg
├── cup2.jpg
├── cup3.png
└── ...
```

Each image should show exactly one instance of the object you want to detect.

### Method 2: Using as a Standalone Filter

```bash
# Set environment variables
export FILTER_TEXT_PROMPT="person"
export FILTER_CONFIDENCE_THRESHOLD=0.7
export FILTER_DEVICE=cuda
export FILTER_SOURCES="tcp://127.0.0.1:5555"
export FILTER_OUTPUTS="tcp://127.0.0.1:5556"

# Run the filter
filter-sam3-detector
```

### Method 3: Using in Python Code

```python
from filter_sam3_detector import FilterSAM3Detector
from openfilter.filter_runtime.filter import Filter

# Define pipeline
filters = [
    ("VideoIn", {
        "sources": "file://input.mp4",
        "outputs": ["tcp://127.0.0.1:5555"],
    }),
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",
        "confidence_threshold": 0.5,
        "device": "cuda",
    }),
    ("Recorder", {
        "sources": "tcp://127.0.0.1:5556",
        "path": "detections.jsonl",
        "format": "jsonl",
    }),
]

# Run pipeline
runner = Filter.Runner(filters)
runner.join()
```

## Usage Scenarios

### 1. Person Detection

Detect people in surveillance videos:

```bash
python scripts/filter_object_detection.py \
    --video surveillance.mp4 \
    --prompt "person" \
    --output-dir ./person_detections \
    --confidence 0.6
```

### 2. Vehicle Detection

Detect cars in traffic monitoring:

```bash
python scripts/filter_object_detection.py \
    --video traffic.mp4 \
    --prompt "car" \
    --confidence 0.6 \
    --output-dir ./vehicle_detections \
    --resize 480
```

### 3. Custom Object Detection with Exemplars

For objects that are hard to describe with text:

```bash
# Prepare exemplar images
mkdir -p custom_objects
# Add cropped images to custom_objects/

python scripts/filter_exemplar_detection.py \
    --video assembly_line.mp4 \
    --exemplars ./custom_objects/ \
    --confidence 0.3 \
    --output-dir ./custom_detections
```

### 4. Pipeline Integration

Combine with other OpenFilter filters:

```python
from openfilter.filter_runtime.filter import Filter
from filter_sam3_detector import FilterSAM3Detector

filters = [
    ("VideoIn", {"sources": "file://input.mp4"}),
    ("Resize", {"width": 640, "height": 480}),  # Pre-processing
    (FilterSAM3Detector, {"text_prompt": "person"}),
    ("Recorder", {"path": "output.jsonl"}),
]

Filter.Runner(filters).join()
```

## Output Format

Detections are stored in `frame.data['meta'][output_label]` and saved to JSONL when `output_path` is configured:

### JSONL Format (from FilterSAM3Detector)

```json
{
  "filename": "frame_000001_ts1767057416_677_count000001.jpg",
  "num_detections": 2,
  "meta": {
    "detections": [
      {
        "id": 1,
        "class": "person",
        "rois": [[0.1, 0.2, 0.5, 0.8]],
        "bbox": [100.0, 150.0, 200.0, 300.0],
        "box": [100, 150, 300, 450],
        "score": 0.95,
        "segmentation": [[...]],
        "area": 60000,
        "category_id": 1,
        "iscrowd": 0
      }
    ]
  }
}
```

### Recorder Output Format

When using the Recorder filter, the output format includes frame metadata:

```json
{
  "main": {
    "meta": {
      "id": 0,
      "ts": 1767126697.8538108,
      "src": "file:///path/to/video.mp4",
      "src_fps": 19.996322290984892,
      "detections": [
        {
          "id": 1,
          "class": "person",
          "rois": [[0.0005208333333333333, 0.0, 0.29244791666666664, 0.4152777777777778]],
          "bbox": [2.84, 0.0, 1120.67, 897.20],
          "box": [2, 0, 1123, 897],
          "score": 0.9753850698471069,
          "category_id": 1
        }
      ],
      "detection_confidence": 0.9753850698471069
    }
  }
}
```

### Field Descriptions

- **`class`**: Class name from text prompt (e.g., "person", "car")
- **`rois`**: Normalized bounding box coordinates `[[x1, y1, x2, y2]]` (values between 0 and 1)
- **`bbox`**: COCO format `[x, y, width, height]` in pixels
- **`box`**: Absolute coordinates `[x1, y1, x2, y2]` in pixels
- **`score`**: Confidence score (0.0-1.0)
- **`segmentation`**: COCO polygon format (if `output_masks=true`)
- **`area`**: Mask area in pixels (if masks are enabled)
- **`category_id`**: Category ID (default: 1)
- **`detection_confidence`**: Maximum confidence score across all detections (in meta)

## Performance Tips

### Image Processing
- **Resize Videos**: Use `--resize 480` for faster processing
- **Limit Detections**: Reduce `FILTER_MAX_DETECTIONS` for better performance
- **Disable Masks**: Set `FILTER_OUTPUT_MASKS=false` to save memory

### Device Selection
- **Use GPU**: Set `FILTER_DEVICE=cuda` for 10-50x speedup
- **CPU Fallback**: Automatically falls back to CPU if GPU unavailable
- **Apple Silicon**: Use `FILTER_DEVICE=mps` on macOS

### Confidence Thresholds
- **Text Prompts**: Default `0.5` works well
- **Exemplar-Based**: Use `0.3` for better recall
- **High Precision**: Use `0.7` or higher
- **High Recall**: Use `0.3` or lower

## Development

### Project Structure

```
filter-sam3-detector/
├── filter_sam3_detector/
│   ├── __init__.py
│   └── filter.py              # Main filter implementation
├── scripts/                   # Example usage scripts
│   ├── filter_object_detection.py
│   ├── filter_exemplar_detection.py
│   └── README.md
├── examples/                  # Additional examples
│   └── detect_objects_video.py
├── docs/                      # Documentation
│   ├── API.md
│   ├── configuration.md
│   ├── advanced-usage.md
│   └── performance.md
├── tests/                     # Test files
│   ├── test_filter.py
│   └── test_integration.py
├── sam3/                      # Vendorized SAM3 library
├── env.example               # Environment configuration example
└── pyproject.toml           # Project dependencies
```

### Key Dependencies

- `openfilter[all]>=0.1.0` - Filter framework
- `torch>=2.0.0` - PyTorch for model inference
- `torchvision>=0.15.0` - Image processing
- `transformers>=4.40.0` - HuggingFace model loading
- `opencv-python>=4.8.0` - Image manipulation
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations

### Testing

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Check code quality
make lint

# Format code
make format
```

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load or takes too long

**Solutions**:
- Ensure you have sufficient GPU memory (recommended: 8GB+)
- Use CPU mode if GPU is unavailable: `--device cpu`
- Check internet connection (model downloads from HuggingFace on first use)
- Verify CUDA installation: `nvidia-smi`

### No Detections Found

**Problem**: Filter runs but finds no objects

**Solutions**:
- Lower confidence threshold: `--confidence 0.3`
- Try different text prompts (be more specific or more general)
- For exemplar-based: ensure exemplar images are clear and representative
- Check that input video has the objects you're looking for

### Out of Memory Errors

**Problem**: CUDA out of memory errors

**Solutions**:
- Resize input: `--resize 480`
- Reduce max detections: `export FILTER_MAX_DETECTIONS=50`
- Disable masks: `export FILTER_OUTPUT_MASKS=false`
- Use CPU mode: `--device cpu` (slower but uses less memory)

### Import Errors

**Problem**: `ImportError: cannot import name 'FilterSAM3Detector'`

**Solutions**:
- Ensure package is installed: `uv pip install -e .`
- Check Python version (requires 3.10+)
- Verify all dependencies are installed
- Reinstall: `uv pip install -e . --force-reinstall`

### Slow Processing

**Problem**: Processing is very slow

**Solutions**:
- Use GPU: `--device cuda`
- Resize videos: `--resize 480`
- Reduce max detections
- Disable masks if not needed
- Process fewer frames (use sample rate in video input)

### Performance Optimization

To improve processing speed:
1. Use GPU acceleration (`FILTER_DEVICE=cuda`)
2. Resize inputs to appropriate resolution (`--resize 480`)
3. Limit detections (`FILTER_MAX_DETECTIONS=50`)
4. Disable unused outputs (masks if not needed)
5. Use smaller model variant (if available)

## Documentation

For more detailed information, configuration examples, and advanced usage scenarios, see the comprehensive documentation:

- [Installation Guide](INSTALL.md) - Detailed installation instructions
- [Quick Start Guide](QUICKSTART.md) - Get started in minutes
- [API Reference](docs/API.md) - Complete API documentation
- [Configuration Guide](docs/configuration.md) - Configuration options
- [Advanced Usage](docs/advanced-usage.md) - Advanced patterns and examples
- [Performance Tuning](docs/performance.md) - Optimization guide
- [Scripts Documentation](scripts/README.md) - Example scripts usage

## License

See LICENSE file for details.

## References

- [SAM3 Paper](https://arxiv.org/abs/2406.05663)
- [OpenFilter Documentation](https://openfilter.io)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
