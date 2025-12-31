# Using .env File

Guide on how to configure the filter using environment variables and `.env` file.

## Quick Start

1. **Copy the example file:**
```bash
cp env.example .env
```

2. **Edit `.env` with your settings:**
```bash
# Edit the file with your preferred editor
nano .env
# or
vim .env
```

3. **Use the filter** - Environment variables are automatically loaded!

## Environment Variables

All configuration parameters can be set via environment variables with the `FILTER_` prefix.

### Basic Configuration

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
```

### Output Configuration

```bash
# Output options
FILTER_OUTPUT_MASKS=true                     # Output segmentation masks
FILTER_OUTPUT_BOXES=true                     # Output bounding boxes
FILTER_OUTPUT_SCORES=true                    # Output confidence scores
FILTER_OUTPUT_LABEL=sam3_detections          # Key in frame.data['meta']
```

### Visualization and Debugging

```bash
FILTER_VISUALIZE=false                       # Draw detections on frames
FILTER_DEBUG=false                           # Enable debug logging
```

## Example .env File

Create a `.env` file in the project root:

```bash
# .env file example

# Text prompt for detection
FILTER_TEXT_PROMPT=person

# Or use exemplars instead
# FILTER_EXEMPLARS_PATH=./exemplars/

# Device configuration
FILTER_DEVICE=cuda

# Detection settings
FILTER_CONFIDENCE_THRESHOLD=0.6
FILTER_MAX_DETECTIONS=50

# Output settings
FILTER_OUTPUT_MASKS=true
FILTER_OUTPUT_BOXES=true
FILTER_OUTPUT_SCORES=true
FILTER_OUTPUT_LABEL=detections

# Visualization
FILTER_VISUALIZE=true
FILTER_DEBUG=false
```

## Usage Methods

### Method 1: Using .env File (Recommended)

1. Create `.env` file:
```bash
cp env.example .env
```

2. Edit `.env` with your settings

3. Run scripts - they automatically load `.env`:
```bash
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --output-dir ./results
```

The script will use `FILTER_TEXT_PROMPT` from `.env` if `--prompt` is not provided.

### Method 2: Export Environment Variables

```bash
# Set variables in your shell
export FILTER_TEXT_PROMPT="person"
export FILTER_DEVICE="cuda"
export FILTER_CONFIDENCE_THRESHOLD=0.6

# Run filter
filter-sam3-detector
```

### Method 3: Inline with Command

```bash
FILTER_TEXT_PROMPT="person" FILTER_DEVICE="cpu" \
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --output-dir ./results
```

## Using .env with Scripts

### Script 1: filter_object_detection.py

The script can use `.env` for default values, but command-line arguments override them:

```bash
# Uses FILTER_TEXT_PROMPT from .env
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --output-dir ./results

# Overrides FILTER_TEXT_PROMPT from .env
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --prompt "car" \
    --output-dir ./results
```

### Script 2: filter_exemplar_detection.py

```bash
# Uses FILTER_EXEMPLARS_PATH from .env
python scripts/filter_exemplar_detection.py \
    --video video.mp4 \
    --output-dir ./results

# Overrides FILTER_EXEMPLARS_PATH from .env
python scripts/filter_exemplar_detection.py \
    --video video.mp4 \
    --exemplars ./custom_exemplars/ \
    --output-dir ./results
```

### Standalone Filter (CLI)

When using the filter as a standalone command, `.env` is automatically loaded:

```bash
# Set in .env or export
FILTER_TEXT_PROMPT=person
FILTER_SOURCES=tcp://127.0.0.1:5555
FILTER_OUTPUTS=tcp://127.0.0.1:5556

# Run
filter-sam3-detector
```

## Python Code Usage

When using in Python code, you can load `.env` manually:

```python
from dotenv import load_dotenv
from filter_sam3_detector import FilterSAM3Detector

# Load .env file
load_dotenv()

# Filter will automatically use FILTER_* environment variables
filter_instance = FilterSAM3Detector()
config = FilterSAM3Detector.normalize_config({})
filter_instance.setup(config)
```

## Variable Priority

Configuration is applied in this order (later overrides earlier):

1. **Default values** (from `normalize_config`)
2. **Environment variables** (`FILTER_*`)
3. **Config dict** (passed to `normalize_config`)
4. **Command-line arguments** (in scripts)

## Complete Variable List

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FILTER_TEXT_PROMPT` | string | None | Text prompt for detection |
| `FILTER_EXEMPLARS_PATH` | string | None | Path to exemplar images |
| `FILTER_MODEL_ID` | string | "facebook/sam2-hiera-large" | HuggingFace model ID |
| `FILTER_DEVICE` | string | "cuda" | Device: cuda, cpu, mps |
| `FILTER_CONFIDENCE_THRESHOLD` | float | 0.5 | Confidence threshold |
| `FILTER_MASK_THRESHOLD` | float | 0.5 | Mask binarization threshold |
| `FILTER_MAX_DETECTIONS` | int | 100 | Max detections per frame |
| `FILTER_OUTPUT_MASKS` | bool | true | Output masks |
| `FILTER_OUTPUT_BOXES` | bool | true | Output boxes |
| `FILTER_OUTPUT_SCORES` | bool | true | Output scores |
| `FILTER_OUTPUT_LABEL` | string | "sam3_detections" | Output label key |
| `FILTER_VISUALIZE` | bool | false | Draw detections |
| `FILTER_DEBUG` | bool | false | Debug logging |

## Examples

### Example 1: Person Detection

```bash
# .env
FILTER_TEXT_PROMPT=person
FILTER_DEVICE=cuda
FILTER_CONFIDENCE_THRESHOLD=0.6
FILTER_VISUALIZE=true
```

```bash
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --output-dir ./people
```

### Example 2: CPU Mode

```bash
# .env
FILTER_TEXT_PROMPT=car
FILTER_DEVICE=cpu
FILTER_CONFIDENCE_THRESHOLD=0.5
FILTER_MAX_DETECTIONS=50
```

### Example 3: Exemplar-based Detection

```bash
# .env
FILTER_EXEMPLARS_PATH=./custom_objects/
FILTER_DEVICE=cuda
FILTER_CONFIDENCE_THRESHOLD=0.3
FILTER_VISUALIZE=true
```

```bash
python scripts/filter_exemplar_detection.py \
    --video video.mp4 \
    --output-dir ./results
```

## Troubleshooting

### .env file not loading

Make sure:
- File is named exactly `.env` (with the dot)
- File is in the project root directory
- You're running from the project root

### Variables not being used

Check:
- Variable names start with `FILTER_`
- Variable names are uppercase
- Values are correct type (string, number, boolean)

### Boolean values

For boolean variables, use:
- `true`, `1`, `yes` → True
- `false`, `0`, `no` → False

```bash
FILTER_VISUALIZE=true
FILTER_DEBUG=false
```

## Best Practices

1. **Don't commit `.env`** - Add to `.gitignore`
2. **Use `env.example`** - Keep example file updated
3. **Document custom variables** - Add comments in `.env`
4. **Use different `.env` files** - For different environments (dev, prod)

## See Also

- [env.example](env.example) - Example environment file
- [README.md](README.md) - Full documentation
- [docs/configuration.md](docs/configuration.md) - Configuration guide

