# How to Install and Run

Quick guide to install and run filter-sam3-detector.

## 1. Installation

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd filter-sam3-detector

# Install the package
uv pip install -e .

# Or with pip
pip install -e .
```

## 2. Simple Script - Text-based Detection

### Basic Example

```bash
# Detect people in a video
python scripts/filter_object_detection.py \
    --video your_video.mp4 \
    --prompt "person" \
    --output-dir ./results
```

### Main Parameters

- `--video`: Path to input video file
- `--prompt`: Text describing what to detect (e.g., "person", "car", "dog")
- `--output-dir`: Directory to save results
- `--confidence`: Confidence threshold (default: 0.5)
- `--device`: Device ("cuda", "cpu", or "mps")
- `--visualize`: Draw detections on frames (optional)

### Practical Examples

```bash
# Detect people
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --prompt "person" \
    --output-dir ./people

# Detect cars with visualization
python scripts/filter_object_detection.py \
    --video traffic.mp4 \
    --prompt "car" \
    --confidence 0.6 \
    --visualize \
    --output-dir ./cars

# Use CPU (if no GPU available)
python scripts/filter_object_detection.py \
    --video video.mp4 \
    --prompt "dog" \
    --device cpu \
    --output-dir ./dogs
```

## 3. Exemplar-based Script (Few-Shot)

For objects that are hard to describe with text:

```bash
# 1. Prepare exemplars (cropped images of the object)
mkdir exemplars
# Add .jpg or .png images to exemplars/

# 2. Run detection
python scripts/filter_exemplar_detection.py \
    --video video.mp4 \
    --exemplars ./exemplars/ \
    --output-dir ./results \
    --confidence 0.3
```

## 4. Verify Installation

```bash
# Test if it's installed
python -c "from filter_sam3_detector import FilterSAM3Detector; print('✓ OK!')"

# See script help
python scripts/filter_object_detection.py --help
```

## 5. First Run

On the first run, the model will be downloaded from HuggingFace (~2-3GB):
- Requires internet connection
- May take a few minutes
- Downloaded automatically

## 6. Results

Results are saved in `output-dir/`:

```
results/
├── detections.jsonl    # Detections in JSONL format
└── frames/            # Annotated frames (if --visualize)
    ├── 00000.jpg
    ├── 00001.jpg
    └── ...
```

## Troubleshooting

### Error: "No module named 'filter_sam3_detector'"
```bash
# Reinstall
uv pip install -e . --force-reinstall
```

### Error: "CUDA out of memory"
```bash
# Use CPU or reduce resolution
--device cpu
# Or add --resize 480
```

### Error: "Model not found"
```bash
# Check internet connection
# Model will be downloaded automatically on first run
```

### Too slow
```bash
# Use GPU if available
--device cuda

# Or reduce resolution
--resize 480
```

## Next Steps

- See [README.md](README.md) for complete documentation
- See [QUICKSTART.md](QUICKSTART.md) for more examples
- See [scripts/README.md](scripts/README.md) for script details
