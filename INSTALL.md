# Installation Guide

Complete installation guide for filter-sam3-detector.

## Prerequisites

### Required

- **Python**: 3.10, 3.11, or 3.12
- **Package Manager**: `uv` (recommended) or `pip`

### Optional but Recommended

- **CUDA**: For GPU acceleration (NVIDIA GPU with CUDA 11.8+)
- **Memory**: 16GB+ RAM for GPU usage, 8GB+ for CPU
- **Disk Space**: ~5GB for model files and dependencies

## Installation Methods

### Method 1: Using uv (Recommended)

`uv` is a fast Python package installer. Install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

Then install the package:

```bash
# Clone repository
git clone <repository-url>
cd filter-sam3-detector

# Install package
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Method 2: Using pip

```bash
# Clone repository
git clone <repository-url>
cd filter-sam3-detector

# Install package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 3: From Source (Development)

```bash
# Clone repository
git clone <repository-url>
cd filter-sam3-detector

# Install in development mode
uv pip install -e ".[dev]"

# Run tests to verify
make test
```

## Post-Installation

### Verify Installation

```bash
# Test Python import
python -c "from filter_sam3_detector import FilterSAM3Detector; print('✓ Installed successfully!')"

# Test CLI command
filter-sam3-detector --help

# Run tests
make test
```

### First Run

On first run, the model will be downloaded from HuggingFace (~2-3GB). This requires:
- Internet connection
- Sufficient disk space
- Patience (download may take a few minutes)

## GPU Setup (Optional)

### CUDA Installation

For NVIDIA GPUs:

1. **Install CUDA Toolkit** (11.8 or later)
   ```bash
   # Check CUDA version
   nvidia-smi
   ```

2. **Install PyTorch with CUDA support**
   ```bash
   # PyTorch will be installed automatically with the package
   # But you can verify CUDA support:
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Verify GPU Access**
   ```bash
   python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
   ```

### Apple Silicon (MPS)

For Apple Silicon Macs:

```bash
# MPS is automatically available if PyTorch supports it
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Troubleshooting Installation

### Import Errors

**Problem**: `ImportError: cannot import name 'FilterSAM3Detector'`

**Solution**:
```bash
# Reinstall the package
uv pip install -e . --force-reinstall
```

### CUDA Not Available

**Problem**: GPU not detected

**Solutions**:
1. Verify CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. Use CPU mode: Set `FILTER_DEVICE=cpu`

### Out of Memory During Installation

**Problem**: Installation fails due to memory

**Solutions**:
1. Close other applications
2. Install dependencies separately
3. Use `--no-cache-dir` flag

### Model Download Issues

**Problem**: Model fails to download from HuggingFace

**Solutions**:
1. Check internet connection
2. Set HuggingFace token if needed: `export HF_TOKEN=your_token`
3. Use VPN if HuggingFace is blocked in your region

## Development Setup

For contributing to the project:

```bash
# Install with all dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install

# Run tests
make test

# Check code quality
make lint

# Format code
make format
```

## Uninstallation

```bash
# Uninstall package
uv pip uninstall filter-sam3-detector

# Or with pip
pip uninstall filter-sam3-detector
```

## Next Steps

After installation:

1. Read the [README](https://github.com/PlainsightAI/filter-sam3-detector/blob/main/README.md) for usage examples
2. Check [Configuration Guide](https://github.com/PlainsightAI/filter-sam3-detector/blob/main/docs/configuration.md) for options
3. Try the example scripts in `scripts/`
4. See [Performance Tuning](https://github.com/PlainsightAI/filter-sam3-detector/blob/main/docs/performance.md) for optimization

## Support

If you encounter issues:

1. Check [Troubleshooting](https://github.com/PlainsightAI/filter-sam3-detector/blob/main/README.md#troubleshooting) section
2. Review [GitHub Issues](https://github.com/PlainsightAI/filter-sam3-detector/issues)
3. Check [Documentation](https://github.com/PlainsightAI/filter-sam3-detector/tree/main/docs)

