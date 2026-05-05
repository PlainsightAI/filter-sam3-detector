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

SAM3 weights are fetched **at image build time** and baked into the Docker image; runtime is air-gap-safe (`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`), so failures here are almost always build-time auth issues, not runtime network issues.

**Solutions**:
1. Ensure your HuggingFace token is passed to the build via the `hf_token` BuildKit secret (not `HF_TOKEN` at runtime): `docker build --secret id=hf_token,env=HF_TOKEN ...`
2. Confirm your account has accepted the gated-model license for `facebook/sam3` on huggingface.co
3. If your build host can't reach huggingface.co, build on a networked host and distribute the resulting image; the running container does not require network access

### Airgap: HEAD Requests to huggingface.co at Startup

**Problem**: Container starts, but logs show outbound HEAD requests to `https://huggingface.co/facebook/sam3/resolve/main/{config.json,sam3.pt}` (often `401 Unauthorized` since no `HF_TOKEN` is in scope at runtime). The model still loads from the baked cache, but the network round-trip defeats true airgap and breaks deployments where outbound DNS/TLS to huggingface.co is blocked.

**Cause**: `huggingface_hub` parses `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` with a **strict** truthy check — only the case-insensitive values `1`, `ON`, `YES`, `TRUE` count as enabled. Anything else (including the empty string, `"1"` with literal quote characters, or `true!`) silently disables offline mode.

The most common way to land malformed values is via Docker Compose default-value syntax. **Compose substitutes the default literally, including any surrounding quotes:**

```yaml
# WRONG — substitutes the three literal characters [", 1, "],
# which fails huggingface_hub's check and silently turns offline mode OFF.
environment:
  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-"1"}
  TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-"1"}

# CORRECT — substitutes the single character 1.
environment:
  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-1}
  TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-1}
```

You can verify what your compose file actually passes with `docker compose config` (look at the resolved `environment:` block) or by exec'ing into a running container and printing the env: `docker exec <container> env | grep HF_HUB_OFFLINE` — if it shows `HF_HUB_OFFLINE="1"`, the quotes are part of the value.

**Solutions**:
1. Drop the quotes around the default in your compose file: `${VAR:-1}` not `${VAR:-"1"}`.
2. Or omit the override entirely — the published image already sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in the Dockerfile, so leaving them out of the compose `environment:` block is the simplest correct configuration.

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

