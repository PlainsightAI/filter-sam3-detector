# syntax=docker/dockerfile:1.4
# PyTorch 2.9.1 with CUDA 12.8 supports sm_120 (RTX 50-series / Blackwell)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy vendorized sam3 first (needed for local install)
COPY sam3/ /app/sam3/

# Copy project files
COPY pyproject.toml VERSION README.md /app/
COPY filter_sam3_detector/ /app/filter_sam3_detector/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install \
    --index-url https://python.openfilter.io/simple \
    --extra-index-url https://pypi.org/simple \
    -e /app/sam3 && \
    pip install \
    --index-url https://python.openfilter.io/simple \
    --extra-index-url https://pypi.org/simple \
    -e .

# Download SAM3 model weights during build and bake into image.
# Uses HF_TOKEN secret for authentication with gated models.
# The model is cached at /root/.cache/huggingface/hub (default HF cache location).
# At runtime, no HF_TOKEN is needed since model is already in the image.
RUN --mount=type=secret,id=hf_token \
    if [ -f /run/secrets/hf_token ]; then \
      export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='facebook/sam3'); \
print('SAM3 model weights baked into container')"

# Default: run SAM3 detector filter
CMD ["python", "-m", "filter_sam3_detector.filter"]
