# syntax=docker/dockerfile:1.4
# PyTorch 2.9.1 with CUDA 12.8 supports sm_120 (RTX 50-series / Blackwell)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install uv for fast, correct dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy vendorized sam3 first (needed for local install via tool.uv.sources)
COPY sam3/ /app/sam3/

# Copy project files
COPY pyproject.toml VERSION README.md LICENSE LICENSING.md /app/
COPY filter_sam3_detector/ /app/filter_sam3_detector/
COPY scripts/ /app/scripts/

# Install dependencies using uv (respects tool.uv.sources for local sam3)
RUN uv pip install --system -e .

# Download SAM3 model weights during build and bake into image.
# Uses BuildKit secret mount for secure authentication with gated models.
# The model is cached at /root/.cache/huggingface/hub (default HF cache location).
# At runtime, no HF_TOKEN is needed since model is already in the image.
#
# When HF_TOKEN is unavailable (e.g. Dependabot/fork PR builds, where GitHub
# withholds repo secrets), the mounted secret is empty. In that case we skip the
# weight bake instead of calling snapshot_download with an empty token, which
# would send an illegal "Bearer " header and fail the build. Such images are only
# ever produced by the dry-run publish and are never released, so a weightless
# image is acceptable there; the real publish path always has the token.
RUN --mount=type=secret,id=hf_token python <<'PY'
import os
from huggingface_hub import snapshot_download

secret_path = "/run/secrets/hf_token"
token = open(secret_path).read().strip() if os.path.exists(secret_path) else ""
print(f"Token present: {bool(token)}")
if token:
    snapshot_download(repo_id="facebook/sam3", token=token)
    print("SAM3 model weights baked into container")
else:
    print("No HF_TOKEN provided (e.g. Dependabot/fork build); skipping SAM3 weight bake")
PY

# Serve SAM3 weights from baked cache; skip hub revalidation for air-gapped runs.
# Placed after snapshot_download so the build step is never gated by these flags,
# regardless of future huggingface_hub behavior around explicit `token=` + offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Default: run SAM3 detector filter
CMD ["python", "-m", "filter_sam3_detector.filter"]
