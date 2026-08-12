# syntax=docker/dockerfile:1.4
# Runtime on openfilter-base (python:3.11-slim + weekly apt-upgrade) instead of
# pytorch/pytorch:*-cuda*-runtime, which was never apt-upgraded and carried OS-package CVEs.
# torch is pinned to >=2.9,<2.10 in pyproject.toml; that wheel bundles CUDA 12.8 (cu128) —
# verified: torch 2.9.1 Requires nvidia-cuda-runtime-cu12==12.8.90, which is what Blackwell
# (sm_120) needs. The pin is deliberate: an unpinned torch now resolves to 2.13.x, whose wheel
# bundles CUDA 13 (cu13) and drops the cu12 runtime — a silent CUDA-stack change. torch 2.9.1
# +cu128 is validated on Blackwell (RTX 5060, sm_120) via the lab GPU smoke, so treat any torch
# bump as a deliberate, re-tested change.
FROM plainsightai/openfilter-base:py3.11

# Install uv for fast, correct dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# PYTHONDONTWRITEBYTECODE / PYTHONUNBUFFERED are provided by openfilter-base.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# STEP 1: Install system packages immediately. 
# These rarely change, so they should be cached at the very top.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# STEP 2: Pre-install huggingface_hub globally so we can download assets early.
RUN uv pip install --system huggingface_hub


# STEP 3: Download heavy assets (weights and kernels) BEFORE copying your project code.
RUN --mount=type=secret,id=hf_token python <<'PY'
import os
from huggingface_hub import snapshot_download

# Read Hugging Face token if present.
# Note: when the hf_token secret is absent (e.g. Dependabot/fork PRs), the mount
# still creates an empty file. Coerce "" to None so anonymous downloads work and
# huggingface_hub does not emit an invalid "Bearer " (empty) auth header.
secret_path = "/run/secrets/hf_token"
token = (open(secret_path).read().strip() or None) if os.path.exists(secret_path) else None
print(f"Token present: {bool(token)}")

# 1. Download the custom cv-utils GPU kernel (Version 1 is stored on revision 'v1')
print("Caching cv-utils GPU kernel...")
try:
    snapshot_download(
        repo_id="kernels-community/cv-utils",
        repo_type="kernel",
        revision="v1",
        token=token
    )
    print("✓ cv-utils kernel cached successfully")
except Exception as e:
    print(f"✗ Failed to cache cv-utils kernel: {e}")
    raise e

# 2. Download the SAM3 model weights
if token:
    print("Caching SAM3 model weights...")
    snapshot_download(
        repo_id="facebook/sam3", 
        token=token
    )
    print("✓ SAM3 model weights baked into container")
else:
    print("No HF_TOKEN provided; skipping SAM3 weight bake")
PY
# STEP 4: Copy the vendorized dependency (changes rarely)
COPY sam3/ /app/sam3/

# STEP 5: Copy ONLY your dependency definitions
COPY pyproject.toml VERSION README.md LICENSE LICENSING.md /app/

# STEP 6: Dummy Skeleton Trick
# Because 'uv pip install -e .' expects your project folder to exist, we create a 
# placeholder directory. This allows uv to resolve and download all heavy external 
# PyPI packages and cache them. 
# We also use a BuildKit cache mount (--mount=type=cache) so even if you add a new 
# package to pyproject.toml, uv won't re-download the unchanged ones!

# Create the placeholder skeleton directory first
RUN mkdir -p /app/filter_sam3_detector && touch /app/filter_sam3_detector/__init__.py

# Run the installer with the cache mount correctly attached to the RUN command
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -e .

# STEP 7: Copy your actual changing source code at the ABSOLUTE BOTTOM.
# Now, editing your code will build the container in less than 1 second.
COPY filter_sam3_detector/ /app/filter_sam3_detector/
COPY scripts/ /app/scripts/

# Finalize the editable install tracking to point to the real files (instantaneous)
RUN uv pip install --system --break-system-packages -e .

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-m", "filter_sam3_detector.filter"]