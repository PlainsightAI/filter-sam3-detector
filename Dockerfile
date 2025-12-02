# Multi-stage build for smaller final image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir build uv

# Copy source
COPY . .

# Build wheel
RUN python -m build --wheel

# Runtime image with CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN python3.12 -m pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Set environment defaults
ENV FILTER_DEVICE=cuda

ENTRYPOINT ["python", "-m", "filter_sam3_detector"]
