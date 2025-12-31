FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the sam3 submodule first for layer caching
COPY sam3/ /app/sam3/

# Install the filter from OpenFilter registry
RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    set -eux; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://python.openfilter.io/simple \
      --extra-index-url https://pypi.org/simple \
      "filter-sam3-detector==${PKG_VERSION}"

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser
CMD ["python", "-m", "filter_sam3_detector.filter"]
