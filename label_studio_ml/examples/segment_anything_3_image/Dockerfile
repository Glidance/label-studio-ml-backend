# SAM3 (Segment Anything Model 3) ML Backend for Label Studio
# Uses Hugging Face Transformers for SAM3 inference

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget curl \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4

# Install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --upgrade pip && \
    pip install -r requirements-base.txt

# Install SAM3 specific requirements
# SAM3 requires transformers >= 5.0.0 (pre-release as of Jan 2026)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --pre transformers && \
    pip install -r requirements.txt

# Install test requirements if needed
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# Copy application code
COPY . ./

# Create directories for data
RUN mkdir -p /data/models

# Start script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 9090

CMD ["./start.sh"]
