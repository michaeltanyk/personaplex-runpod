# PersonaPlex RunPod Serverless Worker
# ======================================
# Runs NVIDIA PersonaPlex as a persistent WebSocket server on GPU.
# Model loads once on worker start, stays warm for all requests.
# Pay per second of GPU time, scales to zero when idle.

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    libopus-dev libsndfile1 ffmpeg \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone PersonaPlex and install
RUN git clone https://github.com/NVIDIA/personaplex.git /app/personaplex \
    && pip install --no-cache-dir /app/personaplex/moshi/ \
    && pip install --no-cache-dir huggingface_hub

# RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY rp_handler.py /app/

# Pre-download voice files (skip model weights — too big for Docker layer)
COPY download_voices.py /app/
RUN python3 /app/download_voices.py || echo "[build] Voice download deferred to runtime"

# RunPod entrypoint — starts PersonaPlex server then enters worker loop
CMD ["python3", "-u", "/app/rp_handler.py"]
