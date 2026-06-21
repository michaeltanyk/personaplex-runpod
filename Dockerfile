# PersonaPlex RunPod Serverless Worker
# GPU-accelerated, pay-per-second, full-duplex voice AI

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache

# System deps in one layer with cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-dev python3-venv python3-pip \
        libopus-dev libsndfile-dev ffmpeg \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app

# Venv (python3 = 3.10 on Ubuntu 22.04)
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# PyTorch + cleanup
RUN pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && pip cache purge && rm -rf /root/.cache/pip

# PersonaPlex — clone, install moshi, remove repo
RUN git clone https://github.com/NVIDIA/personaplex.git /tmp/personaplex \
    && cd /tmp/personaplex && pip install --no-cache-dir ./moshi/ \
    && rm -rf /tmp/personaplex \
    && pip cache purge && rm -rf /root/.cache/pip

# RunPod + HF
RUN pip install --no-cache-dir huggingface_hub runpod \
    && pip cache purge && rm -rf /root/.cache/pip

# Handler
COPY rp_handler.py /app/
COPY download_voices.py /app/
RUN python3 /app/download_voices.py || echo "[build] Voice download deferred"

CMD ["python3", "-u", "/app/rp_handler.py"]
