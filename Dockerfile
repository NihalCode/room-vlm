# Inference-oriented image. Prefer a CUDA base when deploying on GPU hosts.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct \
    DEVICE=auto \
    MAX_VIDEO_FRAMES=32 \
    MAX_UPLOAD_MB=50

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY web ./web
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e .

# Weights/adapters/data are mounted at runtime — never baked into the image by default.
EXPOSE 8000 7860

CMD ["uvicorn", "room_vlm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
