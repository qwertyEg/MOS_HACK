# СтройВзор — образ веб-сервиса.
#   docker build -t stroyvzor .                  # с локальными моделями (torch CPU, ultralytics, transformers)
#   docker build -t stroyvzor --build-arg WITH_ML=0 .   # лёгкий: только внешний API
# Веса моделей в образ не кладём — монтируются томом в /app/models.
FROM python:3.12-slim

ARG WITH_ML=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf

# libgl/libglib — для opencv (ultralytics тянет не-headless сборку), curl — для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt
# torch — строго с CPU-индекса: иначе pip на linux/amd64 скачает CUDA-колёса на гигабайты.
RUN if [ "$WITH_ML" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3" "torchvision>=0.19" \
        && pip install -r requirements-ml.txt ; \
    fi

COPY core/ ./core/
COPY app/ ./app/
COPY reference/ ./reference/
COPY simcam/ ./simcam/
COPY tools/ ./tools/

RUN mkdir -p /app/var /app/models /app/datasets
VOLUME ["/app/var", "/app/models"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# Один процесс намеренно: очередь кадров и загруженные модели живут в памяти.
CMD ["python", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
