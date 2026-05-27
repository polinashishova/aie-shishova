FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

COPY src/ src/
COPY scripts/download_models.py scripts/

RUN pip install --no-cache-dir --prefix=/install ".[api]"


FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY artifacts/models/ artifacts/models/
COPY data/processed/ data/processed/
COPY data/features/ data/features/


RUN mkdir -p /app/artifacts/logs \
    && chown -R appuser:appuser /app/artifacts
RUN python scripts/download_models.py

USER appuser

ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    CB_FEATURE_COLUMN=genres_decade_tags \
    CB_THRESHOLD=3.0 \
    CF_THRESHOLD=4.0 \
    CONFIGS_DIR=./configs/ \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "scripts/run_api.py", "--workers", "2"]