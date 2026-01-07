FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ARG CATALOGUE_PATH=catalogue.csv
ARG QUERIES_PATH=queries.csv
ARG ARTIFACTS_DIR=artifacts
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG INDEX_VERSION=1.0.0

ENV HF_HOME=/build/cache/huggingface
ENV ARTIFACTS_DIR=/build/${ARTIFACTS_DIR}
ENV INDEX_VERSION=${INDEX_VERSION}
ENV MODEL_NAME=${MODEL_NAME}
ENV PYTHONPATH=/build/src

COPY ${CATALOGUE_PATH} ./catalogue.csv
COPY ${QUERIES_PATH} ./queries.csv

RUN mkdir -p ${ARTIFACTS_DIR} ${HF_HOME}

RUN python scripts/build_index.py \
    --catalogue ./catalogue.csv \
    --output ${ARTIFACTS_DIR} \
    --index-version ${INDEX_VERSION} \
    --model ${MODEL_NAME}

RUN python scripts/fit_calibration.py \
    --queries ./queries.csv \
    --artifacts ${ARTIFACTS_DIR} \
    --index-version ${INDEX_VERSION}

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /build/artifacts ./artifacts

ENV HF_HOME=/root/.cache/huggingface
COPY --from=builder /build/cache/huggingface /root/.cache/huggingface

COPY src/ ./src/
COPY api/ ./api/

ENV ARTIFACTS_DIR=/app/artifacts
ENV INDEX_VERSION=1.0.0
ENV MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]