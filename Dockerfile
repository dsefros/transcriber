FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/runtime.txt requirements/runtime.txt
COPY requirements/ml.txt requirements/ml.txt

ARG INSTALL_ML=0
RUN pip install --upgrade pip \
    && pip install -r requirements/runtime.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements/ml.txt; fi

COPY src src
COPY models.yaml.example models.yaml.example
COPY README.md README.md

ENTRYPOINT ["python", "-m", "src.app.cli"]
