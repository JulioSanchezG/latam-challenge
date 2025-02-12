# syntax=docker/dockerfile:1.2
ARG REPO_NAME
ARG MODEL_VERSION

FROM python:3.11-slim

ENV MODEL_VERSION=1.0

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel

WORKDIR /model-api-app

RUN adduser --system api-user
RUN chown -R api-user /model-api-app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER api-user

COPY models/v${MODEL_VERSION}/ models/v${MODEL_VERSION}/
COPY challenge/ challenge/

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
