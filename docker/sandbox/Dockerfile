# ---------------------------------------------------------------------------
# AutoScientist-ETFT — Sandbox Container for CPU Edge / TinyML
# ---------------------------------------------------------------------------
# Provides a CPU-only environment with edge ML tools (e.g. tflite-micro).
# Designed to be run with tight memory constraints by `etft/sandbox.py`.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

LABEL maintainer="Lukas Benda <lukas.benda@boldpivot.cz>"
LABEL description="AutoScientist-ETFT TinyML sandbox"

# System packages for edge ML compilation (e.g. tflite, microTVM basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Edge ML python dependencies
RUN pip install --no-cache-dir \
    tflite-runtime \
    numpy \
    scipy

# Security: run as non-root
RUN addgroup --system etft && adduser --system --ingroup etft etft

WORKDIR /script

USER etft
