# ---------------------------------------------------------------------------
# AutoScientist-ETFT — Coding-Agent Proxy Container
# ---------------------------------------------------------------------------
# Lightweight image that runs the FastAPI proxy server.
# Provider SDK(s) are installed based on PROXY_LLM_BACKEND at runtime;
# all three are included here so the image is backend-agnostic.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

LABEL maintainer="Lukas Benda <lukas.benda@boldpivot.cz>"
LABEL description="AutoScientist-ETFT coding-agent proxy"

# Security: run as non-root
RUN addgroup --system etft && adduser --system --ingroup etft etft

WORKDIR /app

# Install server dependencies
COPY docker/agent/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server source
COPY docker/agent/server.py /app/server.py

# Switch to non-root user
USER etft

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "server.py"]
