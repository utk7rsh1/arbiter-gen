# ── ARBITER — HuggingFace Spaces Docker deployment ────────────────────────────
#
# What this container does:
#   • Runs FastAPI (arbiter/server.py) on port 7860
#   • Serves the React frontend (arbiter/demo/frontend/) at GET /
#   • Exposes the full REST API at /sessions, /metrics, /health, /docs
#
# What this container does NOT do:
#   • Run LLM inference (no GPU, no torch) — training happens on Colab
#   • Serve the Gradio demo (app.py) — that's a separate entrypoint
#
# Image size target: < 250 MB
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="ARBITER Team" \
      description="ARBITER AI Oversight Research Environment — HuggingFace Spaces" \
      version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
# gcc/g++ needed to compile any C extensions (e.g. httptools from uvicorn[standard])
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps: install before copying source for layer-cache efficiency ────
# Copy requirements first so Docker reuses this layer on code-only changes
COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-server.txt

# ── Copy application source ───────────────────────────────────────────────────
# .dockerignore excludes: lora checkpoints, training data, test scripts,
# __pycache__, and other large/dev-only files
COPY . .

# ── Permissions ──────────────────────────────────────────────────────────────
# HuggingFace Spaces executes containers as a non-root user (uid 1000).
# Create matching user and transfer ownership of the working directory.
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Disable tokenizer parallelism warning (safe since we have no tokenizers here)
    TOKENIZERS_PARALLELISM=false

# ── Health check ──────────────────────────────────────────────────────────────
# HF Spaces polls /health to decide when the container is ready.
# Interval 30s, timeout 10s, 3 retries before marking unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Port ──────────────────────────────────────────────────────────────────────
# HuggingFace Spaces REQUIRES port 7860. Do not change this.
EXPOSE 7860

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --workers 1   : single worker keeps in-memory session state consistent.
#                 (Multiple workers would each have a separate _sessions dict,
#                  causing 404s when the next request hits a different worker.)
# --timeout-keep-alive 75 : standard for HF Spaces (proxy timeout is 60s)
# --log-level info        : enough detail for debugging without flooding logs
CMD ["uvicorn", "arbiter.server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "75"]
