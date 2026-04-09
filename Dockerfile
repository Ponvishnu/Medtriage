# ─────────────────────────────────────────────────────────────
# MedTriageEnv — Dockerfile
# Deploys on Hugging Face Spaces (port 7860) or any Docker host.
#
# Build:  docker build -t medtriage-env .
# Run:    docker run -p 7860:7860 medtriage-env
# Test:   docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... medtriage-env
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces requirement) ────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Create package init files ─────────────────────────────────
RUN touch baseline/__init__.py tests/__init__.py 2>/dev/null || true

# ── Switch to non-root user ───────────────────────────────────
USER appuser

# ── Environment ───────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Startup ───────────────────────────────────────────────────
CMD ["python", "-m", "server.app"]
