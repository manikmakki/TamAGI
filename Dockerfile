FROM python:3.13-slim

LABEL maintainer="tamagi"
LABEL description="TamAGI — Your local-first AI companion"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy application and package metadata for optional legacy retrieval backends.
COPY pyproject.toml README.md LICENSE.md ./
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

ARG TAMAGI_MEMORY_EXTRA=chroma
RUN case "$TAMAGI_MEMORY_EXTRA" in \
    none) ;; \
    chroma|elasticsearch) pip install --no-cache-dir -c requirements.lock ".[${TAMAGI_MEMORY_EXTRA}]" ;; \
    *) echo "Choose TAMAGI_MEMORY_EXTRA=none, chroma, or elasticsearch" >&2; exit 1 ;; \
    esac


# Default port
EXPOSE 7741

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7741/login'); r.raise_for_status()" || exit 1

# Run
CMD ["python", "-m", "backend.main"]
