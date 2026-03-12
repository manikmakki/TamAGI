FROM python:3.12-slim

LABEL maintainer="tamagi"
LABEL description="TamAGI — Your local-first AI companion"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend ./backend
COPY frontend ./frontend


# Default port
EXPOSE 7741

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7741/login'); r.raise_for_status()" || exit 1

# Run
CMD ["python", "-m", "backend.main"]
