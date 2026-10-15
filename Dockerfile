# Multi-platform production Dockerfile for Customer Retention Intelligence Platform (CRIP) API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install system dependencies required for LightGBM (libgomp1) and healthchecks (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application and model files
COPY api/ api/
COPY src/ src/

# Create a non-privileged user and adjust permissions
RUN useradd -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
