# Multi-stage Docker build for optimal control
# Production-ready container with GPU support

# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-gpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Install GPU dependencies
# Uncomment for GPU support
# RUN pip install --no-cache-dir -r requirements-gpu.txt

# ============================================================
# Stage 2: Runtime - Optimized production image
# ============================================================
FROM python:3.10-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser solvers ./solvers
COPY --chown=appuser:appuser ml_optimal_control ./ml_optimal_control
COPY --chown=appuser:appuser applications ./applications
COPY --chown=appuser:appuser hpc ./hpc
COPY --chown=appuser:appuser visualization ./visualization
COPY --chown=appuser:appuser gpu_kernels ./gpu_kernels
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser deployment ./deployment
COPY --chown=appuser:appuser cloud ./cloud

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER appuser

# Default command (override for specific services)
CMD ["python", "-m", "api.rest_api"]
