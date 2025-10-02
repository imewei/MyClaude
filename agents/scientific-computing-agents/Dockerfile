# Multi-stage Dockerfile for Scientific Computing Agents

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder with full dependencies
FROM base as builder

WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Stage 3: Production image
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY agents/ ./agents/
COPY core/ ./core/
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Create non-root user
RUN useradd -m -u 1000 sciuser && \
    chown -R sciuser:sciuser /app

USER sciuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent; WorkflowOrchestrationAgent()" || exit 1

# Default command
CMD ["python"]

# Stage 4: Development image
FROM production as development

USER root

# Install development dependencies
COPY --from=builder /build/requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER sciuser

# Jupyter notebook port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 5: GPU-enabled image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy application
COPY agents/ ./agents/
COPY core/ ./core/
COPY examples/ ./examples/

# Create user
RUN useradd -m -u 1000 sciuser && \
    chown -R sciuser:sciuser /app

USER sciuser

CMD ["python"]
