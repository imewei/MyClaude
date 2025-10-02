# Deployment Guide

**Version**: 1.0
**Date**: 2025-10-01
**Status**: Production-Ready MVP

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Deployment Environments](#deployment-environments)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Scaling Recommendations](#scaling-recommendations)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Scientific Computing Agents system is a production-ready MVP designed for distributed deployment across various environments. This guide covers installation, configuration, and deployment best practices.

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface / API                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Workflow Orchestration Agent                    │
│  (Sequential/Parallel Execution, Dependency Management)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐   ┌────────▼──────────┐
│  Core Agents      │   │  Support Agents    │
│  - ODE/PDE        │   │  - Profiler        │
│  - Optimization   │   │  - Parallel Exec   │
│  - Linear Algebra │   │  - Problem Analyzer│
│  - Integration    │   │  - Validator       │
└───────────────────┘   └───────────────────┘
```

---

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 4 GB
- **CPU**: 2 cores
- **Disk**: 2 GB free space
- **OS**: Linux, macOS, Windows

### Recommended Requirements (Production)

- **Python**: 3.11
- **RAM**: 16 GB
- **CPU**: 8+ cores
- **Disk**: 10 GB free space
- **OS**: Ubuntu 22.04 LTS, macOS 13+

### Optional Requirements

- **GPU**: NVIDIA GPU with CUDA 11+ (for ML agents)
- **MPI**: OpenMPI 4.0+ (for distributed computing)
- **Docker**: 24.0+ (for containerized deployment)

---

## Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Install base package
pip install scientific-computing-agents

# Install with development tools
pip install scientific-computing-agents[dev]

# Install with documentation tools
pip install scientific-computing-agents[docs]

# Install everything
pip install scientific-computing-agents[all]
```

### Method 2: Source Installation

```bash
# Clone repository
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

### Method 3: Docker Deployment

```bash
# Pull Docker image
docker pull scientific-computing-agents:latest

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  scientific-computing-agents:latest

# With GPU support
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  scientific-computing-agents:latest
```

### Method 4: Conda Installation

```bash
# Create conda environment
conda create -n sci-agents python=3.11
conda activate sci-agents

# Install from conda-forge (when available)
conda install -c conda-forge scientific-computing-agents

# Or install from source
pip install scientific-computing-agents
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Parallel execution settings
SCI_AGENTS_MAX_WORKERS=8
SCI_AGENTS_PARALLEL_MODE=threads  # threads, processes, async

# Performance settings
SCI_AGENTS_CACHE_SIZE=1000
SCI_AGENTS_TIMEOUT=300

# Logging settings
SCI_AGENTS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
SCI_AGENTS_LOG_FILE=/var/log/sci-agents.log

# GPU settings (if available)
SCI_AGENTS_USE_GPU=true
SCI_AGENTS_GPU_MEMORY_FRACTION=0.8

# Profiling (production: false)
SCI_AGENTS_ENABLE_PROFILING=false
```

### Configuration File

Create `config/agents.yaml`:

```yaml
# Workflow Orchestration
orchestration:
  parallel_mode: threads
  max_workers: 8
  timeout: 300

# Agent-specific settings
agents:
  ode_pde_solver:
    default_method: RK45
    rtol: 1e-6
    atol: 1e-9

  optimization:
    default_method: L-BFGS-B
    max_iter: 1000
    ftol: 1e-8

  linear_algebra:
    default_solver: direct
    sparse_threshold: 10000

# Performance
performance:
  cache_enabled: true
  cache_size: 1000
  profiling_enabled: false

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: /var/log/sci-agents.log
```

---

## Deployment Environments

### Development Environment

```bash
# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Enable profiling
export SCI_AGENTS_ENABLE_PROFILING=true
export SCI_AGENTS_LOG_LEVEL=DEBUG
```

### Staging Environment

```bash
# Install production dependencies only
pip install scientific-computing-agents

# Use staging configuration
export SCI_AGENTS_CONFIG=config/staging.yaml
export SCI_AGENTS_LOG_LEVEL=INFO

# Run smoke tests
pytest tests/ -m "not slow" -v
```

### Production Environment

```bash
# Install from PyPI
pip install scientific-computing-agents

# Use production configuration
export SCI_AGENTS_CONFIG=config/production.yaml
export SCI_AGENTS_LOG_LEVEL=WARNING
export SCI_AGENTS_ENABLE_PROFILING=false

# Set resource limits
export SCI_AGENTS_MAX_WORKERS=16
export SCI_AGENTS_TIMEOUT=600

# Run health checks
python -c "from agents import *; print('System OK')"
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

The system includes automated CI/CD with GitHub Actions:

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests

**Jobs:**
1. **Test** (Multi-OS, Multi-Python)
   - Ubuntu, macOS, Windows
   - Python 3.9, 3.10, 3.11, 3.12
   - Run full test suite
   - Generate coverage reports

2. **Lint**
   - flake8 code quality checks
   - black code formatting validation
   - isort import sorting validation

3. **Type Check**
   - mypy static type checking

4. **Performance**
   - Run performance benchmarks
   - Store results as artifacts

### Coverage Reporting

Coverage reports are automatically uploaded to Codecov:

- **Target**: >80% coverage
- **Current**: ~78-80%
- **Report**: Available at codecov.io dashboard

### Manual Deployment

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI (production)
twine upload dist/*

# Upload to TestPyPI (staging)
twine upload --repository testpypi dist/*
```

---

## Monitoring and Logging

### Logging Configuration

```python
import logging
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sci-agents.log'),
        logging.StreamHandler()
    ]
)

# Create orchestrator with logging
orchestrator = WorkflowOrchestrationAgent()
```

### Performance Monitoring

```python
from agents.performance_profiler_agent import PerformanceProfilerAgent

profiler = PerformanceProfilerAgent()

# Profile workflow execution
result = profiler.process({
    'task': 'profile_function',
    'function': my_workflow,
    'args': [data]
})

print(result.data['report'])
```

### Health Checks

Create `health_check.py`:

```python
#!/usr/bin/env python3
"""Health check script for monitoring."""

import sys
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent

def health_check():
    """Verify system is operational."""
    try:
        # Test orchestrator creation
        orchestrator = WorkflowOrchestrationAgent()

        # Test basic workflow
        result = orchestrator.execute_workflow([])

        if result.success:
            print("OK: System operational")
            return 0
        else:
            print("ERROR: Workflow execution failed")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())
```

Run periodically:
```bash
# Add to crontab
*/5 * * * * /path/to/health_check.py >> /var/log/health.log 2>&1
```

---

## Security Considerations

### Input Validation

- All agent inputs are validated
- Type checking enforced via dataclasses
- Sanitize user-provided code before execution

### Dependency Security

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Access Control

- Restrict file system access in production
- Use environment variables for sensitive config
- Never commit credentials to version control

### Network Security

- Use HTTPS for all external communication
- Validate SSL certificates
- Implement rate limiting for API endpoints

---

## Scaling Recommendations

### Vertical Scaling

**Small Scale** (1-10 users):
- 8 GB RAM, 4 CPU cores
- `max_workers=4`
- Local SQLite database

**Medium Scale** (10-100 users):
- 32 GB RAM, 16 CPU cores
- `max_workers=12`
- PostgreSQL database
- Redis cache

**Large Scale** (100+ users):
- 64+ GB RAM, 32+ CPU cores
- `max_workers=24`
- Distributed PostgreSQL
- Redis cluster
- Load balancer

### Horizontal Scaling

**Distributed Deployment**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: sci-agents:latest
    replicas: 3
    environment:
      - SCI_AGENTS_MAX_WORKERS=8
      - SCI_AGENTS_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=sci_agents
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
```

Deploy with:
```bash
docker-compose up -d --scale api=3
```

### Performance Optimization

1. **Enable caching**: Reduce redundant computations
2. **Use parallel execution**: `parallel_mode=threads` for I/O, `processes` for CPU
3. **Profile bottlenecks**: Use `PerformanceProfilerAgent`
4. **Optimize data transfer**: Use shared memory for large arrays
5. **GPU acceleration**: Enable for ML-heavy workloads

---

## Troubleshooting

### Common Issues

**Issue**: Tests failing with profiler conflicts
```bash
# Solution: Run tests with isolation
pytest tests/ -v --forked
```

**Issue**: Out of memory errors
```bash
# Solution: Reduce max_workers
export SCI_AGENTS_MAX_WORKERS=4
```

**Issue**: Slow performance
```bash
# Solution: Profile and optimize
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats
```

**Issue**: Import errors
```bash
# Solution: Verify installation
pip list | grep scientific-computing-agents
pip install --force-reinstall scientific-computing-agents
```

### Debug Mode

Enable detailed logging:
```bash
export SCI_AGENTS_LOG_LEVEL=DEBUG
export SCI_AGENTS_ENABLE_PROFILING=true

python your_script.py 2>&1 | tee debug.log
```

### Support

- **GitHub Issues**: https://github.com/scientific-computing-agents/scientific-computing-agents/issues
- **Documentation**: https://scientific-computing-agents.readthedocs.io
- **Discussions**: https://github.com/scientific-computing-agents/scientific-computing-agents/discussions

---

## Version History

- **v0.1.0** (2025-10-01): Initial MVP release
  - 14 agents operational
  - 379 tests (97.6% pass rate)
  - 78-80% code coverage
  - Production-ready infrastructure

---

**Document Version**: 1.0
**Last Updated**: 2025-10-01
**Maintainers**: Scientific Computing Agents Team
