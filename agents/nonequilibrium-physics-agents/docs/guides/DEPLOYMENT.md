# Deployment Guide

**Nonequilibrium Physics Optimal Control Framework**
**Version**: 1.0.0
**Date**: 2025-10-01

Complete deployment guide for local, HPC, Docker, and cloud environments.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Local Installation](#local-installation)
- [HPC Cluster Deployment](#hpc-cluster-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Installation

```bash
# Install core package
pip install nonequilibrium-control

# Verify installation
python -c "import nonequilibrium_control; print('✓ Installation successful')"
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-org/nonequilibrium-control.git
cd nonequilibrium-control

# Install in development mode with all dependencies
pip install -e ".[all]"

# Run tests to verify
pytest tests/ -v
```

---

## Local Installation

### Prerequisites

**Required**:
- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.11

**Optional** (for enhanced features):
- JAX >= 0.4.0 (GPU acceleration)
- Dask >= 2023.1.0 (distributed computing)
- scikit-optimize >= 0.9.0 (Bayesian optimization)

### Installation Methods

#### Method 1: pip (Recommended)

```bash
# Basic installation
pip install nonequilibrium-control

# With GPU support
pip install "nonequilibrium-control[gpu]"

# With distributed computing
pip install "nonequilibrium-control[distributed]"

# With all optional dependencies
pip install "nonequilibrium-control[all]"
```

#### Method 2: conda

```bash
# Create environment
conda create -n neqcontrol python=3.10
conda activate neqcontrol

# Install from conda-forge
conda install -c conda-forge nonequilibrium-control

# Optional: GPU support
conda install -c conda-forge jax jaxlib
```

#### Method 3: From Source

```bash
# Clone repository
git clone https://github.com/your-org/nonequilibrium-control.git
cd nonequilibrium-control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run tests
pytest tests/
```

### Verifying Installation

```python
# Test basic functionality
import nonequilibrium_control as nec

# Check version
print(f"Version: {nec.__version__}")

# Test GPU availability (if installed)
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"GPU available: {jax.devices('gpu')}")
except ImportError:
    print("JAX not installed (optional)")

# Test distributed computing (if installed)
try:
    import dask
    print(f"Dask version: {dask.__version__}")
except ImportError:
    print("Dask not installed (optional)")
```

---

## HPC Cluster Deployment

### SLURM Cluster Setup

#### 1. Module Loading

Create module file `/path/to/modules/nonequilibrium-control/1.0.0`:

```tcl
#%Module1.0

proc ModulesHelp { } {
    puts stderr "Nonequilibrium Physics Optimal Control Framework v1.0.0"
}

module-whatis "Optimal control and ML for nonequilibrium physics"

# Prerequisites
prereq python/3.10
prereq cuda/11.8

# Set paths
set root /path/to/nonequilibrium-control
prepend-path PYTHONPATH $root
prepend-path PATH $root/bin
```

#### 2. User Installation

```bash
# On login node
module load python/3.10 cuda/11.8

# Install in user directory
pip install --user nonequilibrium-control[hpc]

# Verify
python -c "import nonequilibrium_control; print(nonequilibrium_control.__version__)"
```

#### 3. Job Script Template

Create `job_template.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=neq_control
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load modules
module load python/3.10 cuda/11.8

# Activate environment
source ~/venv/neqcontrol/bin/activate

# Set OpenMP threads
export OMP_NUM_THREADS=8

# Run computation
python -m nonequilibrium_control.hpc.sweep \
    --config config.yaml \
    --scheduler slurm \
    --workers 128
```

#### 4. Parameter Sweep Execution

```bash
# Submit parameter sweep
sbatch job_template.sh

# Monitor progress
squeue -u $USER

# Check results
tail -f logs/*.out
```

### PBS Cluster Setup

#### Job Script Template

```bash
#!/bin/bash
#PBS -N neq_control
#PBS -l nodes=4:ppn=32:gpus=4
#PBS -l walltime=12:00:00
#PBS -q gpu
#PBS -o logs/$PBS_JOBID.out
#PBS -e logs/$PBS_JOBID.err

# Change to working directory
cd $PBS_O_WORKDIR

# Load modules
module load python/3.10 cuda/11.8

# Activate environment
source ~/venv/neqcontrol/bin/activate

# Run computation
python -m nonequilibrium_control.hpc.sweep \
    --config config.yaml \
    --scheduler pbs \
    --workers 128
```

### Configuration File Example

Create `config.yaml`:

```yaml
# Parameter sweep configuration
problem:
  type: "quantum_control"
  hilbert_dim: 100
  time_horizon: 10.0

parameters:
  learning_rate:
    type: continuous
    lower: 0.001
    upper: 0.1
    log_scale: true

  hidden_size:
    type: integer
    lower: 32
    upper: 256

sweep:
  strategy: "adaptive"
  n_initial: 50
  n_iterations: 10
  exploration_weight: 0.1

distributed:
  backend: "dask"
  workers_per_node: 32
  threads_per_worker: 2

output:
  directory: "results/"
  format: "json"
  save_intermediate: true
```

---

## Docker Deployment

### Building Docker Image

#### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Set entrypoint
ENTRYPOINT ["python", "-m", "nonequilibrium_control"]
CMD ["--help"]
```

#### Build and Run

```bash
# Build image
docker build -t nonequilibrium-control:1.0.0 .

# Tag as latest
docker tag nonequilibrium-control:1.0.0 nonequilibrium-control:latest

# Run interactive
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    nonequilibrium-control:latest \
    python examples/neural_control.py

# Run parameter sweep
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config.yaml:/app/config.yaml \
    nonequilibrium-control:latest \
    python -m nonequilibrium_control.hpc.sweep --config config.yaml
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  control:
    build: .
    image: nonequilibrium-control:latest
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
    command: python -m nonequilibrium_control.hpc.sweep --config /app/config/sweep.yaml

  dask-scheduler:
    image: daskdev/dask:latest
    command: dask-scheduler
    ports:
      - "8786:8786"
      - "8787:8787"

  dask-worker:
    image: daskdev/dask:latest
    command: dask-worker tcp://dask-scheduler:8786
    depends_on:
      - dask-scheduler
    deploy:
      replicas: 4
```

### Running with Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f control

# Scale workers
docker-compose up -d --scale dask-worker=8

# Stop services
docker-compose down
```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance with GPU
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type p3.8xlarge \
    --key-name my-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --count 1

# Connect to instance
ssh -i my-key-pair.pem ubuntu@<instance-ip>

# Install CUDA and drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit

# Install framework
pip install nonequilibrium-control[gpu]
```

#### AWS Batch Configuration

Create `batch-job-definition.json`:

```json
{
  "jobDefinitionName": "neq-control-job",
  "type": "container",
  "containerProperties": {
    "image": "nonequilibrium-control:latest",
    "vcpus": 4,
    "memory": 16384,
    "command": [
      "python", "-m", "nonequilibrium_control.hpc.sweep",
      "--config", "/mnt/config/sweep.yaml"
    ],
    "mountPoints": [
      {
        "sourceVolume": "config",
        "containerPath": "/mnt/config",
        "readOnly": true
      }
    ],
    "resourceRequirements": [
      {
        "type": "GPU",
        "value": "1"
      }
    ]
  }
}
```

### Google Cloud Platform

#### GKE Cluster Setup

```bash
# Create GKE cluster
gcloud container clusters create neq-control-cluster \
    --num-nodes=4 \
    --machine-type=n1-standard-4 \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials neq-control-cluster

# Deploy application
kubectl apply -f k8s/deployment.yaml
```

#### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neq-control
spec:
  replicas: 4
  selector:
    matchLabels:
      app: neq-control
  template:
    metadata:
      labels:
        app: neq-control
    spec:
      containers:
      - name: control
        image: gcr.io/project-id/nonequilibrium-control:1.0.0
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: neq-control-config
      - name: data
        persistentVolumeClaim:
          claimName: neq-control-data
```

### Azure Deployment

#### Azure ML Workspace

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig

# Connect to workspace
ws = Workspace.from_config()

# Create experiment
experiment = Experiment(ws, 'neq-control-sweep')

# Configure run
config = ScriptRunConfig(
    source_directory='.',
    script='run_sweep.py',
    compute_target='gpu-cluster',
    environment=ws.environments['nonequilibrium-control-env']
)

# Submit
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
```

---

## Configuration

### Environment Variables

```bash
# Set Python path
export PYTHONPATH=/path/to/nonequilibrium-control:$PYTHONPATH

# Configure number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# JAX configuration
export JAX_PLATFORM_NAME=gpu  # or cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Dask configuration
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.8
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.9
```

### Configuration File

Create `~/.neqcontrol/config.yaml`:

```yaml
# Default configuration
compute:
  backend: "auto"  # auto, cpu, gpu
  threads: 8
  memory_limit: "16GB"

gpu:
  enabled: true
  devices: [0]  # GPU device IDs
  memory_fraction: 0.9

distributed:
  scheduler: "auto"  # auto, local, slurm, pbs
  workers: 4
  threads_per_worker: 2

output:
  directory: "results/"
  format: "json"  # json, hdf5, npz
  compression: true
  log_level: "INFO"
```

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Problem: ImportError: No module named 'nonequilibrium_control'
# Solution:
pip install nonequilibrium-control
# OR
export PYTHONPATH=/path/to/package:$PYTHONPATH
```

#### GPU Not Detected

```bash
# Problem: JAX not using GPU
# Solution 1: Install JAX with CUDA
pip install --upgrade jax[cuda11_cudnn82]

# Solution 2: Check CUDA installation
nvidia-smi
nvcc --version

# Solution 3: Set JAX platform
export JAX_PLATFORM_NAME=gpu
```

#### Out of Memory

```bash
# Problem: CUDA out of memory
# Solution 1: Reduce batch size
# Edit config.yaml: batch_size: 32  # smaller

# Solution 2: Enable memory growth
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Solution 3: Limit GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
```

#### SLURM Job Fails

```bash
# Check job status
squeue -u $USER

# View error log
cat logs/$JOBID.err

# Common fixes:
# 1. Check module loading
# 2. Verify paths in job script
# 3. Check resource limits
# 4. Test interactively first
```

### Performance Issues

#### Slow Execution

```python
# Profile code
from ml_optimal_control.performance import FunctionProfiler

profiler = FunctionProfiler()
result = profiler.profile(your_function)
print(profiler.get_report())
```

#### Poor Parallel Scaling

```bash
# Check worker utilization
# View Dask dashboard at http://localhost:8787

# Increase task granularity
# Edit config: increase problem size per task

# Reduce communication overhead
# Use larger batch sizes
```

### Getting Help

- **Documentation**: https://docs.nonequilibrium-control.org
- **Issues**: https://github.com/your-org/nonequilibrium-control/issues
- **Discussions**: https://github.com/your-org/nonequilibrium-control/discussions
- **Email**: support@nonequilibrium-control.org

---

## Next Steps

After deployment:
1. **Run Examples**: See `examples/` directory
2. **Read Tutorials**: Check `docs/tutorials/`
3. **Review API**: See `docs/api/`
4. **Join Community**: GitHub Discussions

---

**Version**: 1.0.0
**Last Updated**: 2025-10-01
**Maintainer**: Nonequilibrium Physics Agents Team
