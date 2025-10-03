# Installation Guide

## Prerequisites

- Python 3.10 or higher
- conda or virtualenv
- Basic Python knowledge
- Git

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd materials-science-agents
```

### Step 2: Create Environment

#### Using conda (recommended)

```bash
conda create -n materials-agents python=3.10
conda activate materials-agents
```

#### Using venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run a quick test
python3 -m pytest tests/test_light_scattering_agent.py::TestLightScatteringBasics::test_agent_creation -v
```

Expected output: `1 passed`

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the project directory
cd materials-science-agents

# Verify Python path
python3 -c "import sys; print(sys.path)"

# Try direct import
python3 -c "from light_scattering_agent import LightScatteringAgent; print('Success!')"
```

### Test Failures

```bash
# Check pytest version
pytest --version  # Should be 7.0+

# Run with verbose output
pytest tests/ -vv

# Run single test for debugging
pytest tests/test_light_scattering_agent.py::TestLightScatteringBasics::test_agent_creation -vv
```

### Performance Issues

```python
# Enable resource estimation
agent = LightScatteringAgent()
resources = agent.estimate_resources({'technique': 'DLS'})
print(f"Estimated time: {resources.estimated_time_sec}s")
print(f"Memory required: {resources.memory_gb}GB")
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Run your first analysis in 5 minutes
- [First Analysis Guide](first-analysis.md) - Learn the basics step by step
- [User Guide](../guides/user-guide.md) - Comprehensive documentation
