# Quick Start Guide

**Time to first calculation**: 5 minutes

---

## 1. Installation (2 minutes)

```bash
# Navigate to project
cd /Users/b80985/.claude/agents/nonequilibrium-physics-agents

# Create conda environment
conda create -n neph-agents python=3.10
conda activate neph-agents

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Verify Installation (1 minute)

```bash
# Test imports
python -c "from base_agent import BaseAgent; print('✅ Base agent working')"
python -c "from transport_agent import TransportAgent; print('✅ Transport agent working')"

# Run a single test
pytest tests/test_transport_agent.py::test_transport_initialization -v
```

---

## 3. First Calculation (2 minutes)

### Example 1: Calculate Thermal Conductivity

```python
from transport_agent import TransportAgent
import numpy as np

# Create agent
agent = TransportAgent(config={'backend': 'local'})

# Generate mock heat flux data (or use your own trajectory)
time = np.linspace(0, 10, 1000)  # ps
heat_flux = np.random.randn(1000, 3) * 5  # eV/ps/Å²

# Calculate thermal conductivity
result = agent.execute({
    'method': 'thermal_conductivity',
    'data': {'heat_flux': heat_flux, 'time': time},
    'parameters': {
        'temperature': 300,  # K
        'volume': 1000,      # Å³
        'mode': 'green_kubo'
    }
})

# Check result
if result.status.value == 'success':
    kappa = result.data['thermal_conductivity']['value']
    print(f"✅ Thermal conductivity: κ = {kappa:.3f} W/(m·K)")
else:
    print(f"❌ Calculation failed: {result.error}")
```

### Example 2: Simulate Active Matter

```python
from active_matter_agent import ActiveMatterAgent

# Create agent
agent = ActiveMatterAgent(config={'backend': 'local'})

# Simulate Vicsek model
result = agent.execute({
    'method': 'vicsek_model',
    'parameters': {
        'N_particles': 1000,
        'box_size': 50.0,
        'velocity': 1.0,
        'noise': 0.1,
        'interaction_radius': 1.0,
        'timesteps': 10000,
        'dt': 0.01
    },
    'analysis': ['order_parameter', 'structure_factor']
})

# Check result
if result.status.value == 'success':
    phi = result.data['order_parameter']['value']
    print(f"✅ Order parameter: φ = {phi:.3f}")
else:
    print(f"❌ Simulation failed: {result.error}")
```

### Example 3: Validate Jarzynski Equality

```python
from fluctuation_agent import FluctuationAgent
import numpy as np

# Create agent
agent = FluctuationAgent(config={'backend': 'local'})

# Generate mock work distribution
# (In real use, this comes from NEMD simulations)
work_values = np.random.exponential(scale=5.0, size=1000)  # kT

# Validate Jarzynski equality
result = agent.execute({
    'method': 'jarzynski',
    'data': {'work': work_values},
    'parameters': {
        'temperature': 300,  # K
        'free_energy_change': 3.5  # kT (if known)
    },
    'analysis': ['work_distribution', 'free_energy', 'jarzynski_ratio']
})

# Check result
if result.status.value == 'success':
    delta_F = result.data['free_energy']['value']
    ratio = result.data['jarzynski_ratio']['value']
    print(f"✅ Free energy: ΔF = {delta_F:.3f} kT")
    print(f"✅ Jarzynski ratio: {ratio:.6f} (should be ~1.0)")
else:
    print(f"❌ Analysis failed: {result.error}")
```

---

## 4. Multi-Agent Workflow (Optional)

```python
from active_matter_agent import ActiveMatterAgent
from pattern_formation_agent import PatternFormationAgent

# Step 1: Simulate active matter
active = ActiveMatterAgent()
result1 = active.execute({
    'method': 'vicsek_model',
    'parameters': {'N_particles': 1000, 'noise': 0.1, ...}
})

# Step 2: Analyze patterns
pattern = PatternFormationAgent()
result2 = pattern.detect_patterns_in_active_matter(result1)

# Step 3: Check detected patterns
if result2.status.value == 'success':
    patterns = result2.data['patterns_detected']
    print(f"✅ Patterns found: {patterns}")
```

---

## 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for specific agent
pytest tests/test_transport_agent.py -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

## 6. Next Steps

- **Learn more**: Read [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
- **See all agents**: Check [README.md](../README.md) for complete agent catalog
- **Phase details**: Explore [docs/phases/](phases/) for phase-specific achievements
- **Verification**: Review [docs/VERIFICATION_HISTORY.md](VERIFICATION_HISTORY.md) for quality scores

---

## Common Issues

### ImportError: No module named 'base_agent'

```bash
# Make sure you're in the project directory
cd /Users/b80985/.claude/agents/nonequilibrium-physics-agents

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Tests failing with "fixture not found"

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Re-run tests
pytest tests/ -v
```

### Agent execution returns ERROR status

Check the `result.error` message for details:

```python
result = agent.execute({...})
if result.status.value == 'error':
    print(f"Error: {result.error}")
    print(f"Metadata: {result.metadata}")
```

---

**Ready to go!** 🚀

For more advanced workflows, see [README.md](../README.md#example-workflows).
