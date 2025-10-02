# Quick Start Guide

Get started with materials characterization analysis in 5 minutes.

## Your First Analysis

### Example 1: Dynamic Light Scattering (DLS)

Measure particle size distribution in a polymer solution.

```python
from light_scattering_agent import LightScatteringAgent

# Create agent
agent = LightScatteringAgent()

# Run DLS measurement
result = agent.execute({
    'technique': 'DLS',
    'parameters': {
        'temperature': 298,  # Kelvin
        'angle': 90,         # degrees
        'duration': 300      # seconds
    }
})

# Check results
if result.success:
    print(f"Mean diameter: {result.data['size_distribution']['mean_diameter_nm']:.1f} nm")
    print(f"PDI: {result.data['size_distribution']['pdi']:.3f}")
    print(f"Data quality: {result.data['data_quality']:.2f}")
else:
    print(f"Error: {result.errors}")
```

**Output Example**:
```
Mean diameter: 125.3 nm
PDI: 0.045
Data quality: 0.94
```

### Example 2: Rheology Measurement

Measure viscoelastic properties of a polymer gel.

```python
from rheologist_agent import RheologistAgent

agent = RheologistAgent()

result = agent.execute({
    'mode': 'oscillatory',
    'frequency_range': [0.1, 100],  # Hz
    'strain': 0.01,                  # 1% strain
    'temperature': 298               # Kelvin
})

if result.success:
    print(f"Storage modulus G': {result.data['G_prime_Pa'][-1]:.1f} Pa")
    print(f"Loss modulus G'': {result.data['G_double_prime_Pa'][-1]:.1f} Pa")
    print(f"tan(δ): {result.data['tan_delta'][-1]:.3f}")
```

### Example 3: CLI Usage

```bash
# Light scattering
/light-scattering --technique=DLS --sample=polymer.dat --temp=298

# Rheology
/rheology --mode=oscillatory --sample=gel.dat --freq-range=0.1,100

# MD simulation
/simulate --engine=lammps --structure=polymer.xyz --steps=1000000

# DFT calculation
/dft --code=vasp --calc=relax --structure=crystal.cif
```

## Understanding Results

### AgentResult Object

Every agent returns an `AgentResult` object with:

```python
result.success          # bool: True if execution succeeded
result.status          # AgentStatus: SUCCESS, FAILED, PENDING
result.data            # dict: Analysis results
result.metadata        # dict: Execution metadata
result.errors          # list: Error messages if any
result.warnings        # list: Warning messages
result.provenance      # ProvenanceInfo: Reproducibility data
```

### Data Quality Scores

All results include data quality scores (0-1 scale):
- **0.95-1.0**: Excellent data quality
- **0.85-0.95**: Good data quality
- **0.70-0.85**: Acceptable data quality
- **< 0.70**: Questionable data quality (check warnings)

## Next Steps

1. **Explore Workflows**: See [Common Workflows](../guides/workflows.md)
2. **Learn More**: Read the [User Guide](../guides/user-guide.md)
3. **Deep Dive**: Check [Architecture Documentation](../architecture/overview.md)
4. **Contributing**: See [Development Guide](../development/contributing.md)
