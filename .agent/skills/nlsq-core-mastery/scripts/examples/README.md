# NLSQ Examples Collection

Comprehensive examples demonstrating NLSQ library capabilities across scientific domains, streaming optimization, and advanced features.

**Total Examples**: 20 Python scripts + 3 Jupyter notebooks

---

## 📁 Directory Structure

```
examples/
├── gallery/           # Domain-specific complete workflows (11 examples)
│   ├── physics/       # Physical sciences (3 examples)
│   ├── chemistry/     # Chemical processes (2 examples)
│   ├── biology/       # Biological systems (3 examples)
│   └── engineering/   # Engineering applications (3 examples)
├── streaming/         # Large-scale optimization patterns (4 examples)
├── demos/             # Feature showcases (4 examples)
└── README.md          # This file

../../assets/          # Interactive notebooks (3 notebooks)
├── nlsq_quickstart.ipynb
├── nlsq_interactive_tutorial.ipynb
└── advanced_features_demo.ipynb
```

---

## 🎯 Quick Start

### For Beginners

**Start here**: `../../assets/nlsq_quickstart.ipynb`
- Interactive Jupyter notebook
- 10-minute introduction to NLSQ
- Basic curve fitting workflow
- No prior experience needed

### For Learning

**Next**: `../../assets/nlsq_interactive_tutorial.ipynb`
- Comprehensive tutorial
- Covers all major features
- Hands-on exercises
- Best practices

### For Production

**Then**: Explore `gallery/` and `streaming/` examples
- Real-world complete workflows
- Production-ready patterns
- Domain-specific applications

---

## 📚 Gallery Examples

Complete, production-ready examples for specific scientific domains.

### Physics (3 examples)

#### 1. `gallery/physics/radioactive_decay.py`
**Application**: Half-life determination from radioactive decay data

**Demonstrates**:
- Exponential decay fitting
- Uncertainty propagation to derived quantities
- Goodness-of-fit analysis (χ² statistic)
- Parameter correlation analysis
- Weighted least squares with measurement uncertainties

**Model**: N(t) = N₀ * exp(-λt)

**Run**:
```bash
python gallery/physics/radioactive_decay.py
```

**Output**: Fitted half-life with uncertainty, 4-panel diagnostic plots

---

#### 2. `gallery/physics/damped_oscillation.py`
**Application**: Fitting damped harmonic oscillator (pendulum, spring-mass systems)

**Demonstrates**:
- Periodic function fitting
- Damping coefficient extraction
- Resonance frequency determination
- Phase analysis

**Model**: y = A * exp(-γt) * cos(ωt + φ)

**Run**:
```bash
python gallery/physics/damped_oscillation.py
```

---

#### 3. `gallery/physics/spectroscopy_peaks.py`
**Application**: Multi-peak fitting for spectroscopy data (XRF, NMR, chromatography)

**Demonstrates**:
- Simultaneous fitting of multiple Gaussian peaks
- Peak position, amplitude, and width extraction
- Baseline correction
- Peak area integration for quantification
- Many-parameter optimization (TRF algorithm)

**Model**: y = Σᵢ Aᵢ * exp(-(x-μᵢ)²/(2σᵢ²)) + baseline

**Run**:
```bash
python gallery/physics/spectroscopy_peaks.py
```

---

### Chemistry (2 examples)

#### 4. `gallery/chemistry/titration_curves.py`
**Application**: Acid-base titration curve fitting to determine pKa

**Demonstrates**:
- Sigmoidal curve fitting
- Equivalence point determination
- pKa extraction from inflection point
- Buffer capacity analysis

**Model**: Henderson-Hasselbalch equation

**Run**:
```bash
python gallery/chemistry/titration_curves.py
```

---

#### 5. `gallery/chemistry/reaction_kinetics.py`
**Application**: Chemical reaction rate constant determination

**Demonstrates**:
- First-order, second-order kinetics
- Arrhenius equation fitting
- Activation energy calculation
- Rate constant extraction

**Model**: [A] = [A]₀ * exp(-kt) or 1/[A] = 1/[A]₀ + kt

**Run**:
```bash
python gallery/chemistry/reaction_kinetics.py
```

---

### Biology (3 examples)

#### 6. `gallery/biology/dose_response.py`
**Application**: Drug efficacy, IC50/EC50 determination

**Demonstrates**:
- Sigmoid dose-response curve fitting
- 4-parameter logistic model
- IC50/EC50 calculation with confidence intervals
- Hill slope interpretation
- Biological parameter extraction

**Model**: y = bottom + (top-bottom) / (1 + (x/EC50)^(-hill))

**Run**:
```bash
python gallery/biology/dose_response.py
```

**Output**: EC50, Hill slope, dynamic range, dose-response plot

---

#### 7. `gallery/biology/growth_curves.py`
**Application**: Bacterial/cell growth kinetics

**Demonstrates**:
- Logistic growth model fitting
- Growth rate calculation
- Carrying capacity determination
- Lag phase analysis

**Model**: N(t) = K / (1 + ((K-N₀)/N₀) * exp(-rt))

**Run**:
```bash
python gallery/biology/growth_curves.py
```

---

#### 8. `gallery/biology/enzyme_kinetics.py`
**Application**: Michaelis-Menten enzyme kinetics

**Demonstrates**:
- Michaelis-Menten model fitting
- Vmax and Km determination
- Enzyme efficiency (kcat/Km)
- Substrate affinity analysis

**Model**: v = Vmax * [S] / (Km + [S])

**Run**:
```bash
python gallery/biology/enzyme_kinetics.py
```

---

### Engineering (3 examples)

#### 9. `gallery/engineering/sensor_calibration.py`
**Application**: Sensor response calibration and linearization

**Demonstrates**:
- Multi-point calibration
- Nonlinear response correction
- Uncertainty quantification
- Calibration curve validation

**Model**: Polynomial or power law calibration

**Run**:
```bash
python gallery/engineering/sensor_calibration.py
```

---

#### 10. `gallery/engineering/system_identification.py`
**Application**: Transfer function fitting, control system characterization

**Demonstrates**:
- First-order, second-order system identification
- Time constant extraction
- Damping ratio determination
- Step response fitting

**Model**: H(s) = K / (τs + 1) or H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)

**Run**:
```bash
python gallery/engineering/system_identification.py
```

---

#### 11. `gallery/engineering/materials_characterization.py`
**Application**: Stress-strain curves, material properties

**Demonstrates**:
- Ramberg-Osgood model fitting
- Young's modulus extraction
- Yield strength determination
- Strain hardening exponent

**Model**: ε = σ/E + K(σ/σ₀)ⁿ

**Run**:
```bash
python gallery/engineering/materials_characterization.py
```

---

## 🌊 Streaming Examples

Production patterns for large-scale optimization (millions+ data points).

### 12. `streaming/01_basic_fault_tolerance.py`
**Focus**: Automatic error handling and recovery

**Demonstrates**:
- Automatic best parameter tracking
- NaN/Inf detection at three validation points
- Adaptive retry strategies for failed batches
- Success rate validation
- Detailed diagnostics

**When to use**: Noisy data, unstable optimization, production environments

**Run**:
```bash
python streaming/01_basic_fault_tolerance.py
```

---

### 13. `streaming/02_checkpoint_resume.py`
**Focus**: Long-running optimization with checkpointing

**Demonstrates**:
- Periodic checkpoint saving
- Resume from checkpoint after interruption
- State preservation (parameters, loss history)
- Fault recovery

**When to use**: Large datasets, cloud interruptions, long-running jobs

**Run**:
```bash
python streaming/02_checkpoint_resume.py
```

---

### 14. `streaming/03_custom_retry_settings.py`
**Focus**: Fine-tuned retry strategies

**Demonstrates**:
- Custom retry limits per batch
- Adaptive batch size reduction
- Failure tolerance configuration
- Diagnostic logging

**When to use**: Difficult convergence, custom fault tolerance requirements

**Run**:
```bash
python streaming/03_custom_retry_settings.py
```

---

### 15. `streaming/04_interpreting_diagnostics.py`
**Focus**: Understanding StreamingOptimizer diagnostics

**Demonstrates**:
- Interpreting convergence metrics
- Analyzing batch success rates
- Identifying optimization issues
- Performance monitoring

**When to use**: Debugging streaming optimization, performance tuning

**Run**:
```bash
python streaming/04_interpreting_diagnostics.py
```

---

## 🎪 Demo Examples

Feature showcases for advanced NLSQ capabilities.

### 16. `demos/hybrid_streaming_demo.py`
**Focus**: Adaptive Hybrid Streaming Optimizer with Parameter Normalization

**Demonstrates**:
- `method='hybrid_streaming'` for multi-scale parameters
- HybridStreamingConfig presets (aggressive, conservative, memory_optimized)
- Automatic parameter normalization (auto, bounds, p0, none strategies)
- Direct ParameterNormalizer and NormalizedModelWrapper usage
- Covariance transformation for normalized parameters
- Three-phase optimization pipeline (Phase 0/1/2)
- Comparison with TRF for ill-conditioned problems

**Three Issues Solved**:
1. Weak gradients (scale imbalance) -> Parameter normalization
2. Slow convergence -> Streaming Gauss-Newton
3. Crude covariance -> Exact J^T J + transform

**When to use**: Parameters differing by >1000x, need accurate covariance, TRF/LM converges slowly

**Run**:
```bash
python demos/hybrid_streaming_demo.py
```

---

### 17. `demos/enhanced_error_messages_demo.py`
**Focus**: Improved error diagnostics

**Demonstrates**:
- Detailed convergence failure messages
- Actionable troubleshooting suggestions
- Parameter bound violations
- Numerical stability warnings

**Run**:
```bash
python demos/enhanced_error_messages_demo.py
```

---

### 18. `demos/function_library_demo.py`
**Focus**: Built-in model functions

**Demonstrates**:
- Pre-defined common models (exponential, sigmoid, polynomial)
- Function composition
- Custom function derivatives
- Model library usage

**Run**:
```bash
python demos/function_library_demo.py
```

---

### 19. `demos/result_enhancements_demo.py`
**Focus**: Rich result objects

**Demonstrates**:
- Extended OptimizeResult attributes
- Covariance matrix analysis
- Parameter correlations
- Confidence intervals
- Diagnostic metrics

**Run**:
```bash
python demos/result_enhancements_demo.py
```

---

### 20. `demos/callbacks_demo.py`
**Focus**: Real-time monitoring with callbacks

**Demonstrates**:
- Per-iteration callbacks
- Progress monitoring
- Early stopping conditions
- Custom logging
- Convergence visualization

**Run**:
```bash
python demos/callbacks_demo.py
```

---

## 📓 Interactive Notebooks (Assets)

Located in `../../assets/`, these Jupyter notebooks provide interactive learning experiences.

### 1. `nlsq_quickstart.ipynb`
**Level**: Beginner
**Duration**: 10 minutes
**Content**:
- Installation and setup
- First curve fit in 5 lines
- Basic API overview
- Common patterns

**Use as**: Template for new projects, onboarding

---

### 2. `nlsq_interactive_tutorial.ipynb`
**Level**: Intermediate
**Duration**: 45 minutes
**Content**:
- Complete NLSQ workflow
- CurveFit and StreamingOptimizer
- Loss functions and algorithms
- Convergence diagnostics
- Best practices

**Use as**: Comprehensive learning resource

---

### 3. `advanced_features_demo.ipynb`
**Level**: Advanced
**Duration**: 30 minutes
**Content**:
- Custom Jacobians
- Constraint handling
- Multi-model fitting
- Performance optimization
- Production patterns

**Use as**: Advanced techniques reference

---

## 🚀 Usage Patterns

### Pattern 1: Learning NLSQ
```bash
# 1. Start with quickstart notebook
jupyter notebook ../../assets/nlsq_quickstart.ipynb

# 2. Explore a domain-specific example
python gallery/physics/radioactive_decay.py

# 3. Learn streaming for large data
python streaming/01_basic_fault_tolerance.py
```

### Pattern 2: Finding Domain-Specific Example
```bash
# Physics: radioactive decay, oscillations, spectroscopy
ls gallery/physics/

# Chemistry: titration, kinetics
ls gallery/chemistry/

# Biology: dose-response, growth, enzymes
ls gallery/biology/

# Engineering: sensors, control, materials
ls gallery/engineering/
```

### Pattern 3: Production Deployment
```bash
# 1. Study streaming patterns
cd streaming/
python 01_basic_fault_tolerance.py
python 02_checkpoint_resume.py

# 2. Explore diagnostics
python 04_interpreting_diagnostics.py

# 3. Adapt for your application
# Copy example to your project and modify
```

---

## 📊 Example Complexity

| Example | Lines | Level | Domain | GPU Benefit |
|---------|-------|-------|--------|-------------|
| radioactive_decay | 262 | Beginner | Physics | Medium |
| dose_response | ~250 | Beginner | Biology | Medium |
| damped_oscillation | ~280 | Intermediate | Physics | Medium |
| spectroscopy_peaks | ~320 | Intermediate | Physics | High |
| enzyme_kinetics | ~240 | Intermediate | Biology | Low |
| sensor_calibration | ~270 | Intermediate | Engineering | Medium |
| basic_fault_tolerance | ~180 | Intermediate | Streaming | High |
| checkpoint_resume | ~200 | Advanced | Streaming | High |
| hybrid_streaming_demo | ~290 | Intermediate | Features | High |
| callbacks_demo | ~160 | Advanced | Features | Medium |

---

## 🎓 Learning Path

**Week 1: Foundations**
- Day 1-2: nlsq_quickstart.ipynb
- Day 3-4: radioactive_decay.py, dose_response.py
- Day 5: nlsq_interactive_tutorial.ipynb

**Week 2: Domain Applications**
- Choose 2-3 examples from your domain (physics/chemistry/biology/engineering)
- Study model formulation
- Adapt to your data

**Week 3: Advanced Topics**
- Streaming examples for large-scale data
- Demo examples for advanced features
- advanced_features_demo.ipynb

**Week 4: Production**
- Fault tolerance patterns
- Checkpoint/resume workflows
- Performance optimization

---

## 🔧 Common Tasks

### Task: Fit exponential decay
→ `gallery/physics/radioactive_decay.py`

### Task: Fit sigmoid curve
→ `gallery/biology/dose_response.py` or `gallery/chemistry/titration_curves.py`

### Task: Fit multiple peaks
→ `gallery/physics/spectroscopy_peaks.py`

### Task: Large dataset (>1M points)
→ `streaming/01_basic_fault_tolerance.py`

### Task: Production deployment
→ `streaming/02_checkpoint_resume.py`

### Task: Real-time monitoring
→ `demos/callbacks_demo.py`

### Task: Multi-scale parameters (>1000x difference)
→ `demos/hybrid_streaming_demo.py`

### Task: Uncertainty analysis
→ `gallery/physics/radioactive_decay.py` (see uncertainty propagation)

---

## 💡 Tips

**All examples**:
- Are self-contained (run independently)
- Generate synthetic data (no external files needed)
- Include visualization
- Show best practices
- Have comprehensive docstrings

**For production use**:
- Start with gallery examples matching your domain
- Add streaming patterns from streaming/ for large data
- Integrate fault tolerance from streaming/01_basic_fault_tolerance.py
- Use callbacks from demos/callbacks_demo.py for monitoring

**For learning**:
- Begin with notebooks (assets/)
- Progress through gallery examples
- Study streaming patterns
- Explore advanced demos

---

## 📖 Further Resources

- **NLSQ Documentation**: https://nlsq.readthedocs.io/
- **NLSQ Repository**: https://github.com/imewei/NLSQ
- **Skill Guide**: `../../SKILL.md`
- **Diagnostic Reference**: `../../references/convergence_diagnostics.md`
- **Loss Functions**: `../../references/loss_functions.md`

---

**Examples Collection Version**: 1.1.0
**Last Updated**: 2025-12-18
**Total Examples**: 20 scripts + 3 notebooks
**Source**: NLSQ official examples (adapted for skill)
