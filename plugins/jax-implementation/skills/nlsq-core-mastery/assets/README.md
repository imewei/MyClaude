# NLSQ Interactive Notebooks

Interactive Jupyter notebooks for learning and exploring NLSQ capabilities.

---

## 📓 Available Notebooks

### 1. `nlsq_quickstart.ipynb`
**Quick 10-minute introduction to NLSQ**

**What you'll learn**:
- Install and setup NLSQ
- Your first curve fit in 5 lines
- Basic API overview (CurveFit, fit(), results)
- When to use NLSQ vs SciPy

**Perfect for**:
- First-time users
- Quick project start
- API familiarization

**Duration**: 10 minutes

**Run**:
```bash
jupyter notebook nlsq_quickstart.ipynb
```

---

### 2. `nlsq_interactive_tutorial.ipynb`
**Comprehensive hands-on tutorial**

**What you'll learn**:
- Complete NLSQ workflow from data to results
- CurveFit API in depth
- StreamingOptimizer for large datasets
- Loss functions (linear, huber, cauchy, arctan)
- Algorithm selection (TRF vs LM)
- Convergence diagnostics
- Parameter bounds and constraints
- Best practices for production

**Perfect for**:
- Deep learning after quickstart
- Understanding all features
- Building production systems

**Duration**: 45 minutes

**Run**:
```bash
jupyter notebook nlsq_interactive_tutorial.ipynb
```

---

### 3. `advanced_features_demo.ipynb`
**Advanced techniques and optimization**

**What you'll learn**:
- Custom Jacobian implementations (analytical derivatives)
- Constraint handling beyond simple bounds
- Multi-model simultaneous fitting
- Performance profiling and optimization
- GPU/TPU utilization maximization
- Memory management for very large datasets
- Production deployment patterns

**Perfect for**:
- Power users
- Performance-critical applications
- Complex fitting scenarios

**Duration**: 30 minutes

**Run**:
```bash
jupyter notebook advanced_features_demo.ipynb
```

---

## 🎯 Suggested Learning Path

### Beginner (Day 1)
1. Start with `nlsq_quickstart.ipynb` (10 min)
2. Run through examples interactively
3. Modify code cells to fit your data

### Intermediate (Day 2-3)
1. Complete `nlsq_interactive_tutorial.ipynb` (45 min)
2. Explore each feature section
3. Try exercises and challenges

### Advanced (Day 4-5)
1. Study `advanced_features_demo.ipynb` (30 min)
2. Implement custom Jacobians for your models
3. Profile and optimize your fits

---

## 💻 Running Notebooks

### Prerequisites
```bash
pip install nlsq jupyter matplotlib numpy scipy
```

### Launch Jupyter
```bash
cd /path/to/assets/
jupyter notebook
```

### Or use JupyterLab
```bash
jupyter lab
```

### Google Colab
These notebooks can also run in Google Colab (free GPU):
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Install NLSQ: `!pip install nlsq`
4. Run cells

---

## 🔧 Using as Templates

These notebooks serve as **templates** for your projects:

### Template 1: Quick Fit Project
```bash
# Copy quickstart as template
cp nlsq_quickstart.ipynb my_project_fit.ipynb

# Edit to:
# - Load your data
# - Define your model
# - Fit and analyze
```

### Template 2: Production Workflow
```bash
# Copy interactive tutorial as template
cp nlsq_interactive_tutorial.ipynb production_workflow.ipynb

# Adapt sections:
# - Data loading pipeline
# - Model definition
# - Convergence monitoring
# - Results export
```

### Template 3: Advanced Application
```bash
# Copy advanced demo as template
cp advanced_features_demo.ipynb advanced_fitting.ipynb

# Implement:
# - Custom Jacobians
# - Streaming optimization
# - Performance profiling
```

---

## 📚 Notebook Contents Overview

### nlsq_quickstart.ipynb
```
1. Installation
2. First Fit (Exponential Decay)
3. API Basics
   - curve_fit() function
   - Model definition
   - Results interpretation
4. When to Use NLSQ
5. Next Steps
```

### nlsq_interactive_tutorial.ipynb
```
1. Introduction
2. Installation and Setup
3. Basic Curve Fitting
   - CurveFit API
   - Model functions
   - Initial guesses
4. Loss Functions
   - Linear (standard LS)
   - Huber (robust)
   - Cauchy (very robust)
   - When to use each
5. Algorithm Selection
   - TRF (bounded)
   - LM (unbounded)
   - Comparison
6. Parameter Bounds
7. Convergence Diagnostics
8. StreamingOptimizer
   - Large dataset handling
   - Memory management
9. Best Practices
10. Exercises
```

### advanced_features_demo.ipynb
```
1. Custom Jacobians
   - Analytical derivatives
   - Performance comparison
   - When to use
2. Constraint Handling
   - Nonlinear constraints
   - Parameter relationships
3. Multi-Model Fitting
   - Simultaneous fits
   - Shared parameters
4. Performance Optimization
   - Profiling
   - GPU utilization
   - Memory optimization
5. Production Patterns
   - Error handling
   - Monitoring
   - Logging
6. Advanced Examples
```

---

## 🎓 Interactive Exercises

Each notebook includes hands-on exercises:

**Quickstart**:
- Exercise 1: Fit your own data
- Exercise 2: Try different initial guesses
- Exercise 3: Compare with SciPy

**Tutorial**:
- Exercise 1: Fit with different loss functions
- Exercise 2: Set appropriate parameter bounds
- Exercise 3: Diagnose poor convergence
- Exercise 4: Use StreamingOptimizer on 1M points

**Advanced**:
- Exercise 1: Implement analytical Jacobian
- Exercise 2: Set up multi-model fit
- Exercise 3: Profile and optimize your fit
- Exercise 4: Deploy with monitoring

---

## 📊 Feature Coverage

| Feature | Quickstart | Tutorial | Advanced |
|---------|-----------|----------|----------|
| Basic fitting | ✅ | ✅ | ✅ |
| Loss functions | ❌ | ✅ | ✅ |
| Algorithms | ❌ | ✅ | ✅ |
| Parameter bounds | ❌ | ✅ | ✅ |
| Diagnostics | ❌ | ✅ | ✅ |
| StreamingOptimizer | ❌ | ✅ | ✅ |
| Custom Jacobians | ❌ | ❌ | ✅ |
| Multi-model | ❌ | ❌ | ✅ |
| Performance | ❌ | ❌ | ✅ |
| Production patterns | ❌ | ✅ | ✅ |

---

## 🚀 Quick Start Commands

```bash
# Open all notebooks
jupyter notebook

# Open specific notebook
jupyter notebook nlsq_quickstart.ipynb

# Convert to Python script (for automation)
jupyter nbconvert --to script nlsq_quickstart.ipynb

# Execute notebook from command line
jupyter nbconvert --to notebook --execute nlsq_quickstart.ipynb

# Generate HTML report
jupyter nbconvert --to html nlsq_interactive_tutorial.ipynb
```

---

## 💡 Tips

**For learning**:
- Run cells sequentially (don't skip)
- Modify parameters and re-run
- Complete all exercises
- Compare with expected outputs

**For projects**:
- Copy notebook as template
- Replace synthetic data with your data
- Keep diagnostic visualizations
- Add your domain-specific analysis

**For collaboration**:
- Clear outputs before committing: `jupyter nbconvert --clear-output *.ipynb`
- Use version control for notebooks
- Add markdown cells documenting your changes
- Share via nbviewer or Colab

---

## 🔗 Related Resources

**Python Scripts**: See `../scripts/examples/` for executable Python examples
- Gallery examples: Domain-specific complete workflows
- Streaming examples: Large-scale patterns
- Demo examples: Feature showcases

**Documentation**: See `../references/` for detailed guides
- `convergence_diagnostics.md`: Troubleshooting guide
- `loss_functions.md`: Loss function reference

**Main Skill**: See `../SKILL.md` for complete NLSQ core mastery guide

---

**Notebooks Version**: 1.0.0
**Source**: NLSQ official examples
**Last Updated**: 2025-10-28
