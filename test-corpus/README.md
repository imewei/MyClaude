# Skill Validator Test Corpus

A small set of representative code samples used by `tools/validation/skill_validator.py` to test skill triggering accuracy. Each sample exercises a different language ecosystem and a different cluster of plugin skills.

## Layout

```
test-corpus/
├── README.md                  (this file)
├── <sample-name>/
│   ├── metadata.json          required: at least {"name": "..."}
│   └── <code-files>           any .py/.jl/.js/.ts/.rs/.cpp/.c/.go/.java files
```

The validator walks every code file in every sample subdirectory, runs skill matching against each file, and reports per-skill precision / recall / over-triggering rates.

## Adding a new sample

1. Create `test-corpus/<descriptive-name>/`
2. Add `metadata.json`:
   ```json
   {"name": "descriptive-name", "language": "julia", "domain": "scientific-ml"}
   ```
   The validator only reads `name`. Extra fields are advisory metadata for humans.
3. Add 1–3 code files that realistically exercise the target skill cluster — imports, function names, and patterns matter more than runtime correctness.
4. Re-run `python3 -m tools.validation.skill_validator --corpus-dir test-corpus`.

## Current samples

| Sample | Language | Exercises |
|--------|----------|-----------|
| `julia-sciml` | Julia | SciML, DifferentialEquations.jl, Lux, Turing |
| `julia-gnn` | Julia | GraphNeuralNetworks.jl, GNNLux, MLDatasets |
| `julia-bayesian` | Julia | Turing.jl NUTS sampling, ArviZ diagnostics |
| `python-numpyro` | Python | NumPyro, JAX, Bayesian inference |
| `python-jax-physics` | Python | JAX physics simulations, optimization |
| `python-nonlinear-dynamics` | Python | Lorenz attractor, Lyapunov exponents, SciPy |
| `python-statistical-physics` | Python | Monte Carlo Ising model, Metropolis algorithm |
| `python-pytorch-training` | Python | PyTorch training loop, DataLoader, checkpointing |
| `python-ml-pipeline` | Python | Scikit-learn pipeline, MLflow tracking |
| `python-fastapi-backend` | Python | FastAPI, Pydantic, async SQLAlchemy |
| `python-dspy` | Python | DSPy RAG program, MIPROv2 optimizer |
| `typescript-react` | TypeScript | React, hooks, components |
| `rust-cli` | Rust | CLI tools, error handling, async |
| `go-microservice` | Go | HTTP microservice, goroutines, graceful shutdown |
| `github-actions-ci` | YAML | GitHub Actions CI/CD, lint/test/deploy stages |
