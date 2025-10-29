# Test Corpus for Plugin Triggering Pattern Testing

This directory contains 16 test samples for validating plugin activation accuracy.

## Overview

The test corpus is organized into categories representing different plugin types and use cases:

- **development**: 3 samples
- **devops**: 2 samples
- **edge-case**: 4 samples
- **multi-language**: 2 samples
- **scientific-computing**: 5 samples

## Sample Types

- **Regular samples**: Typical project structures that should trigger specific plugins
- **Edge cases**: Unusual or boundary-case projects
- **Negative tests**: Projects that should NOT trigger plugins
- **Multi-language**: Projects combining multiple programming languages

## Statistics

- Total samples: 16
- Edge cases: 4
- Negative tests: 2
- Multi-language: 2
- Expected triggers: 14

## Usage

Each sample directory contains:
- **metadata.json**: Sample information and expected behavior
- **Source files**: Sample code files representing the project type

Use these samples with the triggering pattern analysis tools:
- `activation-tester.py`: Test plugin activation accuracy
- `command-analyzer.py`: Test command suggestion relevance
- `skill-validator.py`: Test skill pattern matching

## Sample List

### julia-diffeq-project
- **Category**: scientific-computing
- **Description**: Julia project with differential equations
- **Expected Plugins**: julia-development
- **Should Trigger**: Yes

### jax-neural-ode-project
- **Category**: scientific-computing
- **Description**: Python project with JAX for neural ODEs
- **Expected Plugins**: python-development, jax-implementation
- **Should Trigger**: Yes

### mpi-simulation-project
- **Category**: scientific-computing
- **Description**: C++ HPC project with MPI parallelization
- **Expected Plugins**: hpc-computing
- **Should Trigger**: Yes

### molecular-dynamics-project
- **Category**: scientific-computing
- **Description**: Molecular dynamics simulation with LAMMPS
- **Expected Plugins**: molecular-simulation
- **Should Trigger**: Yes

### pytorch-transformer-project
- **Category**: scientific-computing
- **Description**: PyTorch transformer model training
- **Expected Plugins**: deep-learning, python-development
- **Should Trigger**: Yes

### typescript-react-app
- **Category**: development
- **Description**: TypeScript React application with modern tooling
- **Expected Plugins**: javascript-typescript, frontend-mobile-development
- **Should Trigger**: Yes

### rust-cli-tool
- **Category**: development
- **Description**: Rust CLI tool with async I/O
- **Expected Plugins**: systems-programming, cli-tool-design
- **Should Trigger**: Yes

### fastapi-backend
- **Category**: development
- **Description**: FastAPI backend with SQLAlchemy and authentication
- **Expected Plugins**: python-development, backend-development
- **Should Trigger**: Yes

### github-actions-cicd
- **Category**: devops
- **Description**: GitHub Actions CI/CD workflow
- **Expected Plugins**: cicd-automation
- **Should Trigger**: Yes

### pytest-test-suite
- **Category**: devops
- **Description**: Comprehensive pytest test suite
- **Expected Plugins**: unit-testing, python-development
- **Should Trigger**: Yes

### empty-project
- **Category**: edge-case
- **Description**: Empty project directory (should not trigger)
- **Expected Plugins**: None
- **Should Trigger**: No
- **Flags**: Edge Case, Negative Test

### web-frontend-only
- **Category**: edge-case
- **Description**: Pure frontend web project (no backend/scientific)
- **Expected Plugins**: javascript-typescript, frontend-mobile-development
- **Should Trigger**: Yes
- **Flags**: Edge Case

### julia-web-api
- **Category**: edge-case
- **Description**: Julia web API (not scientific computing)
- **Expected Plugins**: julia-development
- **Should Trigger**: Yes
- **Flags**: Edge Case

### config-files-only
- **Category**: edge-case
- **Description**: Only configuration files, no code
- **Expected Plugins**: None
- **Should Trigger**: No
- **Flags**: Edge Case, Negative Test

### python-cpp-extension
- **Category**: multi-language
- **Description**: Python with C++ extensions using pybind11
- **Expected Plugins**: python-development, systems-programming
- **Should Trigger**: Yes
- **Flags**: Multi-Language

### julia-python-workflow
- **Category**: multi-language
- **Description**: Julia and Python integration for data science
- **Expected Plugins**: julia-development, python-development
- **Should Trigger**: Yes
- **Flags**: Multi-Language

