# Code Migration Plugin

Legacy scientific code modernization with cross-language migration for Fortran/C/MATLAB to Python/JAX/Julia while preserving numerical accuracy.

## Overview

This plugin specializes in migrating legacy scientific codebases to modern frameworks, ensuring numerical accuracy and computational efficiency are preserved throughout the migration process.

## Features

- **Cross-Language Migration**: Fortran/C/MATLAB → Python/JAX/Julia
- **Numerical Accuracy Preservation**: Bit-for-bit validation where possible
- **Dependency Analysis**: Automatic detection and modern equivalents
- **Build System Modernization**: Makefile/CMake → modern Python/Julia packaging
- **Performance Optimization**: Leveraging modern frameworks for acceleration

## Commands

### /adopt-code
Analyze, integrate, and optimize scientific computing codebases for modern frameworks.

**Usage:**
```bash
/adopt-code <path-to-legacy-code> [target-framework]
```

**Features:**
- Automatic language and framework detection
- Dependency mapping and modernization
- Parallelization pattern identification (MPI, OpenMP)
- Numerical validation strategies
- Step-by-step migration planning

## Agents

### scientific-code-adoptor
Expert in legacy scientific code modernization with deep understanding of Fortran/C/MATLAB patterns and modern Python/JAX/Julia equivalents.

**Specializations:**
- Fortran 77/90/95/2003 to Python/Julia
- MATLAB to NumPy/JAX
- C/C++ numerical libraries to modern equivalents
- Numerical accuracy validation
- Performance benchmarking

## Installation

### From GitHub Marketplace

```bash
/plugin marketplace add <your-username>/scientific-computing-workflows
/plugin install code-migration
```

### Local Installation

```bash
/plugin add ./plugins/code-migration
```

## Usage Examples

### Migrate Legacy Fortran Code
```bash
/adopt-code ./legacy_fortran_code python
```

### Analyze MATLAB Codebase
```bash
/adopt-code ./matlab_project jax
```

## Requirements

- Python 3.12+
- For migrations: target framework installed (JAX, Julia, etc.)
- Optional: f2py, ctypes for hybrid solutions

## License

MIT

## Author

Wei Chen (wchen@anl.gov)
