---
title: "JAX Init"
description: "Initialize a JAX project with essential imports, PRNG setup, and functional random number generation"
category: scientific-computing
subcategory: jax-core
complexity: basic
argument-hint: "[--agents=auto|jax|scientific|ai|setup|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--configure]"
allowed-tools: "*"
model: inherit
tags: jax, initialization, prng, random-numbers, project-setup
dependencies: []
related: [jax-essentials, jax-models, jax-training, jax-debug]
workflows: [jax-development, project-initialization, scientific-computing]
version: "2.1"
last-updated: "2025-09-28"
---

# JAX Init

Initialize a JAX project with essential imports, PRNG setup, and functional random number generation.

## Quick Start

```bash
# Basic JAX project initialization
/jax-init

# JAX setup with agent assistance
/jax-init --agents=jax --configure

# Advanced initialization with optimization
/jax-init --agents=auto --intelligent --optimize

# Complete setup with breakthrough configuration
/jax-init --agents=all --orchestrate --breakthrough
```

## Usage

```bash
/jax-init [options]
```

**Parameters:**
- `options` - Project configuration, agent selection, and optimization settings

## Options

- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, setup, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with project intelligence
- `--intelligent`: Enable intelligent agent selection based on project analysis
- `--breakthrough`: Enable breakthrough project setup optimization
- `--optimize`: Apply performance optimization to project configuration
- `--configure`: Advanced configuration optimization with agent intelligence

## What This Command Does

1. **Essential Imports**: Sets up core JAX modules (`jax`, `jax.numpy`, `jax.random`)
2. **PRNG Initialization**: Creates a PRNG key for reproducible randomness
3. **Key Management**: Provides examples of proper key splitting and usage
4. **Best Practices**: Shows functional random number generation patterns
5. **23-Agent Project Intelligence**: Multi-agent collaboration for optimal project setup
6. **Advanced Configuration**: Agent-driven project configuration and optimization
7. **Intelligent Setup**: Agent-coordinated setup across multiple domains and use cases

## 23-Agent Intelligent Project Setup System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes project requirements, use case patterns, and development goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Project Type Detection → Agent Selection
- Research Projects → research-intelligence-master + scientific-computing-master + jax-pro
- Production ML → ai-systems-architect + jax-pro + systems-architect
- Scientific Computing → scientific-computing-master + jax-pro + research-intelligence-master
- Educational Projects → documentation-architect + jax-pro + neural-networks-master
- Experimental Development → research-intelligence-master + multi-agent-orchestrator + jax-pro
```

### Core JAX Project Setup Agents

#### **`jax-pro`** - JAX Ecosystem Setup Expert
- **JAX Configuration**: Deep expertise in JAX project setup and configuration optimization
- **Performance Setup**: Initial project configuration for optimal JAX performance
- **Device Configuration**: Multi-device and GPU/TPU setup optimization
- **Best Practices**: JAX ecosystem best practices and development patterns
- **Integration Setup**: JAX integration with other scientific computing libraries

#### **`systems-architect`** - Project Infrastructure & Architecture
- **Project Architecture**: System-level project design and infrastructure setup
- **Development Environment**: Optimal development environment configuration
- **Resource Management**: Computational resource setup and optimization
- **Scalability Planning**: Project architecture for scalable development
- **Infrastructure Integration**: Project integration with larger system architectures

#### **`ai-systems-architect`** - AI/ML Project Setup & Integration
- **ML Project Architecture**: Machine learning project setup and configuration
- **Production Readiness**: Project setup for production AI system deployment
- **Scalability Engineering**: Project design for distributed and large-scale AI systems
- **MLOps Integration**: Project setup for continuous integration and deployment
- **Performance Engineering**: AI system performance optimization from project inception

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Project Setup
Automatically analyzes project requirements and selects optimal agent combinations:
- **Project Analysis**: Detects project type, complexity, development goals
- **Requirement Assessment**: Evaluates project requirements and constraints
- **Agent Matching**: Maps project needs to relevant agent expertise
- **Setup Optimization**: Balances comprehensive setup with development efficiency

#### **`jax`** - JAX-Specialized Project Setup Team
- `jax-pro` (JAX ecosystem lead)
- `systems-architect` (infrastructure setup)
- `ai-systems-architect` (ML integration)
- `neural-networks-master` (ML project setup)

#### **`scientific`** - Scientific Computing Project Setup Team
- `scientific-computing-master` (lead)
- `jax-pro` (JAX implementation)
- `research-intelligence-master` (research methodology)
- `documentation-architect` (research documentation)

#### **`ai`** - AI/ML Project Setup Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (ML setup)
- `jax-pro` (JAX optimization)
- `systems-architect` (infrastructure)

#### **`all`** - Complete 23-Agent Project Setup Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough project initialization.

### Advanced 23-Agent Project Setup Examples

```bash
# Intelligent auto-selection for project setup
/jax-init --agents=auto --intelligent --optimize

# Scientific computing project setup with specialized agents
/jax-init --agents=scientific --breakthrough --orchestrate

# AI/ML project setup with production focus
/jax-init --agents=ai --optimize --configure --intelligent

# Research-grade project development
/jax-init --agents=all --breakthrough --orchestrate --configure

# JAX-specialized project optimization
/jax-init --agents=jax --optimize --intelligent

# Complete 23-agent project ecosystem
/jax-init --agents=all --orchestrate --breakthrough --intelligent
```

## Generated Code

```python
import jax
import jax.numpy as jnp
import jax.random as random

# Initialize PRNG key
key = random.PRNGKey(0)

# JAX Principle 3: Functional Random Number Generation
# JAX requires explicit key management for reproducible randomness
# Always split keys before generating random numbers

# Example usage:
def generate_data(key, shape):
    key, subkey = random.split(key)
    data = random.normal(subkey, shape)
    return key, data  # Return updated key

# Split key for multiple random operations
key, subkey1, subkey2 = random.split(key, 3)
x = random.normal(subkey1, (100,))
y = random.uniform(subkey2, (100,))

# Best practices:
# 1. Always split keys before use
# 2. Never reuse the same key
# 3. Thread keys through your program
# 4. Use jax.random functions, not numpy.random

print("JAX project initialized with PRNG setup")
print("Remember: JAX uses functional RNG - always split keys!")
```

## JAX Principles

### Functional Random Number Generation

| Concept | Description | Example |
|---------|-------------|----------|
| **Explicit Keys** | No global random state | `key = random.PRNGKey(42)` |
| **Key Splitting** | Split keys before use | `key, subkey = random.split(key)` |
| **No Reuse** | Never reuse the same key | Split new subkeys for each operation |
| **Threading** | Pass keys through functions | Return updated keys from functions |

### Core JAX Modules

- **`jax`**: Core transformations (jit, grad, vmap, pmap)
- **`jax.numpy as jnp`**: Array operations compatible with NumPy
- **`jax.random`**: Functional random number generation

### Key Management Patterns

```python
# Single operation
key, subkey = random.split(key)
data = random.normal(subkey, shape)

# Multiple operations
key, *subkeys = random.split(key, 3)
x = random.normal(subkeys[0], shape1)
y = random.uniform(subkeys[1], shape2)
```

## Agent-Enhanced Project Setup Integration Patterns

### Complete Project Initialization Workflow
```bash
# Intelligent project setup and development pipeline
/jax-init --agents=auto --intelligent --optimize --configure
/jax-essentials --agents=auto --intelligent --operation=all
/jax-performance --agents=jax --optimization --gpu-accel
```

### Scientific Computing Project Pipeline
```bash
# High-performance scientific computing project setup
/jax-init --agents=scientific --breakthrough --orchestrate
/jax-models --agents=scientific --breakthrough --architecture=mlp
/jax-training --agents=scientific --optimize --schedule=cosine
```

### Production ML Project Infrastructure
```bash
# Large-scale production ML project initialization
/jax-init --agents=ai --optimize --configure --breakthrough
/jax-data-load --agents=ai --distributed --optimize
/ci-setup --agents=ai --monitoring --performance
```

## Related Commands

**Prerequisites**: Foundation for JAX development
- **Project Setup**: `/jax-init --agents=auto` - Intelligent JAX project initialization

**Core Development**: Essential JAX development with agent intelligence
- `/jax-essentials --agents=auto` - Core JAX operations with agent optimization
- `/jax-debug --agents=jax` - JAX debugging with specialized agents
- `/jax-performance --agents=jax` - Performance optimization with JAX agents

**Application Development**: Specialized JAX development
- `/jax-models --agents=auto` - Neural network model development with agents
- `/jax-training --agents=auto` - Training workflows with intelligent optimization
- `/jax-data-load --agents=auto` - Data pipeline setup with agent intelligence

**Advanced Integration**: Comprehensive JAX workflows
- `/jax-numpyro-prob --agents=scientific` - Probabilistic modeling with scientific agents
- `/jax-sparse-ops --agents=scientific` - Sparse operations with optimization agents
- `/jax-orbax-checkpoint --agents=ai` - Model checkpointing with production agents

**Quality Assurance**: Project validation and optimization
- `/generate-tests --agents=auto --type=jax` - Generate JAX tests with agent intelligence
- `/run-all-tests --agents=jax --scientific` - Comprehensive testing with specialized agents
- `/check-code-quality --agents=auto --language=python` - Code quality with agent optimization