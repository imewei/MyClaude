---
title: "Clean Codebase"
description: "Advanced codebase cleanup with AST-based unused import removal, dead code elimination, and multi-agent analysis with ultrathink intelligence"
category: project-maintenance
subcategory: code-cleanup
complexity: intermediate
argument-hint: "[--dry-run] [--analysis=basic|thorough|comprehensive|ultrathink] [--agents=auto|core|scientific|engineering|domain-specific|all] [--imports] [--dead-code] [--duplicates] [--ast-deep] [--orchestrate] [--intelligent] [--breakthrough] [--parallel] [path]"
allowed-tools: Bash, Read, Write, Edit, MultiEdit, Glob, Grep, Task, TodoWrite
model: inherit
tags: cleanup, refactoring, maintenance, ast-analysis, 23-agent-system, ultrathink, unused-imports, dead-code, duplicates, orchestration
dependencies: []
related: [refactor-clean, check-code-quality, optimize, multi-agent-optimize, adopt-code, debug]
workflows: [project-cleanup, maintenance-workflow, code-organization]
version: "2.1"
last-updated: "2025-09-28"
---

# Clean Codebase

Advanced codebase cleanup with ultrathink intelligence, AST-based unused import removal, sophisticated dead code elimination, and comprehensive 23-agent personal agent analysis for safe, precise cleanup operations.

## Quick Start

```bash
# Safe preview with ultrathink analysis
/clean-codebase --dry-run --analysis=ultrathink

# Remove unused imports across project
/clean-codebase --imports --ast-deep --dry-run

# Comprehensive dead code elimination with 23-agent analysis
/clean-codebase --dead-code --analysis=comprehensive --agents=all --orchestrate

# Complete cleanup with 23-agent orchestration
/clean-codebase --analysis=ultrathink --imports --dead-code --duplicates --agents=all --orchestrate

# Language-specific intelligent cleanup with auto-agent selection
/clean-codebase --language=python --analysis=ultrathink --agents=auto --intelligent
```

## Usage

```bash
/clean-codebase [options] [path]
```

**Parameters:**
- `options` - Analysis depth, agent selection, and execution configuration
- `path` - Directory path to clean (defaults to current directory, moved to end for better UX)

## Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--analysis=<level>` | basic\|thorough\|comprehensive\|ultrathink | thorough | Analysis depth and intelligence level |
| `--agents=<types>` | auto\\|core\\|scientific\\|engineering\\|domain-specific\\|all | auto | 23-agent personal agent selection |
| `--imports` | - | false | Remove unused imports with AST analysis |
| `--dead-code` | - | false | Eliminate unreachable and dead code |
| `--duplicates` | - | false | Remove duplicate files and code blocks |
| `--ast-deep` | - | false | Deep AST analysis for precise detection |
| `--parallel` | - | false | Parallel processing for faster analysis |
| `--dry-run`, `-n` | - | false | Preview changes without executing |
| `--interactive`, `-i` | - | false | Interactive confirmation for each change |
| `--language=<lang>` | python\|javascript\|java\|auto | auto | Language-specific optimization |
| `--exclude=<pattern>` | glob-pattern | none | Exclude files/folders matching pattern |
| `--size-threshold=<size>` | size-spec | none | Only process files larger than threshold |
| `--age-threshold=<days>` | number | none | Only process files older than specified days |
| `--report` | - | false | Generate comprehensive analysis report |
| `--backup` | - | false | Create backup before making changes |
| `--rollback` | - | false | Enable rollback capability for safety |
| `--orchestrate` | - | false | Enable advanced 23-agent orchestration |
| `--intelligent` | - | false | Enable intelligent agent selection based on codebase analysis |
| `--breakthrough` | - | false | Enable breakthrough optimization discovery |

### Analysis Levels
- **`basic`**: Quick scan for obvious duplicates and unused files
- **`thorough`**: AST-based analysis with dependency checking (default)
- **`comprehensive`**: Deep analysis with performance and security audits
- **`ultrathink`**: Advanced intelligence with cross-file analysis and reasoning

### 23-Agent Personal Agent System

#### **Agent Selection Strategies**

##### **`auto`** - Intelligent Agent Selection for Cleanup
Automatically analyzes codebase characteristics and selects optimal agent combinations from the 23-agent library:
- **Pattern Recognition**: Detects code patterns, languages, and cleanup requirements
- **Complexity Assessment**: Evaluates project complexity and required cleanup scope
- **Agent Matching**: Maps cleanup needs to relevant agent expertise
- **Resource Optimization**: Balances comprehensive analysis with execution efficiency

##### **`core`** - Essential Cleanup Team (5 agents)
- `code-quality-master` - Code quality analysis and optimization
- `systems-architect` - Architecture and dependency analysis
- `scientific-computing-master` - Scientific code optimization
- `documentation-architect` - Documentation and code organization
- `multi-agent-orchestrator` - Workflow coordination and resource management

##### **`scientific`** - Scientific Computing Focus (8 agents)
- `scientific-computing-master` - Lead scientific computing expert
- `jax-pro` - JAX ecosystem optimization
- `neural-networks-master` - ML/AI code optimization
- `research-intelligence-master` - Research methodology and analysis
- `advanced-quantum-computing-expert` - Quantum computing optimization
- `correlation-function-expert` - Statistical analysis optimization
- `neutron-soft-matter-expert` - Neutron scattering simulation cleanup
- `nonequilibrium-stochastic-expert` - Stochastic process optimization

##### **`engineering`** - Software Engineering Focus (6 agents)
- `systems-architect` - System design and architecture cleanup
- `fullstack-developer` - Full-stack application optimization
- `devops-security-engineer` - DevSecOps and infrastructure cleanup
- `code-quality-master` - Quality engineering and testing
- `database-workflow-engineer` - Database and data workflow optimization
- `command-systems-engineer` - Command system and automation optimization

##### **`domain-specific`** - Specialized Domain Experts (4 agents)
- `xray-soft-matter-expert` - X-ray analysis workflow optimization
- `scientific-code-adoptor` - Legacy scientific code modernization
- `data-professional` - Data engineering and analytics optimization
- `visualization-interface-master` - UI/UX and visualization cleanup

##### **`all`** - Complete 23-Agent Ecosystem
Activates all 23 specialized personal agents with intelligent orchestration for breakthrough cleanup and optimization capabilities:

**Multi-Agent Orchestration (23 Personal Agents):**
- **`multi-agent-orchestrator`** - Workflow coordination and intelligent task allocation
- **`command-systems-engineer`** - Command system optimization
- **Scientific Computing Agents (8)** - Complete scientific computing optimization
- **Engineering Agents (6)** - Full software engineering optimization
- **Quality & Documentation (2)** - Code quality and documentation optimization
- **Domain Specialists (5)** - Specialized domain expertise

## Quick Agent Selection Guide

**🚀 New User? Start Here:**

| **Your Cleanup Goal** | **Recommended Agents** | **Example Command** |
|-----------------------|----------------------|-------------------|
| **Basic cleanup** | `--agents=auto` | `/clean-codebase --agents=auto --dry-run` |
| **Quick quality fixes** | `--agents=core` | `/clean-codebase --agents=core --imports --dry-run` |
| **Scientific code** | `--agents=scientific` | `/clean-codebase --agents=scientific --dead-code --orchestrate` |
| **Production systems** | `--agents=engineering` | `/clean-codebase --agents=engineering --duplicates --intelligent` |
| **Research projects** | `--agents=domain-specific` | `/clean-codebase --agents=domain-specific --ast-deep` |
| **Complete overhaul** | `--agents=all` | `/clean-codebase --agents=all --analysis=ultrathink --breakthrough` |

**🎯 Quick Decision Tree:**
- **Not sure what you need?** → Use `--agents=auto` (intelligent selection)
- **Basic Python/JS cleanup?** → Use `--agents=core`
- **Scientific computing code?** → Use `--agents=scientific`
- **Production application?** → Use `--agents=engineering`
- **Research/academic code?** → Use `--agents=domain-specific`
- **Major refactoring needed?** → Use `--agents=all`

**⚡ Pro Tips**:
- Always start with `--dry-run` to preview changes
- Use `--orchestrate` for better agent coordination
- Add `--intelligent` for smart cleanup decisions
- Use `--breakthrough` for innovative optimization opportunities

## Advanced Implementation

The command executes sophisticated multi-phase analysis with ultrathink intelligence:

### Phase 1: Ultrathink Project Analysis
**Intelligent Codebase Assessment:**
1. **Deep Structure Analysis**
   - Map file dependencies with import/export tracking
   - Identify language/framework patterns and conventions
   - Detect build artifacts, generated files, and temporary files
   - Analyze module hierarchies and cross-file relationships

2. **AST-Based Code Analysis**
   - Parse source files into Abstract Syntax Trees
   - Track symbol definitions, usages, and scopes
   - Identify control flow patterns and reachability
   - Map import statements to actual usage locations

3. **Baseline Intelligence Gathering**
   - Calculate file/folder sizes and modification timestamps
   - Generate content hashes for duplicate detection
   - Catalog all imports vs. actual symbol usage
   - Create dependency graphs for safe removal validation

### Phase 2: 23-Agent Coordinated Analysis
**Advanced Personal Agent Deployment:**

#### **Tier 1: Core Cleanup Agents (5 agents)**
**`code-quality-master`** - Master Code Quality Analysis
- **Unused Import Detection**: Advanced AST traversal with symbol tracking
- **Dead Code Analysis**: Control flow analysis with reachability assessment
- **Code Quality Metrics**: Complexity analysis and maintainability scoring
- **Refactoring Opportunities**: Identification of cleanup-enabled optimizations

**`systems-architect`** - Architecture and Dependency Analysis
- **Dependency Graph Analysis**: Complete system dependency mapping
- **Architecture Pattern Recognition**: Framework and pattern-specific cleanup
- **Modular Design Assessment**: Component isolation and coupling analysis
- **Build System Integration**: Safe removal validation across build systems

**`scientific-computing-master`** - Scientific Code Optimization
- **Scientific Library Analysis**: NumPy, SciPy, JAX, Julia ecosystem cleanup
- **Research Code Patterns**: Academic code optimization and standardization
- **Performance Critical Path**: Scientific computation bottleneck identification
- **Numerical Accuracy Validation**: Ensure cleanup doesn't affect computational results

**`documentation-architect`** - Documentation and Organization
- **Documentation Sync**: Remove documentation for deleted code
- **Code Organization**: Improve file and module organization
- **Comment and Docstring Cleanup**: Remove obsolete inline documentation
- **README and Guide Updates**: Update project documentation post-cleanup

**`multi-agent-orchestrator`** - Workflow Coordination
- **Agent Task Distribution**: Intelligent workload allocation across 23 agents
- **Conflict Resolution**: Handle competing cleanup recommendations
- **Priority Optimization**: Risk-benefit analysis for cleanup operations
- **Resource Management**: Efficient coordination of agent analysis

#### **Tier 2: Scientific Computing Specialists (8 agents)**
**Advanced Scientific Code Analysis:**
- **`jax-pro`**: JAX ecosystem import optimization and dead code removal
- **`neural-networks-master`**: ML/AI model cleanup and optimization
- **`research-intelligence-master`**: Research methodology and experimental code cleanup
- **`advanced-quantum-computing-expert`**: Quantum computing algorithm optimization
- **Domain-specific scientific agents** for specialized cleanup patterns

#### **Tier 3: Engineering Specialists (6 agents)**
**Production System Optimization:**
- **`fullstack-developer`**: Full-stack application cleanup and optimization
- **`devops-security-engineer`**: Infrastructure and security-focused cleanup
- **`database-workflow-engineer`**: Data pipeline and workflow optimization
- **Engineering agents** for comprehensive system-level cleanup

#### **Tier 4: Domain Specialists (4 agents)**
**Specialized Domain Cleanup:**
- **`scientific-code-adoptor`**: Legacy scientific code modernization
- **`data-professional`**: Data science and analytics optimization
- **Domain experts** for specialized application cleanup

### Advanced 23-Agent Coordination Patterns

#### **Intelligent Agent Activation (`--intelligent`)**
**Auto-Selection Algorithm**: Analyzes codebase characteristics and automatically deploys optimal agent combinations:
```bash
# Codebase Analysis → Agent Selection
- Python/NumPy/SciPy → scientific-computing-master + jax-pro + code-quality-master
- JavaScript/React → fullstack-developer + code-quality-master + systems-architect
- Java Enterprise → systems-architect + devops-security-engineer + code-quality-master
- Research/Academic → research-intelligence-master + scientific-computing-master + documentation-architect
- Mixed/Complex → multi-agent-orchestrator coordinates full 23-agent deployment
```

#### **Advanced Orchestration (`--orchestrate`)**
**23-Agent Workflow Coordination:**
- **Parallel Analysis**: Concurrent agent execution with intelligent resource allocation
- **Cross-Agent Communication**: Shared insights and coordinated decision making
- **Conflict Resolution**: Automated resolution of competing cleanup recommendations
- **Quality Gates**: Multi-agent validation at each cleanup phase
- **Emergent Intelligence**: Pattern recognition across agent collaboration

### Phase 3: Ultrathink Reasoning & Safety Validation
**Intelligent Decision Making:**
1. **Cross-Reference Analysis**
   - Validate removal safety across multiple analysis dimensions
   - Check for indirect dependencies and side effects
   - Analyze potential impact on build systems and workflows
   - Verify test coverage isn't compromised by removals

2. **Risk Assessment Matrix**
   - Classify removal operations by safety level (safe/caution/high-risk)
   - Generate confidence scores for each cleanup operation
   - Identify operations requiring human review
   - Create rollback strategies for each change category

3. **Language-Specific Intelligence**
   - **Python**: Handle `__init__.py`, dynamic imports, `__all__` declarations
   - **JavaScript**: Process ES6 modules, CommonJS, dynamic requires, tree-shaking
   - **Java**: Analyze package structures, reflection usage, annotation processing
   - **TypeScript**: Handle type-only imports, namespace imports, declaration files

### Phase 4: Execution Strategy & Safety Implementation
**Safe, Reversible Cleanup Operations:**

1. **Staged Execution Pipeline**
   - **Stage 1**: Remove unused imports (lowest risk)
   - **Stage 2**: Eliminate dead code blocks (medium risk)
   - **Stage 3**: Remove duplicate files (medium risk)
   - **Stage 4**: Clean obsolete/orphaned files (higher risk)

2. **Safety Mechanisms**
   - Automatic backup creation before any modifications
   - Atomic operation grouping with rollback capability
   - Real-time impact validation during execution
   - Comprehensive logging with detailed change tracking

3. **Verification & Validation**
   - Post-cleanup compilation/syntax checking
   - Automated test execution to verify functionality
   - Dependency resolution validation
   - Performance impact measurement

### Phase 5: Advanced Cleanup Algorithms

#### **Unused Import Removal Algorithm**
```python
# Sophisticated AST-based import analysis
1. Parse all source files into AST trees
2. Extract all import statements and imported symbols
3. Traverse AST to find actual symbol usage locations
4. Handle dynamic imports and string-based references
5. Process wildcard imports and namespace imports
6. Validate removal won't break re-exports or __all__
7. Generate minimal import statements preserving only used symbols
```

#### **Dead Code Elimination Algorithm**
```python
# Control flow analysis for precise dead code detection
1. Build control flow graphs for all functions/methods
2. Identify unreachable code blocks after returns/raises
3. Detect unused functions/classes with no call sites
4. Handle conditional compilation and platform-specific code
5. Preserve code referenced by reflection or dynamic calls
6. Maintain code coverage for edge cases and error handling
```

#### **Duplicate Detection Algorithm**
```python
# Multi-level duplicate analysis
1. Content-based: SHA-256 hash comparison for identical files
2. Semantic-based: AST structure comparison for functionally identical code
3. Near-duplicate: Fuzzy matching with configurable similarity thresholds
4. Refactoring-safe: Preserve duplicates that serve different purposes
5. Cross-language: Detect duplicated logic across different file types
```

### Phase 6: Intelligent Reporting & Insights
**Comprehensive Analysis Output:**
- **Cleanup Summary**: Files/lines removed, space saved, improvement metrics
- **Safety Report**: Risk assessment for each operation, confidence scores
- **Performance Impact**: Build time improvements, runtime optimizations
- **Recommendations**: Additional cleanup opportunities requiring manual review
- **Rollback Guide**: Step-by-step instructions for reversing changes if needed

## Expected Outcomes

**Advanced Cleanup Results:**
- **99%+ Accuracy**: Ultrathink intelligence with AST-based analysis vs regex patterns
- **Unused Import Elimination**: Precise removal of unused imports across entire codebase
- **Dead Code Removal**: Control flow analysis identifies truly unreachable code
- **Duplicate Elimination**: Content and semantic duplicate detection and removal
- **Safe Operations**: Multi-layer safety validation with automatic rollback capability
- **Multi-Language Support**: Language-specific optimizations for Python, JavaScript, Java, TypeScript
- **Performance Gains**: 10-50% reduction in codebase size, improved build times
- **Zero Breakage**: Comprehensive dependency analysis prevents functional regressions

## Examples

### Unused Import Cleanup
```bash
# Remove unused imports with intelligent agent selection
/clean-codebase --imports --agents=auto --intelligent --dry-run

# Python-specific import optimization with scientific agents
/clean-codebase --imports --language=python --agents=scientific --orchestrate

# JavaScript/TypeScript import cleanup with engineering agents
/clean-codebase --imports --language=javascript --agents=engineering --backup
```

### Dead Code Elimination
```bash
# Comprehensive dead code analysis with all 23 agents
/clean-codebase --dead-code --analysis=comprehensive --agents=all --orchestrate

# Safe dead code removal with core agents
/clean-codebase --dead-code --agents=core --interactive --rollback

# Cross-language dead code detection with breakthrough analysis
/clean-codebase --dead-code --analysis=ultrathink --agents=all --breakthrough
```

### Duplicate Detection & Removal
```bash
# Find and remove duplicates with intelligent agent coordination
/clean-codebase --duplicates --agents=auto --intelligent --report

# Semantic duplicate detection with engineering focus
/clean-codebase --duplicates --agents=engineering --orchestrate --interactive
```

### Complete Codebase Optimization
```bash
# Comprehensive cleanup with full 23-agent orchestration
/clean-codebase --analysis=ultrathink --imports --dead-code --duplicates --agents=all --orchestrate --breakthrough

# Safe comprehensive cleanup with intelligent agent selection
/clean-codebase --analysis=ultrathink --imports --dead-code --agents=auto --intelligent --backup --rollback

# Language-specific comprehensive optimization with domain agents
/clean-codebase --language=python --analysis=ultrathink --imports --dead-code --duplicates --agents=scientific --orchestrate
```

### Advanced Filtering & Safety
```bash
# Exclude critical directories with size filtering
/clean-codebase --exclude="node_modules,venv,*.log" --size-threshold=1MB --dry-run

# Age-based cleanup with comprehensive safety
/clean-codebase --age-threshold=30 --analysis=ultrathink --backup --rollback

# High-confidence operations with intelligent orchestration
/clean-codebase --analysis=ultrathink --agents=all --orchestrate --intelligent --report
```

### Production-Safe Workflows
```bash
# Pre-production cleanup pipeline with 23-agent analysis
/clean-codebase --analysis=ultrathink --imports --dead-code --agents=all --orchestrate --backup --dry-run
# Review report, then execute:
/clean-codebase --analysis=ultrathink --imports --dead-code --agents=core --backup --rollback

# Continuous integration cleanup with intelligent agent selection
/clean-codebase --imports --duplicates --agents=auto --intelligent --parallel --report
```

## Related Commands

**Prerequisites**: Commands to run before project cleanup
- `/check-code-quality --auto-fix` - Fix quality issues before cleanup
- `/debug --auto-fix` - Fix runtime issues that might affect cleanup
- Version control - Commit current state before major cleanup
- `/run-all-tests` - Ensure tests pass before cleanup

**Alternatives**: Different cleanup approaches
- `/refactor-clean --patterns=modern` - Code modernization and cleanup
- `/optimize --implement` - Performance-focused cleanup
- Manual cleanup for specific file types or directories
- IDE-based refactoring tools for smaller cleanups

**Combinations**: Commands that work with project cleanup
- `/multi-agent-optimize --mode=review --agents=all` - 23-agent analysis before cleanup
- `/adopt-code --analyze --agents=scientific` - Legacy code analysis and modernization with domain experts
- `/generate-tests --coverage=95 --agents=auto` - Add tests before removing code with intelligent agents
- `/double-check --deep-analysis --auto-complete` - Verify cleanup safety and completeness

**Follow-up**: Commands to run after advanced project cleanup
- `/run-all-tests --auto-fix --coverage` - Ensure cleanup didn't break functionality or reduce test coverage
- `/check-code-quality --language=auto --analysis=comprehensive` - Validate improved code quality metrics
- `/optimize --implement --language=auto --category=all` - Apply performance optimizations to cleaned codebase
- `/generate-tests --coverage=95` - Add tests for any exposed functionality after dead code removal
- `/double-check "cleanup validation" --deep-analysis` - Comprehensive verification of cleanup safety and completeness
- `/commit --template=refactor --ai-message --validate` - Commit cleanup changes with detailed description
- `/ci-setup --type=enterprise` - Update CI/CD pipelines for optimized project structure

## Ultrathink Intelligence Integration

### Advanced Reasoning Capabilities
The `--analysis=ultrathink` mode provides sophisticated intelligence beyond standard analysis:

**Cross-File Reasoning:**
- Analyzes dependencies and relationships across the entire codebase
- Understands complex import/export patterns and circular dependencies
- Recognizes framework-specific patterns and conventions
- Identifies subtle usage patterns that traditional AST analysis might miss

**Contextual Understanding:**
- Distinguishes between intentionally unused code (future features, debugging) and truly dead code
- Recognizes test fixtures, example code, and documentation-related files
- Understands build system implications and deployment considerations
- Analyzes code evolution patterns to predict safe removal candidates

**Safety Intelligence:**
- Predicts potential breaking changes before they occur
- Identifies hidden dependencies through dynamic analysis patterns
- Recognizes reflection, metaprogramming, and runtime code generation
- Validates removal safety across multiple dimensions simultaneously

### Integration with Think-Ultra
```bash
# Use think-ultra for complex cleanup planning
/think-ultra "analyze safe cleanup strategy for large codebase" --agents=all --depth=ultra
/clean-codebase --analysis=ultrathink --imports --dead-code --duplicates
/double-check "cleanup results" --deep-analysis --auto-complete
```

## Language-Specific Optimizations

### Python Cleanup Intelligence
```bash
# Python-specific advanced cleanup
/clean-codebase --language=python --analysis=ultrathink --imports --dead-code
```
**Python Features:**
- `__init__.py` and package structure optimization
- `__all__` declaration validation and cleanup
- Dynamic import pattern recognition (`importlib`, `__import__`)
- Decorator and metaclass usage analysis
- Virtual environment and dependency cleanup

### JavaScript/TypeScript Optimization
```bash
# Modern JavaScript/TypeScript cleanup
/clean-codebase --language=javascript --analysis=ultrathink --imports --duplicates
```
**JavaScript/TypeScript Features:**
- ES6 module and CommonJS hybrid analysis
- Tree-shaking compatibility validation
- Type-only import optimization (TypeScript)
- Dynamic import and webpack chunk analysis
- Node.js vs browser environment detection

### Java Enterprise Cleanup
```bash
# Java enterprise codebase optimization
/clean-codebase --language=java --analysis=ultrathink --dead-code --duplicates
```
**Java Features:**
- Package structure and classpath optimization
- Annotation processor and reflection usage detection
- Maven/Gradle dependency analysis integration
- Spring Framework and dependency injection patterns
- Serialization and interface implementation tracking

## Safety & Best Practices

### Pre-Cleanup Checklist
- [ ] **Version Control**: Ensure all changes are committed before cleanup
- [ ] **Backup Strategy**: Use `--backup` flag for critical codebases
- [ ] **Test Coverage**: Run full test suite to establish baseline
- [ ] **Build Verification**: Ensure project builds successfully before cleanup
- [ ] **Dependency Analysis**: Review external dependencies and build scripts

### Cleanup Safety Levels
1. **Conservative** (`--analysis=basic --dry-run`): Preview only, minimal risk
2. **Standard** (`--analysis=thorough --backup`): Safe operations with backup
3. **Aggressive** (`--analysis=comprehensive --rollback`): Deep cleanup with rollback
4. **Ultrathink** (`--analysis=ultrathink --backup --rollback`): Maximum intelligence with full safety

### Production Environment Guidelines
```bash
# Production-safe cleanup pipeline
# Step 1: Analysis and planning
/clean-codebase --analysis=ultrathink --dry-run --report

# Step 2: Conservative cleanup
/clean-codebase --imports --backup --rollback --interactive

# Step 3: Validation
/run-all-tests --coverage --auto-fix
/check-code-quality --analysis=comprehensive

# Step 4: Commit if successful
/commit --template=refactor --ai-message --validate
```

## Integration Patterns

### Complete Codebase Modernization Workflow
```bash
# 1. Initial analysis and planning
/think-ultra "comprehensive codebase modernization strategy" --depth=ultra --agents=all

# 2. Safety preparation
/run-all-tests --coverage --report
/check-code-quality --auto-fix --report

# 3. Advanced cleanup execution
/clean-codebase --analysis=ultrathink --imports --dead-code --duplicates --backup --parallel

# 4. Post-cleanup optimization
/optimize --implement --language=auto --category=all
/refactor-clean --patterns=modern --implement

# 5. Comprehensive validation
/run-all-tests --auto-fix --coverage
/double-check "modernization results" --deep-analysis --auto-complete

# 6. Final quality assurance
/generate-tests --coverage=95 --type=all
/commit --ai-message --validate --push
```

### Continuous Integration Integration
```bash
# CI/CD pipeline integration
name: "Automated Codebase Cleanup"
steps:
  - name: "Safe Import Cleanup"
    run: /clean-codebase --imports --language=auto --dry-run --report

  - name: "Duplicate Detection"
    run: /clean-codebase --duplicates --analysis=thorough --report

  - name: "Dead Code Analysis"
    run: /clean-codebase --dead-code --analysis=comprehensive --dry-run
```

This advanced implementation transforms the clean-codebase command into a sophisticated, intelligent cleanup tool that combines multi-agent analysis, ultrathink reasoning, and AST-based precision for safe, comprehensive codebase optimization.