# Workflow Template Guide

## Available Templates

This guide provides detailed information about all pre-built workflow templates included in the framework.

## Template Overview

| Template | Purpose | Steps | Complexity | Est. Duration |
|----------|---------|-------|------------|---------------|
| quality-improvement | Code quality enhancement | 8 | Intermediate | 5-15 min |
| performance-optimization | Performance tuning | 8 | Advanced | 10-30 min |
| refactoring-workflow | Safe code refactoring | 9 | Advanced | 10-25 min |
| documentation-generation | Auto-generate docs | 7 | Basic | 5-10 min |
| ci-cd-setup | CI/CD pipeline setup | 7 | Intermediate | 5-15 min |
| complete-development-cycle | Full dev workflow | 11 | Expert | 20-45 min |
| research-workflow | Scientific computing | 10 | Expert | 15-40 min |
| migration-workflow | Code migration | 13 | Expert | 30-60 min |

---

## 1. Quality Improvement Workflow

**File:** `quality-improvement.yaml`

### Purpose
Complete code quality improvement with automatic fixes, testing, and commit.

### Steps
1. **check_quality** - Analyze code quality
2. **auto_fix_issues** - Auto-fix identified issues
3. **clean_codebase** - Remove dead code and unused imports
4. **refactor_code** - Apply modern refactoring patterns
5. **generate_tests** - Generate comprehensive test suite
6. **run_tests** - Validate changes with tests
7. **quality_check_final** - Final quality verification
8. **commit_changes** - Commit improvements

### Variables
```yaml
target_path: "."              # Target directory
language: "auto"              # Programming language
coverage_target: 90           # Test coverage goal
commit_message: "..."         # Commit message
```

### Usage
```bash
# Basic usage
python workflows/cli.py run quality-improvement

# Custom target
python workflows/cli.py run quality-improvement --var target_path=src/

# Specific language
python workflows/cli.py run quality-improvement --var language=python

# Higher coverage
python workflows/cli.py run quality-improvement --var coverage_target=95
```

### When to Use
- Before code reviews
- After major feature development
- Regular maintenance cycles
- CI/CD quality gates

---

## 2. Performance Optimization Workflow

**File:** `performance-optimization.yaml`

### Purpose
Comprehensive performance optimization with profiling and benchmarking.

### Steps
1. **profile_baseline** - Profile current performance
2. **identify_bottlenecks** - Identify performance issues
3. **optimize_code** - Apply optimizations
4. **parallel_optimizations** - Parallel optimization tasks
   - optimize_algorithms
   - optimize_memory
   - optimize_io
5. **validate_optimizations** - Ensure functionality preserved
6. **benchmark_improvements** - Measure improvements
7. **update_documentation** - Update docs with results
8. **commit_optimizations** - Commit changes

### Variables
```yaml
target_path: "."              # Target directory
language: "auto"              # Programming language
optimization_category: "all"  # Optimization focus
```

### Usage
```bash
# Full optimization
python workflows/cli.py run performance-optimization

# Specific category
python workflows/cli.py run performance-optimization \
  --var optimization_category=algorithm

# Target specific code
python workflows/cli.py run performance-optimization \
  --var target_path=src/core/
```

### When to Use
- Performance issues identified
- Before production releases
- After major algorithm changes
- Periodic performance audits

---

## 3. Refactoring Workflow

**File:** `refactoring-workflow.yaml`

### Purpose
Safe refactoring with backup, validation, and automatic rollback.

### Steps
1. **create_backup** - Create backup commit
2. **run_tests_baseline** - Baseline test results
3. **analyze_complexity** - Analyze code complexity
4. **apply_refactoring** - Apply refactoring patterns
5. **clean_after_refactor** - Clean up code
6. **run_tests_validation** - Validate refactoring
7. **check_performance** - Check for regression
8. **verify_quality** - Verify quality improvements
9. **commit_refactoring** - Commit changes

### Variables
```yaml
target_path: "."              # Target directory
language: "auto"              # Programming language
backup_branch: "..."          # Backup branch name
```

### Usage
```bash
# Safe refactoring
python workflows/cli.py run refactoring-workflow

# Custom backup branch
python workflows/cli.py run refactoring-workflow \
  --var backup_branch=refactor-2024-backup
```

### When to Use
- Code smells detected
- Technical debt reduction
- Modernizing legacy code
- Improving maintainability

---

## 4. Documentation Generation Workflow

**File:** `documentation-generation.yaml`

### Purpose
Automatically generate comprehensive documentation.

### Steps
1. **analyze_code_structure** - Analyze codebase
2. **parallel_doc_generation** - Generate docs in parallel
   - generate_api_docs
   - generate_readme
   - generate_research_docs
3. **explain_complex_code** - Document complex sections
4. **generate_examples** - Create usage examples
5. **update_changelog** - Update changelog
6. **verify_documentation** - Verify quality
7. **commit_documentation** - Commit docs

### Variables
```yaml
target_path: "."              # Target directory
doc_type: "all"               # Documentation type
format: "markdown"            # Output format
```

### Usage
```bash
# Generate all docs
python workflows/cli.py run documentation-generation

# API docs only
python workflows/cli.py run documentation-generation --var doc_type=api

# LaTeX format
python workflows/cli.py run documentation-generation --var format=latex
```

### When to Use
- New project setup
- Before releases
- Documentation updates needed
- API changes

---

## 5. CI/CD Setup Workflow

**File:** `ci-cd-setup.yaml`

### Purpose
Setup complete CI/CD pipeline with testing, security, and monitoring.

### Steps
1. **analyze_project** - Analyze project structure
2. **setup_ci_pipeline** - Setup main pipeline
3. **parallel_ci_config** - Configure components
   - setup_testing
   - setup_security
   - setup_monitoring
4. **generate_test_suite** - Create test suite
5. **validate_ci_config** - Validate configuration
6. **test_ci_locally** - Test pipeline locally
7. **commit_ci_setup** - Commit CI files

### Variables
```yaml
platform: "github"            # CI platform
ci_type: "basic"              # Configuration type
deploy_env: "both"            # Deployment environments
```

### Usage
```bash
# GitHub Actions setup
python workflows/cli.py run ci-cd-setup

# GitLab CI
python workflows/cli.py run ci-cd-setup --var platform=gitlab

# Enterprise setup
python workflows/cli.py run ci-cd-setup --var ci_type=enterprise
```

### When to Use
- New project initialization
- Adding CI/CD to existing project
- Upgrading CI configuration
- Multi-platform CI setup

---

## 6. Complete Development Cycle

**File:** `complete-development-cycle.yaml`

### Purpose
Full development cycle from quality check to commit.

### Steps
1. **initial_quality_check** - Initial assessment
2. **clean_codebase** - Clean up code
3. **optimize_performance** - Optimize code
4. **refactor_code** - Refactor patterns
5. **generate_tests** - Generate tests
6. **run_all_tests** - Run test suite
7. **parallel_quality_checks** - Multiple checks
   - final_quality_check
   - debug_check
   - security_scan
8. **generate_documentation** - Create docs
9. **double_check_work** - Comprehensive verification
10. **create_commit** - Create commit
11. **verify_commit** - Verify integrity

### Variables
```yaml
target_path: "."              # Target directory
language: "auto"              # Programming language
coverage_target: 90           # Test coverage goal
```

### Usage
```bash
# Complete cycle
python workflows/cli.py run complete-development-cycle

# High coverage
python workflows/cli.py run complete-development-cycle \
  --var coverage_target=95
```

### When to Use
- Feature completion
- Release preparation
- Major refactoring
- Quality assurance

---

## 7. Research Workflow

**File:** `research-workflow.yaml`

### Purpose
Scientific computing workflow with reproducible tests and research documentation.

### Steps
1. **analyze_scientific_code** - Analyze code structure
2. **debug_scientific** - Debug scientific issues
3. **optimize_algorithms** - Optimize algorithms
4. **parallel_optimization** - Parallel optimizations
   - memory_optimization
   - io_optimization
   - concurrency_optimization
5. **generate_scientific_tests** - Generate tests
6. **run_reproducible_tests** - Run reproducible tests
7. **benchmark_performance** - Benchmark results
8. **generate_research_docs** - Create research docs
9. **reflection_analysis** - Scientific reflection
10. **commit_research** - Commit research code

### Variables
```yaml
target_path: "."              # Target directory
language: "auto"              # Programming language
test_framework: "auto"        # Test framework
```

### Usage
```bash
# Scientific workflow
python workflows/cli.py run research-workflow

# Python scientific
python workflows/cli.py run research-workflow --var language=python

# Julia scientific
python workflows/cli.py run research-workflow --var language=julia
```

### When to Use
- Scientific computing projects
- Research code preparation
- Publication-ready code
- Reproducible research

---

## 8. Migration Workflow

**File:** `migration-workflow.yaml`

### Purpose
Code migration and modernization with validation.

### Steps
1. **analyze_legacy** - Analyze legacy code
2. **plan_migration** - Plan strategy
3. **backup_codebase** - Create backup
4. **adopt_code** - Analyze and integrate
5. **apply_modern_patterns** - Modernize code
6. **clean_migrated_code** - Clean up
7. **generate_migration_tests** - Generate tests
8. **validate_functionality** - Validate migration
9. **performance_comparison** - Compare performance
10. **optimize_migrated** - Optimize code
11. **update_migration_docs** - Update docs
12. **final_verification** - Final checks
13. **commit_migration** - Commit changes

### Variables
```yaml
target_path: "."              # Target directory
source_language: "auto"       # Source language
target_language: "auto"       # Target language
```

### Usage
```bash
# Language migration
python workflows/cli.py run migration-workflow \
  --var source_language=python \
  --var target_language=julia

# Modernization
python workflows/cli.py run migration-workflow \
  --var source_language=python \
  --var target_language=python
```

### When to Use
- Language migration
- Framework migration
- Modernizing legacy code
- Technology stack updates

---

## Customizing Templates

### Create Custom Workflow from Template

```bash
# Create from template
python workflows/cli.py create my-custom-workflow \
  --template quality-improvement \
  --description "Custom quality workflow for my project"

# Modify the generated file
# custom/my-custom-workflow.yaml
```

### Override Variables

```bash
# Override at runtime
python workflows/cli.py run quality-improvement \
  --var target_path=src/ \
  --var language=python \
  --var coverage_target=95
```

### Extend Templates

Edit the YAML file to add steps:

```yaml
# Add custom step
steps:
  # ... existing steps ...

  - id: custom_step
    command: your-custom-command
    flags: [--custom-flag]
    depends_on: [previous_step]
```

## Template Selection Guide

### By Project Type

**Web Application:**
- quality-improvement
- ci-cd-setup
- complete-development-cycle

**Scientific Computing:**
- research-workflow
- performance-optimization
- documentation-generation

**Legacy Code:**
- migration-workflow
- refactoring-workflow
- quality-improvement

**New Project:**
- ci-cd-setup
- documentation-generation
- quality-improvement

### By Goal

**Improve Quality:**
- quality-improvement
- refactoring-workflow
- complete-development-cycle

**Improve Performance:**
- performance-optimization
- research-workflow

**Setup Infrastructure:**
- ci-cd-setup
- documentation-generation

**Major Changes:**
- migration-workflow
- refactoring-workflow
- complete-development-cycle

## Best Practices

1. **Start with dry-run** to understand workflow behavior
2. **Customize variables** for your project
3. **Add error handling** for critical steps
4. **Test incrementally** before full workflow
5. **Version control** custom workflows
6. **Document modifications** to templates
7. **Share templates** within team

## Template Development

### Creating New Templates

1. **Define purpose and scope**
2. **Identify required steps**
3. **Add dependencies**
4. **Configure error handling**
5. **Add variables for customization**
6. **Test thoroughly**
7. **Document usage**

### Template Checklist

- [ ] Clear name and description
- [ ] Appropriate complexity level
- [ ] Well-defined dependencies
- [ ] Error handling on critical steps
- [ ] Rollback commands where needed
- [ ] Configurable variables
- [ ] Validation passes
- [ ] Tested with dry-run
- [ ] Tested with actual execution
- [ ] Documentation complete

## Support

For template issues:
1. Validate template: `python workflows/cli.py validate template.yaml`
2. Test with dry-run: `python workflows/cli.py run template --dry-run`
3. Check template info: `python workflows/cli.py info template-name`
4. Review validation output for errors