# Plugin Review and Performance Tools

Automated tools for comprehensive plugin review, validation, performance profiling, triggering pattern analysis, and cross-plugin integration in the Claude Code marketplace.

## Overview

This directory contains sixteen automation tools designed to systematically review, validate, profile, test triggering patterns, analyze cross-plugin relationships, and improve plugin quality:

### Review & Validation Tools
1. **plugin-review-script.py** - Comprehensive plugin review automation
2. **metadata-validator.py** - plugin.json metadata validation
3. **doc-checker.py** - Documentation completeness checker

### Performance Profiling Tools
4. **load-profiler.py** - Plugin load time profiling
5. **activation-profiler.py** - Agent activation performance profiling
6. **memory-analyzer.py** - Memory usage analysis
7. **performance-reporter.py** - Aggregated performance reporting

### Triggering Pattern Analysis Tools
8. **test-corpus-generator.py** - Generate diverse test samples for triggering validation
9. **activation-tester.py** - Test plugin activation accuracy (FP/FN rates)
10. **command-analyzer.py** - Analyze command suggestion relevance and timing
11. **skill-validator.py** - Validate skill pattern matching accuracy
12. **triggering-reporter.py** - Generate comprehensive triggering pattern reports

### Cross-Plugin Integration Tools
13. **dependency-mapper.py** - Map cross-plugin dependencies and relationships
14. **terminology-analyzer.py** - Analyze terminology consistency across plugins
15. **workflow-generator.py** - Generate integration workflow documentation
16. **xref-validator.py** - Validate cross-plugin references

These tools support the plugin review and optimization process by automating checklist validation, identifying issues, profiling performance, validating triggering accuracy, and ensuring consistent cross-plugin integration.

## Review & Validation Tools

### 1. Plugin Review Script

**File:** `plugin-review-script.py`

**Purpose:** Automates comprehensive plugin review using a standardized checklist covering all aspects of plugin quality.

**Features:**
- Validates plugin.json structure and completeness
- Checks agent/command/skill file existence
- Validates documentation completeness
- Identifies broken links and missing files
- Generates structured review reports in markdown
- Categorizes issues by severity (critical, high, medium, low)

**Usage:**

```bash
# Basic usage (assumes plugins directory in current working directory)
python3 tools/plugin-review-script.py <plugin-name>

# Specify plugins root directory
python3 tools/plugin-review-script.py <plugin-name> /path/to/plugins

# Example: Review julia-development plugin
python3 tools/plugin-review-script.py julia-development

# Example: Review with custom plugins path
python3 tools/plugin-review-script.py julia-development /Users/username/Projects/MyClaude/plugins
```

**Output:**
- Console output: Formatted markdown report
- File output: `reviews/<plugin-name>.md`
- Exit codes:
  - 0: Success (no critical/high issues)
  - 1: High priority issues found
  - 2: Critical issues found

**Validation Sections:**

1. **plugin.json** - Structure, required fields, semantic versioning
2. **agents** - Agent definitions, documentation files
3. **commands** - Command definitions, documentation files
4. **skills** - Skill definitions, documentation files
5. **README** - Completeness, required sections, code examples
6. **file-structure** - Expected directories and files
7. **cross-references** - Link validation, broken references

### 2. Metadata Validator

**File:** `metadata-validator.py`

**Purpose:** Validates plugin.json metadata against schema requirements with detailed error reporting.

**Features:**
- JSON schema compliance validation
- Required fields presence checking
- Semantic versioning format verification
- Tag and category validation
- Agent/command/skill structure validation
- Type checking and pattern matching

**Usage:**

```bash
# Validate plugin metadata
python3 tools/metadata-validator.py <plugin-path>

# Example: Validate julia-development
python3 tools/metadata-validator.py plugins/julia-development

# Example: Validate quality-engineering
python3 tools/metadata-validator.py plugins/quality-engineering
```

**Output:**
- Console output: Validation report with errors and warnings
- Exit codes:
  - 0: Valid (may have warnings)
  - 1: Invalid (errors found)

### 3. Documentation Checker

**File:** `doc-checker.py`

**Purpose:** Validates plugin documentation for completeness, formatting, and quality.

**Features:**
- README.md required sections verification
- Markdown formatting validation
- Code block syntax checking
- Cross-reference accuracy checking
- Link validation (broken links detection)
- Common issues detection (TODOs, placeholders, etc.)

**Usage:**

```bash
# Check plugin documentation
python3 tools/doc-checker.py <plugin-path>

# Example: Check julia-development docs
python3 tools/doc-checker.py plugins/julia-development

# Example: Check quality-engineering docs
python3 tools/doc-checker.py plugins/quality-engineering
```

**Output:**
- Console output: Documentation check report
- Exit codes:
  - 0: Success (no errors, may have warnings)
  - 1: Errors found

## Performance Profiling Tools

### 4. Load Time Profiler

**File:** `load-profiler.py`

**Purpose:** Measures plugin loading performance to identify bottlenecks and optimize initialization.

**Features:**
- Measures marketplace initialization time
- Profiles individual plugin loading
- Identifies slow configuration parsing
- Tracks dependency loading overhead
- Target: <100ms load time per plugin

**Usage:**

```bash
# Profile a single plugin
python3 tools/load-profiler.py <plugin-name>

# Profile with custom plugins path
python3 tools/load-profiler.py <plugin-name> /path/to/plugins

# Profile all plugins
python3 tools/load-profiler.py --all

# Examples
python3 tools/load-profiler.py julia-development
python3 tools/load-profiler.py unit-testing
python3 tools/load-profiler.py --all
```

**Output:**
- Console output: Formatted load profile report
- Exit codes:
  - 0: Pass (load time < 100ms)
  - 1: Fail (load time >= 100ms)
  - 2: Error

### 5. Agent Activation Profiler

**File:** `activation-profiler.py`

**Purpose:** Measures agent activation performance to identify bottlenecks in triggering logic.

**Features:**
- Measures context analysis time
- Profiles triggering condition evaluation
- Tracks pattern matching performance
- Identifies bottlenecks in activation logic
- Target: <50ms activation time

**Usage:**

```bash
# Profile a single plugin
python3 tools/activation-profiler.py <plugin-name>

# Profile with custom plugins path
python3 tools/activation-profiler.py <plugin-name> /path/to/plugins

# Profile all plugins
python3 tools/activation-profiler.py --all

# Examples
python3 tools/activation-profiler.py julia-development
python3 tools/activation-profiler.py agent-orchestration
python3 tools/activation-profiler.py --all
```

**Output:**
- Console output: Formatted activation profile report
- Exit codes:
  - 0: Pass (activation time < 50ms)
  - 1: Fail (activation time >= 50ms)
  - 2: Error

### 6. Memory Usage Analyzer

**File:** `memory-analyzer.py`

**Purpose:** Measures plugin memory consumption to identify memory leaks and inefficiencies.

**Features:**
- Measures baseline memory consumption
- Tracks memory during typical operations
- Identifies memory leaks
- Profiles data structure efficiency

**Usage:**

```bash
# Analyze a single plugin
python3 tools/memory-analyzer.py <plugin-name>

# Analyze with custom plugins path
python3 tools/memory-analyzer.py <plugin-name> /path/to/plugins

# Analyze all plugins
python3 tools/memory-analyzer.py --all

# Examples
python3 tools/memory-analyzer.py julia-development
python3 tools/memory-analyzer.py unit-testing
python3 tools/memory-analyzer.py --all
```

**Output:**
- Console output: Formatted memory profile report
- Exit codes:
  - 0: Pass (memory < 10MB)
  - 1: Fail (memory >= 10MB)
  - 2: Error

### 7. Performance Report Generator

**File:** `performance-reporter.py`

**Purpose:** Aggregates performance metrics across all profiling tools and generates comprehensive reports.

**Features:**
- Aggregates metrics across all plugins
- Generates before/after comparison reports
- Visualizes performance trends
- Exports results to CSV/JSON

**Usage:**

```bash
# Generate report for a single plugin
python3 tools/performance-reporter.py <plugin-name>

# Generate report for all plugins
python3 tools/performance-reporter.py --all

# Compare two performance reports
python3 tools/performance-reporter.py --compare before.json after.json

# Export to CSV
python3 tools/performance-reporter.py --export csv output.csv

# Export to JSON
python3 tools/performance-reporter.py --export json output.json

# Examples
python3 tools/performance-reporter.py julia-development
python3 tools/performance-reporter.py --all
python3 tools/performance-reporter.py --compare baseline.json optimized.json
```

**Output:**
- Console output: Formatted performance report
- File output: JSON/CSV export files
- Exit codes:
  - 0: Success
  - 1: Warnings or failures
  - 2: Errors

## Triggering Pattern Analysis Tools

### 8. Test Corpus Generator

**File:** `test-corpus-generator.py`

**Purpose:** Generates diverse sample projects for testing plugin triggering patterns.

**Features:**
- Creates sample projects for each plugin category
- Generates edge case test files
- Builds unrelated codebase samples (negative tests)
- Includes multi-language projects
- Produces comprehensive test coverage

**Usage:**

```bash
# Generate all test samples
python3 tools/test-corpus-generator.py

# Specify output directory
python3 tools/test-corpus-generator.py --output-dir custom-test-corpus

# Generate specific categories only
python3 tools/test-corpus-generator.py --categories scientific-computing development

# Examples
python3 tools/test-corpus-generator.py
python3 tools/test-corpus-generator.py --output-dir test-corpus
python3 tools/test-corpus-generator.py --categories scientific-computing devops
```

**Output:**
- Directory structure: `test-corpus/<sample-name>/`
- Each sample contains:
  - `metadata.json` - Sample information and expected behavior
  - Source files representing the project type
- `test-corpus/index.json` - Index of all samples
- `test-corpus/README.md` - Documentation of test corpus

**Test Sample Categories:**
- **Scientific Computing:** Julia, Python/JAX, HPC, molecular simulation, deep learning
- **Development:** TypeScript/React, Rust CLI, FastAPI backend
- **DevOps:** GitHub Actions CI/CD, pytest test suites
- **Edge Cases:** Empty projects, frontend-only, config-only
- **Multi-Language:** Python+C++, Julia+Python interop

### 9. Activation Accuracy Tester

**File:** `activation-tester.py`

**Purpose:** Tests plugin activation accuracy against test corpus, measuring false positive/negative rates.

**Features:**
- Tests file extension pattern matching
- Validates directory pattern recognition
- Tests content pattern accuracy
- Measures false positive rate (target: <5%)
- Measures false negative rate (target: <5%)
- Calculates precision, recall, and F1 score

**Usage:**

```bash
# Test all plugins
python3 tools/activation-tester.py

# Specify custom paths
python3 tools/activation-tester.py --plugins-dir /path/to/plugins --corpus-dir /path/to/test-corpus

# Test specific plugin only
python3 tools/activation-tester.py --plugin julia-development

# Specify output file
python3 tools/activation-tester.py --output reports/activation-accuracy.md

# Examples
python3 tools/activation-tester.py
python3 tools/activation-tester.py --plugin python-development
python3 tools/activation-tester.py --output custom-report.md
```

**Output:**
- Console output: Formatted accuracy report
- File output: `reports/activation-accuracy.md`
- Exit codes:
  - 0: Pass (FP < 5%, FN < 5%)
  - 1: Fail (FP >= 5% or FN >= 5%)

**Metrics Reported:**
- Overall accuracy
- Precision (TP / (TP + FP))
- Recall (TP / (TP + FN))
- F1 Score
- False Positive Rate
- False Negative Rate
- Confusion matrix
- Per-plugin performance

### 10. Command Suggestion Analyzer

**File:** `command-analyzer.py`

**Purpose:** Analyzes command suggestion relevance, timing, and priority ranking accuracy.

**Features:**
- Tests command relevance in different contexts
- Validates suggestion timing appropriateness
- Analyzes priority ranking accuracy
- Evaluates context-aware suggestions
- Identifies commands with poor performance

**Usage:**

```bash
# Analyze all commands
python3 tools/command-analyzer.py

# Specify custom paths
python3 tools/command-analyzer.py --plugins-dir /path/to/plugins --corpus-dir /path/to/test-corpus

# Analyze specific plugin commands
python3 tools/command-analyzer.py --plugin julia-development

# Specify output file
python3 tools/command-analyzer.py --output reports/command-analysis.md

# Examples
python3 tools/command-analyzer.py
python3 tools/command-analyzer.py --plugin python-development
python3 tools/command-analyzer.py --output custom-report.md
```

**Output:**
- Console output: Formatted analysis report
- File output: `reports/command-analysis.md`
- Exit codes:
  - 0: Pass (relevance > 80%, timing > 85%)
  - 1: Fail (below targets)

**Metrics Reported:**
- Relevance accuracy (target: >80%)
- Timing accuracy (target: >85%)
- Priority accuracy (target: >90%)
- Per-command performance
- Top performing commands
- Commands needing improvement

### 11. Skill Application Validator

**File:** `skill-validator.py`

**Purpose:** Validates skill pattern matching and checks for over/under-triggering issues.

**Features:**
- Tests skill pattern matching
- Validates skill recommendations
- Checks for over-triggering issues
- Detects under-triggering problems
- Analyzes skill application accuracy

**Usage:**

```bash
# Validate all skills
python3 tools/skill-validator.py

# Specify custom paths
python3 tools/skill-validator.py --plugins-dir /path/to/plugins --corpus-dir /path/to/test-corpus

# Validate specific plugin skills
python3 tools/skill-validator.py --plugin julia-development

# Specify output file
python3 tools/skill-validator.py --output reports/skill-validation.md

# Examples
python3 tools/skill-validator.py
python3 tools/skill-validator.py --plugin python-development
python3 tools/skill-validator.py --output custom-report.md
```

**Output:**
- Console output: Formatted validation report
- File output: `reports/skill-validation.md`
- Exit codes:
  - 0: Pass (over-trigger < 10%, under-trigger < 10%)
  - 1: Fail (above thresholds)

**Metrics Reported:**
- Overall accuracy (target: >90%)
- Precision (target: >85%)
- Recall (target: >85%)
- Over-trigger rate (target: <10%)
- Under-trigger rate (target: <10%)
- Per-skill performance
- Skills with triggering issues

### 12. Triggering Pattern Reporter

**File:** `triggering-reporter.py`

**Purpose:** Aggregates all triggering metrics and generates comprehensive reports with actionable recommendations.

**Features:**
- Aggregates activation, command, and skill metrics
- Identifies problematic patterns
- Suggests pattern improvements
- Prioritizes recommendations by severity
- Calculates overall triggering quality score

**Usage:**

```bash
# Generate comprehensive report
python3 tools/triggering-reporter.py

# Specify custom reports directory
python3 tools/triggering-reporter.py --reports-dir /path/to/reports

# Specify output file
python3 tools/triggering-reporter.py --output comprehensive-report.md

# Examples
python3 tools/triggering-reporter.py
python3 tools/triggering-reporter.py --reports-dir reports
python3 tools/triggering-reporter.py --output final-report.md
```

**Output:**
- Console output: Comprehensive triggering report
- File output: `reports/triggering-comprehensive-report.md`
- Exit codes:
  - 0: Pass (overall score >= 80)
  - 1: Fail (overall score < 80)

**Report Sections:**
- Executive summary with overall quality score
- Key findings across all categories
- Detailed metrics breakdown
- Issues identified by severity
- Prioritized recommendations
- Pattern improvement suggestions
- Next steps and action items

## Cross-Plugin Integration Tools

### 13. Plugin Dependency Mapper

**File:** `dependency-mapper.py`

**Purpose:** Analyzes and maps dependencies and relationships between plugins.

**Features:**
- Parses all plugin.json files
- Extracts agent, command, and skill references
- Builds dependency graph (forward and reverse)
- Identifies integration patterns
- Generates visual dependency maps (Mermaid diagrams)
- Detects isolated plugins

**Usage:**

```bash
# Analyze all plugins
python3 tools/dependency-mapper.py

# Specify custom paths
python3 tools/dependency-mapper.py --plugins-dir /path/to/plugins

# Specify output file
python3 tools/dependency-mapper.py --output reports/dependency-map.md

# Export as JSON
python3 tools/dependency-mapper.py --export-json reports/dependency-graph.json

# Examples
python3 tools/dependency-mapper.py --plugins-dir plugins
python3 tools/dependency-mapper.py --output custom-map.md
python3 tools/dependency-mapper.py --export-json graph.json
```

**Output:**
- Console output: Dependency analysis summary
- File output: `reports/dependency-map.md`
- Optional JSON export with complete graph data
- Exit codes:
  - 0: Success

**Report Sections:**
- Summary statistics
- Plugins by category
- Dependency matrix (who depends on whom)
- Most referenced plugins
- Integration patterns
- Reference type distribution
- Detailed reference listing
- Isolated plugins
- Mermaid dependency graph visualization
- Recommendations for improving integration

### 14. Terminology Consistency Analyzer

**File:** `terminology-analyzer.py`

**Purpose:** Analyzes terminology usage and consistency across all plugins.

**Features:**
- Extracts technical terms from all plugins
- Identifies terminology variations (spelling, hyphenation, capitalization)
- Maps synonyms and inconsistencies
- Suggests standardization
- Detects British vs American spelling
- Validates framework name capitalization

**Usage:**

```bash
# Analyze all plugins
python3 tools/terminology-analyzer.py

# Specify custom paths
python3 tools/terminology-analyzer.py --plugins-dir /path/to/plugins

# Specify output file
python3 tools/terminology-analyzer.py --output reports/terminology-analysis.md

# Export glossary as JSON
python3 tools/terminology-analyzer.py --export-glossary reports/glossary.json

# Examples
python3 tools/terminology-analyzer.py --plugins-dir plugins
python3 tools/terminology-analyzer.py --output custom-analysis.md
python3 tools/terminology-analyzer.py --export-glossary glossary.json
```

**Output:**
- Console output: Terminology analysis summary
- File output: `reports/terminology-analysis.md`
- Optional glossary export as JSON
- Exit codes:
  - 0: Success

**Report Sections:**
- Summary statistics
- Most common technical terms
- Terminology variations found
- Internal inconsistencies (same plugin using different terms)
- Recommendations for standardization
- Proposed standardization guide:
  - Framework and library names
  - Compound terms (hyphenation)
  - Acronyms
  - British vs American spelling

### 15. Integration Workflow Generator

**File:** `workflow-generator.py`

**Purpose:** Generates integration workflow documentation for common multi-plugin scenarios.

**Features:**
- Identifies common plugin combinations
- Generates workflow documentation templates
- Creates integration examples
- Documents multi-plugin use cases
- Provides predefined workflow templates

**Predefined Workflows:**
- Scientific Computing Full-Stack Workflow
- Julia SciML + Bayesian Analysis
- Machine Learning Development Pipeline
- Full-Stack Web Application
- Molecular Dynamics Simulation
- Code Quality Assurance
- JAX Scientific Computing
- CI/CD Testing Pipeline

**Usage:**

```bash
# Generate all workflows
python3 tools/workflow-generator.py

# Specify custom paths
python3 tools/workflow-generator.py --plugins-dir /path/to/plugins

# Specify output file
python3 tools/workflow-generator.py --output reports/integration-workflows.md

# Export as JSON
python3 tools/workflow-generator.py --export-json reports/workflows.json

# Generate custom workflow template
python3 tools/workflow-generator.py --generate-template plugin1 plugin2 plugin3

# Examples
python3 tools/workflow-generator.py --plugins-dir plugins
python3 tools/workflow-generator.py --output custom-workflows.md
python3 tools/workflow-generator.py --generate-template julia-development hpc-computing
```

**Output:**
- Console output: Workflow generation summary
- File output: `reports/integration-workflows.md`
- Optional JSON export with structured workflow data
- Exit codes:
  - 0: Success

**Report Sections:**
- Summary by category
- Table of contents
- Workflows organized by category:
  - Scientific Computing Workflows
  - Development Workflows
  - DevOps & Quality Workflows
- For each workflow:
  - Description and purpose
  - Plugins involved
  - Step-by-step workflow
  - Example usage

### 16. Cross-Reference Validator

**File:** `xref-validator.py`

**Purpose:** Validates all cross-plugin references to ensure accuracy and completeness.

**Features:**
- Checks all cross-plugin references in documentation
- Validates agent/command/skill mentions
- Identifies broken references
- Detects invalid plugin names
- Validates markdown links
- Generates validation reports with error details

**Usage:**

```bash
# Validate all references
python3 tools/xref-validator.py

# Specify custom paths
python3 tools/xref-validator.py --plugins-dir /path/to/plugins

# Specify output file
python3 tools/xref-validator.py --output reports/xref-validation.md

# Export as JSON
python3 tools/xref-validator.py --export-json reports/xref-results.json

# Examples
python3 tools/xref-validator.py --plugins-dir plugins
python3 tools/xref-validator.py --output custom-validation.md
python3 tools/xref-validator.py --export-json results.json
```

**Output:**
- Console output: Validation summary
- File output: `reports/xref-validation.md`
- Optional JSON export with detailed results
- Exit codes:
  - 0: Success (all references valid)
  - 1: Broken references found

**Report Sections:**
- Summary statistics
- Status indicator (pass/fail)
- Reference type distribution
- Broken references (by plugin)
- Valid references summary
- Plugin index
- Recommendations and best practices

## Workflow Integration

### Complete Plugin Review Workflow

```bash
#!/bin/bash

# Review a plugin comprehensively
PLUGIN_NAME="julia-development"

echo "=== Step 1: Plugin Review ==="
python3 tools/plugin-review-script.py $PLUGIN_NAME

echo ""
echo "=== Step 2: Metadata Validation ==="
python3 tools/metadata-validator.py plugins/$PLUGIN_NAME

echo ""
echo "=== Step 3: Documentation Check ==="
python3 tools/doc-checker.py plugins/$PLUGIN_NAME

echo ""
echo "=== Step 4: Load Time Profiling ==="
python3 tools/load-profiler.py $PLUGIN_NAME

echo ""
echo "=== Step 5: Activation Profiling ==="
python3 tools/activation-profiler.py $PLUGIN_NAME

echo ""
echo "=== Step 6: Memory Analysis ==="
python3 tools/memory-analyzer.py $PLUGIN_NAME

echo ""
echo "=== Step 7: Performance Report ==="
python3 tools/performance-reporter.py $PLUGIN_NAME
```

### Complete Triggering Pattern Analysis Workflow

```bash
#!/bin/bash

echo "=== Step 1: Generate Test Corpus ==="
python3 tools/test-corpus-generator.py --output-dir test-corpus

echo ""
echo "=== Step 2: Test Activation Accuracy ==="
python3 tools/activation-tester.py --corpus-dir test-corpus

echo ""
echo "=== Step 3: Analyze Command Suggestions ==="
python3 tools/command-analyzer.py --corpus-dir test-corpus

echo ""
echo "=== Step 4: Validate Skill Application ==="
python3 tools/skill-validator.py --corpus-dir test-corpus

echo ""
echo "=== Step 5: Generate Comprehensive Report ==="
python3 tools/triggering-reporter.py --reports-dir reports

echo ""
echo "✓ Triggering pattern analysis complete!"
```

### Complete Cross-Plugin Integration Analysis Workflow

```bash
#!/bin/bash

echo "=== Step 1: Map Plugin Dependencies ==="
python3 tools/dependency-mapper.py --plugins-dir plugins --output reports/dependency-map.md

echo ""
echo "=== Step 2: Analyze Terminology Consistency ==="
python3 tools/terminology-analyzer.py --plugins-dir plugins --output reports/terminology-analysis.md

echo ""
echo "=== Step 3: Generate Integration Workflows ==="
python3 tools/workflow-generator.py --plugins-dir plugins --output reports/integration-workflows.md

echo ""
echo "=== Step 4: Validate Cross-References ==="
python3 tools/xref-validator.py --plugins-dir plugins --output reports/xref-validation.md

echo ""
echo "✓ Cross-plugin integration analysis complete!"
```

### Batch Review Multiple Plugins

```bash
#!/bin/bash

# Review all plugins in a category
PLUGINS=(
    "julia-development"
    "jax-implementation"
    "hpc-computing"
)

for plugin in "${PLUGINS[@]}"; do
    echo "===================================="
    echo "Reviewing: $plugin"
    echo "===================================="

    python3 tools/plugin-review-script.py $plugin
    python3 tools/metadata-validator.py plugins/$plugin
    python3 tools/doc-checker.py plugins/$plugin
    python3 tools/load-profiler.py $plugin
    python3 tools/activation-profiler.py $plugin
    python3 tools/memory-analyzer.py $plugin

    echo ""
done

# Generate aggregate performance report
python3 tools/performance-reporter.py --all
```

### Performance Comparison Workflow

```bash
#!/bin/bash

# Baseline performance before optimization
python3 tools/performance-reporter.py --export json baseline.json

# ... perform optimizations ...

# Measure performance after optimization
python3 tools/performance-reporter.py --export json optimized.json

# Generate comparison report
python3 tools/performance-reporter.py --compare baseline.json optimized.json
```

### CI/CD Integration

```bash
#!/bin/bash

# Run validation in CI/CD pipeline
PLUGIN_NAME=$1

# Exit on any failure
set -e

echo "Validating plugin: $PLUGIN_NAME"

# Run metadata validation (fails on errors)
python3 tools/metadata-validator.py plugins/$PLUGIN_NAME

# Run documentation check (fails on errors)
python3 tools/doc-checker.py plugins/$PLUGIN_NAME

# Run comprehensive review (fails on critical/high issues)
python3 tools/plugin-review-script.py $PLUGIN_NAME

# Run performance profiling (fails if targets exceeded)
python3 tools/load-profiler.py $PLUGIN_NAME
python3 tools/activation-profiler.py $PLUGIN_NAME
python3 tools/memory-analyzer.py $PLUGIN_NAME

# Run triggering tests (fails if FP/FN rates too high)
python3 tools/activation-tester.py --plugin $PLUGIN_NAME

# Run cross-reference validation
python3 tools/xref-validator.py

echo "✅ All validations passed for $PLUGIN_NAME"
```

## Output Format

All tools generate markdown-formatted reports with:

- **Summary section** - Issue counts, statistics
- **Status indicator** - Visual status (✅/⚠️/❌)
- **Issue listings** - Grouped by section/file with severity
- **Suggestions** - Actionable recommendations for fixes
- **Overall assessment** - Final verdict and next steps

**Severity Levels:**

- 🔴 **CRITICAL** - Broken functionality, security issues, invalid JSON
- 🟠 **HIGH** - Missing required files, incomplete documentation
- 🟡 **MEDIUM** - Missing recommended content, minor issues
- 🔵 **LOW** - Optional improvements, style suggestions
- ⚠️  **WARNING** - Recommendations for improvement
- ℹ️  **INFO** - Informational notices
- ✅ **SUCCESS** - Validation passed

## Testing Results

All sixteen tools have been tested on sample plugins:

### Test Coverage

| Plugin | Review | Metadata | Docs | Load | Activation | Memory | Report | Triggering | Integration |
|--------|--------|----------|------|------|------------|--------|--------|------------|-------------|
| julia-development | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| unit-testing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| agent-orchestration | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| javascript-typescript | ✅ | ✅ | ⚠️ | - | - | - | - | ✅ | ✅ |
| quality-engineering | ✅ | ✅ | ✅ | - | - | - | - | ✅ | ✅ |

### Performance Test Results

**julia-development (complex plugin):**
- Load Time: 0.89ms ✅ (Target: <100ms)
- Activation Time: 0.15ms ✅ (Target: <50ms)
- Memory Usage: 119.99KB ✅ (0.12MB)
- Status: All targets met

**unit-testing (simple plugin):**
- Load Time: 0.78ms ✅ (Target: <100ms)
- Activation Time: 0.10ms ✅ (Target: <50ms)
- Memory Usage: 240.93KB ✅ (0.24MB)
- Status: All targets met

**agent-orchestration (meta plugin):**
- Load Time: 1.45ms ✅ (Target: <100ms)
- Activation Time: 0.12ms ✅ (Target: <50ms)
- Memory Usage: 81.92KB ✅ (0.08MB)
- Status: All targets met

### Cross-Plugin Integration Test Results

**Dependency Mapping:**
- Plugins analyzed: 22
- Cross-references found: 29
- Integration patterns identified: 3 categories
- Status: ✅ Complete

**Terminology Analysis:**
- Total term occurrences: 6,604
- Unique normalized terms: 292
- Variations found: 40
- Inconsistencies detected: 254
- Status: ✅ Analysis complete

**Workflow Generation:**
- Workflows created: 5
- Categories covered: 3 (scientific-computing, development, quality)
- Status: ✅ Documentation generated

**Cross-Reference Validation:**
- Total references checked: 28
- Valid references: 28 (100%)
- Broken references: 0
- Status: ✅ All references valid

### Test Corpus Statistics

- **Total Samples:** 16
- **Categories:** 5 (scientific-computing, development, devops, edge-case, multi-language)
- **Edge Cases:** 4
- **Negative Tests:** 2
- **Multi-Language:** 2

## Technical Details

### Requirements

- Python 3.12 or higher (per project requirements)
- No external dependencies (uses standard library only)
- Works on Linux, macOS, and Windows

### Architecture

All tools follow a similar pattern:

1. **Data Models** - Using Python dataclasses for structured data
2. **Validation Logic** - Modular validation functions by section
3. **Reporting** - Markdown generation with structured output
4. **Error Handling** - Graceful degradation, clear error messages
5. **Exit Codes** - Standard exit codes for CI/CD integration

### Code Organization

```
tools/
├── plugin-review-script.py    # Main review automation (650+ lines)
├── metadata-validator.py      # Metadata validation (550+ lines)
├── doc-checker.py             # Documentation checking (600+ lines)
├── load-profiler.py           # Load time profiling (550+ lines)
├── activation-profiler.py     # Activation profiling (600+ lines)
├── memory-analyzer.py         # Memory analysis (550+ lines)
├── performance-reporter.py    # Performance reporting (650+ lines)
├── test-corpus-generator.py   # Test corpus generation (800+ lines)
├── activation-tester.py       # Activation testing (750+ lines)
├── command-analyzer.py        # Command analysis (700+ lines)
├── skill-validator.py         # Skill validation (750+ lines)
├── triggering-reporter.py     # Comprehensive reporting (650+ lines)
├── dependency-mapper.py       # Dependency mapping (750+ lines)
├── terminology-analyzer.py    # Terminology analysis (800+ lines)
├── workflow-generator.py      # Workflow generation (700+ lines)
├── xref-validator.py          # Cross-reference validation (700+ lines)
└── README.md                  # This file
```

### Performance

**Review & Validation Tools:**
- **plugin-review-script.py**: ~1-2 seconds per plugin
- **metadata-validator.py**: <1 second per plugin
- **doc-checker.py**: ~1-2 seconds per plugin

**Performance Profiling Tools:**
- **load-profiler.py**: <100ms per plugin
- **activation-profiler.py**: <100ms per plugin
- **memory-analyzer.py**: <100ms per plugin
- **performance-reporter.py**: ~1 second for aggregation

**Triggering Pattern Tools:**
- **test-corpus-generator.py**: ~2-3 seconds for all samples
- **activation-tester.py**: ~5-10 seconds for all plugins
- **command-analyzer.py**: ~10-15 seconds for all commands
- **skill-validator.py**: ~15-20 seconds for all skills
- **triggering-reporter.py**: <1 second for report generation

**Cross-Plugin Integration Tools:**
- **dependency-mapper.py**: ~2-3 seconds for all plugins
- **terminology-analyzer.py**: ~5-10 seconds for all plugins
- **workflow-generator.py**: ~1-2 seconds for all workflows
- **xref-validator.py**: ~3-5 seconds for all references

Batch processing 31 plugins: ~120-150 seconds total for all tools

## Performance Targets

**Load Time:**
- Target: <100ms per plugin
- Warning: 75-100ms
- Pass: <75ms

**Activation Time:**
- Target: <50ms per plugin
- Warning: 35-50ms
- Pass: <35ms

**Memory Usage:**
- Fail: >10MB per plugin
- Warning: 5-10MB
- Pass: <5MB

**Triggering Accuracy:**
- False Positive Rate: <5% (warning: 5-10%)
- False Negative Rate: <5% (warning: 5-10%)
- Command Relevance: >80% (warning: 70-80%)
- Skill Accuracy: >90% (warning: 80-90%)

**Cross-Plugin Integration:**
- Cross-reference validity: 100% (warning: >95%)
- Terminology consistency: >90% (warning: 80-90%)
- Integration documentation: All major workflows covered

## Future Enhancements

Potential improvements for future versions:

1. **JSON Output** - Add JSON output format for programmatic consumption
2. **HTML Reports** - Generate interactive HTML reports
3. **Diff Reports** - Compare before/after review results
4. **Auto-fix** - Automatically fix simple issues (formatting, etc.)
5. **Custom Rules** - Support custom validation rules via config file
6. **Parallel Processing** - Review multiple plugins in parallel
7. **Integration Tests** - Test cross-plugin integration patterns
8. **Metrics Tracking** - Track quality metrics over time
9. **Real-time Profiling** - Profile actual plugin loading in Claude Code
10. **Flame Graphs** - Generate flame graphs for performance visualization
11. **Machine Learning** - Use ML to improve triggering pattern detection
12. **A/B Testing** - Test different triggering strategies
13. **User Feedback** - Incorporate user feedback on triggering accuracy
14. **Continuous Monitoring** - Real-time monitoring of triggering patterns
15. **Dependency Visualization** - Interactive dependency graph visualization
16. **Workflow Automation** - Automate common multi-plugin workflows

## Contributing

When adding new validation rules or features:

1. Update the corresponding tool's validation logic
2. Add test cases for new rules
3. Update this README with new features
4. Ensure backward compatibility
5. Test on sample plugins before deployment

## License

MIT License - Same as parent project

## Author

DevOps Engineer / Quality Engineer / Performance Engineer / ML Engineer / Technical Writer / Systems Architect
Scientific Computing Workflows Team
