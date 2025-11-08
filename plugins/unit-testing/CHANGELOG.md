# Changelog

All notable changes to the Unit Testing plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v1.0.2.html).

---

## [1.0.3] - 2025-01-08

### Command Optimization with Execution Modes

This release optimizes both unit-testing commands with execution modes, external documentation, and enhanced command descriptions following the proven optimization pattern from quality-engineering and systems-programming v1.0.3.

#### Command Enhancements

**/run-all-tests Command Optimization:**
- **Before**: 1,469 lines (single comprehensive workflow)
- **After**: 492 lines with YAML frontmatter
- **Reduction**: 66.5% (977 lines externalized)

**New Execution Modes:**

| Mode | Duration | Agents | Scope |
|------|----------|--------|-------|
| **Quick** | 30min-1h | test-automator (1 agent) | Single test file, max 3 iterations, basic error analysis |
| **Standard** | 2-4h | test-automator + debugger (2 agents) | Full test suite, max 10 iterations, AI-assisted RCA, coverage >80% |
| **Enterprise** | 1-2d | test-automator + debugger + code-quality (3 agents) | Entire codebase, unlimited iterations, >90% coverage, mutation testing |

**/test-generate Command Optimization:**
- **Before**: 643 lines (embedded examples and patterns)
- **After**: 552 lines with YAML frontmatter
- **Reduction**: 14.2% (91 lines externalized)

**New Execution Modes:**

| Mode | Duration | Agents | Output |
|------|----------|--------|--------|
| **Quick** | 30min-1h | test-automator (1 agent) | Single module, ~50-100 test cases, unit tests only |
| **Standard** | 2-4h | test-automator + hpc-numerical-coordinator (2 agents) | Package/feature, ~200-500 test cases, unit + integration + property-based |
| **Enterprise** | 1-2d | test-automator + hpc-numerical-coordinator + jax-pro (3 agents) | Entire project, ~1,000+ test cases, full test suite + mutation + docs |

**External Documentation** (8 files - 5,858 lines):

**For /run-all-tests** (4 files - 2,949 lines):
- `framework-detection-guide.md` (700 lines) - Auto-detection for Jest, Vitest, pytest, cargo, go test, Maven, Gradle with configuration patterns
- `debugging-strategies.md` (800 lines) - AI-driven RCA, log correlation, flaky test detection, common failure patterns
- `test-execution-workflows.md` (700 lines) - Iterative patterns, parallel strategies, CI/CD integration, performance optimization
- `multi-language-testing.md` (749 lines) - Cross-language patterns, monorepo strategies, framework comparison matrix

**For /test-generate** (4 files - 2,909 lines):
- `test-generation-patterns.md` (800 lines) - AST parsing algorithms, test case generation, edge case identification, mocking strategies
- `scientific-testing-guide.md` (900 lines) - Numerical correctness validation, tolerance-based assertions, JAX gradient verification, PyTorch autograd
- `property-based-testing.md` (700 lines) - Hypothesis patterns, QuickCheck equivalents, shrinking, stateful testing
- `coverage-analysis-guide.md` (509 lines) - Coverage metrics (line/branch/mutation), gap identification, prioritization, pytest-cov/istanbul integration

### Added

- YAML frontmatter in both command files with execution mode definitions
- Interactive mode selection via `AskUserQuestion` for better UX
- 8 comprehensive external documentation files (~5,858 lines total)
- Version fields (1.0.3) to both commands, both agents, and skill for consistency tracking
- Cross-references to external documentation throughout command files
- Enhanced command descriptions with execution modes in plugin.json
- Registered previously unregistered `e2e-testing-patterns` skill in plugin.json
- Scientific computing keywords: scientific-computing, property-based-testing, numerical-validation

### Changed

- Command file structure from linear to execution-mode-based
- Testing content from embedded to externalized documentation
- Agent coordination from implicit to explicit mode-based assignment
- plugin.json version from 1.0.1 to 1.0.3
- Plugin description to highlight v1.0.3 optimizations

### Fixed

- **Critical**: Registered `e2e-testing-patterns` skill that was in skills/ directory but missing from plugin.json
- Version consistency across plugin metadata (all now 1.0.3)

### Improved

- **Token Efficiency**: 50.6% combined command file reduction (2,112 → 1,044 lines)
- **Flexibility**: 3 execution modes per command for different testing complexities
- **Documentation**: Comprehensive external guides (5,858 lines total)
- **Discoverability**: Enhanced command descriptions with execution modes and durations
- **Maintainability**: Separation of concerns (command logic vs. reference documentation)
- **Usability**: Clear execution paths with estimated time commitments
- **Scientific Computing**: JAX gradient verification, NumPy numerical validation, property-based testing
- **Multi-Language Support**: Python, JavaScript/TypeScript, Rust, Go, Java, Julia with framework-specific patterns

### Migration Guide

No breaking changes - existing usage patterns remain compatible. New execution mode selection provides enhanced flexibility:

```bash
# Old usage (still works, defaults to standard mode)
/run-all-tests --fix

# New usage with mode selection
/run-all-tests --fix
# → Prompted to select: Quick / Standard / Enterprise

# test-generate same pattern
/test-generate src/
# → Prompted to select execution mode
```

### Metrics Summary

**Command Optimization**:
- /run-all-tests: 1,469 → 492 lines (66.5% reduction)
- /test-generate: 643 → 552 lines (14.2% reduction)
- **Total**: 2,112 → 1,044 lines (50.6% reduction)

**External Documentation**:
- Total files: 8
- Total lines: 5,858
- Comprehensive coverage of all testing scenarios

**Metadata Enhancements**:
- Commands with version: 2/2 (100%)
- Agents with version: 2/2 (100%)
- Skills with version: 1/1 (100%)
- Skills registered: 1/1 (100%) - fixed e2e-testing-patterns

---

## [1.0.1] - Previous Release

Initial release with comprehensive testing patterns.
