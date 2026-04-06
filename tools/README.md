# MyClaude Plugin Tools

Automated tools for plugin validation, performance profiling, triggering pattern analysis, and ecosystem maintenance. All tools use Python standard library only — no external dependencies.

## Quick Reference

```bash
# Validate all plugins
make validate

# Run all tests (60 tests)
uv run pytest tools/tests/ -v

# Check skill context budget
python3 tools/validation/context_budget_checker.py

# Validate a single plugin
python3 tools/validation/metadata_validator.py plugins/agent-core
```

## Directory Structure

```
tools/
├── common/                # Shared utilities
│   ├── loader.py          # Plugin loading and parsing
│   ├── models.py          # Data models (dataclasses)
│   ├── reporter.py        # Markdown report generation
│   └── timer.py           # Performance timing utilities
├── validation/            # Plugin validation tools
│   ├── metadata_validator.py      # plugin.json schema validation
│   ├── context_budget_checker.py  # Skill size budget enforcement (2% limit)
│   ├── skill_validator.py         # Skill triggering pattern validation
│   ├── xref_validator.py          # Cross-plugin reference validation
│   ├── doc_checker.py             # Documentation completeness checker
│   └── plugin_review_script.py    # Full automated plugin review
├── profiling/             # Performance and triggering profiling
│   ├── load_profiler.py           # Plugin load time measurement
│   ├── activation_profiler.py     # Agent activation performance
│   ├── activation_tester.py       # Activation accuracy (FP/FN rates)
│   ├── command_analyzer.py        # Command suggestion analysis
│   ├── memory_analyzer.py         # Memory usage analysis
│   ├── performance_reporter.py    # Aggregated performance reports
│   └── triggering_reporter.py     # Comprehensive triggering reports
├── maintenance/           # Ecosystem maintenance
│   ├── analyze_ecosystem.py       # Skill/agent ecosystem metrics
│   └── enable_all_plugins.py      # Enable all plugins in settings
├── tests/                 # Pytest suite (60 tests)
│   ├── test_agent_core_integrity.py
│   ├── test_dev_suite_integrity.py
│   ├── test_science_suite_integrity.py
│   ├── test_science_suite_functionality.py
│   ├── test_build_automation.py
│   ├── test_category_pages.py
│   ├── test_content_extraction.py
│   ├── test_doc_checker.py
│   ├── test_plugin_review.py
│   ├── test_refactor_validator.py
│   ├── test_skill_validator.py
│   ├── test_supplementary_docs.py
│   └── test_xref_validator.py
└── README.md
```

## Validation Tools

### metadata_validator.py

Validates `plugin.json` metadata against schema requirements.

```bash
python3 tools/validation/metadata_validator.py plugins/<suite>
```

Checks: JSON schema compliance, required fields, semantic versioning, agent/command/skill structure, file-path references.

### context_budget_checker.py

Ensures all skills fit within the 2% context window budget.

```bash
python3 tools/validation/context_budget_checker.py [--plugins-dir DIR] [--context-size N]
```

Flags skills at >80% budget as at-risk, >90% as needing refactoring, >3000 bytes as requiring review.

### skill_validator.py

Tests skill triggering pattern matching accuracy.

```bash
python3 tools/validation/skill_validator.py [--plugins-dir DIR] [--plugin NAME]
```

Measures over-trigger rate (<10% target) and under-trigger rate (<10% target).

### xref_validator.py

Validates cross-plugin references and broken links.

```bash
python3 tools/validation/xref_validator.py [--plugins-dir DIR]
```

Checks all `../` relative links in hub skills, agent delegation tables, and cross-suite references.

### doc_checker.py

Checks documentation completeness across plugin READMEs.

```bash
python3 tools/validation/doc_checker.py plugins/<suite>
```

Validates required README sections, markdown formatting, code blocks, and link integrity.

### plugin_review_script.py

Full automated plugin review generating structured markdown reports.

```bash
python3 tools/validation/plugin_review_script.py <suite-name>
```

Runs all validation checks and produces a severity-categorized report (critical/high/medium/low).

## Profiling Tools

### load_profiler.py

Measures plugin loading performance. Target: <100ms per plugin.

```bash
python3 tools/profiling/load_profiler.py <suite-name> [--all]
```

### activation_profiler.py / activation_tester.py

Profile agent activation performance (target: <50ms) and test activation accuracy (FP/FN rates <5%).

```bash
python3 tools/profiling/activation_profiler.py <suite-name> [--all]
python3 tools/profiling/activation_tester.py [--plugin NAME]
```

### command_analyzer.py

Analyzes command suggestion relevance and timing accuracy.

```bash
python3 tools/profiling/command_analyzer.py [--plugin NAME]
```

### memory_analyzer.py

Measures plugin memory consumption. Target: <10MB per plugin.

```bash
python3 tools/profiling/memory_analyzer.py <suite-name> [--all]
```

### performance_reporter.py / triggering_reporter.py

Aggregate performance metrics and triggering pattern reports.

```bash
python3 tools/profiling/performance_reporter.py [--all] [--export json output.json]
python3 tools/profiling/triggering_reporter.py [--reports-dir DIR]
```

## Maintenance Tools

### analyze_ecosystem.py

Analyzes the full skill/agent ecosystem: counts, coverage, hub routing completeness.

```bash
python3 tools/maintenance/analyze_ecosystem.py
```

### enable_all_plugins.py

Enables all MyClaude plugins in the Claude Code settings file.

```bash
python3 tools/maintenance/enable_all_plugins.py
```

## Tests

60 tests covering plugin integrity, validation logic, and documentation completeness.

```bash
# Run all tests
uv run pytest tools/tests/ -v

# Run a single test file
uv run pytest tools/tests/test_agent_core_integrity.py -v

# Run with coverage
uv run pytest tools/tests/ --cov=tools -v
```

**Test coverage by area:**

| Test File | Scope |
|-----------|-------|
| `test_agent_core_integrity.py` | agent-core plugin structure and metadata |
| `test_dev_suite_integrity.py` | dev-suite plugin structure and metadata |
| `test_science_suite_integrity.py` | science-suite plugin structure and metadata |
| `test_science_suite_functionality.py` | science-suite agent/skill functional checks |
| `test_build_automation.py` | Build and packaging automation |
| `test_category_pages.py` | Sphinx category page generation |
| `test_content_extraction.py` | Skill/agent content extraction |
| `test_doc_checker.py` | Documentation validator tests |
| `test_plugin_review.py` | Plugin review script tests |
| `test_refactor_validator.py` | Refactoring validation logic |
| `test_skill_validator.py` | Skill triggering validation tests |
| `test_supplementary_docs.py` | Guide and supplementary doc checks |
| `test_xref_validator.py` | Cross-reference validation tests |

## Output Directories (Auto-Generated)

Tools generate output in gitignored directories:

- **`/reports/`** — Validation and profiling reports
- **`/reviews/`** — Individual plugin review reports

## CI/CD Integration

```bash
#!/bin/bash
set -e
SUITE=$1

python3 tools/validation/metadata_validator.py plugins/$SUITE
python3 tools/validation/doc_checker.py plugins/$SUITE
python3 tools/validation/plugin_review_script.py $SUITE
python3 tools/validation/context_budget_checker.py
python3 tools/validation/xref_validator.py

echo "All validations passed for $SUITE"
```

## Performance Targets

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| Load time | <75ms | 75-100ms | >100ms |
| Activation time | <35ms | 35-50ms | >50ms |
| Memory usage | <5MB | 5-10MB | >10MB |
| FP/FN rate | <5% | 5-10% | >10% |
| Skill accuracy | >90% | 80-90% | <80% |
| Xref validity | 100% | >95% | <95% |

## Requirements

- Python 3.13+ (per `pyproject.toml`)
- No external dependencies (standard library only)
- `uv` for test runner (`uv run pytest`)

## License

MIT License — Same as parent project
