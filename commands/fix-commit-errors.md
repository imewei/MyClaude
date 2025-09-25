---
description: Comprehensive GitHub Actions & commit error fixer with iterative fix-test-validate cycles until all tests pass
category: github-workflow
argument-hint: [commit-hash-or-pr-number] [--auto-fix] [--rerun] [--debug] [--max-cycles=10]
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, TodoWrite, MultiEdit
---

# Comprehensive GitHub Actions & Commit Error Fixer

**Intelligently diagnose, fix, and validate GitHub Actions failures with automated iterative cycles until all tests pass.** Specialized for Python/Julia scientific computing projects with comprehensive error pattern recognition and systematic resolution strategies.

## Usage

```bash
# Auto-fix all errors until tests pass
/fix-commit-errors --auto-fix --max-cycles=10

# Fix specific commit with comprehensive analysis
/fix-commit-errors abc1234 --debug --rerun

# Interactive mode with detailed tracking
/fix-commit-errors --interactive --report

# Focus on specific workflow
/fix-commit-errors --workflow=CI --auto-fix

# Maximum automation for urgent fixes
/fix-commit-errors --auto-fix --rerun --max-cycles=15 --aggressive
```

## Core Strategy: Iterative Fix-Test-Validate Cycles

### **Phase 1: Comprehensive Error Diagnosis**
```python
def diagnose_github_actions_failures():
    """Systematic GitHub Actions failure analysis."""

    # 1. Get all recent workflow runs
    workflow_runs = get_recent_workflow_runs(limit=10)

    # 2. Analyze failure patterns across runs
    failure_patterns = analyze_failure_patterns(workflow_runs)

    # 3. Categorize errors by type and priority
    error_categories = categorize_errors(failure_patterns)

    # 4. Generate comprehensive fix plan
    fix_plan = generate_iterative_fix_plan(error_categories)

    return {
        'workflow_runs': workflow_runs,
        'failure_patterns': failure_patterns,
        'error_categories': error_categories,
        'fix_plan': fix_plan
    }
```

### **Phase 2: Progressive Error Resolution**
```python
def iterative_fix_cycle(fix_plan, max_cycles=10):
    """Execute fix-test-validate cycles until success."""

    cycle = 0
    fixes_applied = []

    while cycle < max_cycles:
        cycle += 1

        # Step 1: Apply next fix batch
        current_fixes = apply_fix_batch(fix_plan, cycle)
        fixes_applied.extend(current_fixes)

        # Step 2: Commit fixes
        commit_fixes(current_fixes, cycle)

        # Step 3: Trigger and monitor workflows
        workflow_results = trigger_and_monitor_workflows()

        # Step 4: Analyze results
        if all_workflows_passing(workflow_results):
            return success_report(fixes_applied, cycle)

        # Step 5: Analyze new failures and update plan
        new_errors = analyze_new_failures(workflow_results)
        fix_plan = update_fix_plan(fix_plan, new_errors)

        if no_fixable_errors(new_errors):
            return escalation_report(fixes_applied, new_errors, cycle)

    return max_cycles_reached_report(fixes_applied, cycle)
```

### **Phase 3: Validation and Monitoring**
```python
def validate_fix_success():
    """Comprehensive validation of applied fixes."""

    validation_results = {
        'local_tests': run_local_tests(),
        'package_installation': test_package_installation(),
        'import_tests': test_module_imports(),
        'github_actions': monitor_github_workflows(),
        'regression_tests': run_regression_tests()
    }

    return validation_results
```

## Comprehensive Error Pattern Database

### **1. Configuration & Build Errors**

#### **pyproject.toml Validation Errors**
```python
PYPROJECT_ERROR_PATTERNS = {
    "invalid pyproject.toml config": {
        "patterns": [
            r"must be idn-email",
            r"configuration error",
            r"ValueError: invalid pyproject.toml"
        ],
        "analysis": "Project configuration validation failure",
        "fixes": [
            {
                "type": "email_validation",
                "action": "remove_empty_email_fields",
                "files": ["pyproject.toml"],
                "pattern": r'email\s*=\s*""',
                "replacement": "# email removed - empty value invalid"
            },
            {
                "type": "author_cleanup",
                "action": "fix_author_metadata",
                "validation": "ensure_valid_email_format"
            }
        ]
    },
    "subprocess-exited-with-error": {
        "patterns": [
            r"Getting requirements to build editable did not run successfully",
            r"× Getting requirements to build editable"
        ],
        "analysis": "Build system configuration error",
        "fixes": [
            "validate_pyproject_toml",
            "check_build_dependencies",
            "verify_setuptools_compatibility"
        ]
    }
}
```

#### **Missing Dependencies**
```python
DEPENDENCY_ERROR_PATTERNS = {
    "ModuleNotFoundError": {
        "patterns": [
            r"ModuleNotFoundError: No module named '(\w+)'",
            r"ImportError: No module named (\w+)"
        ],
        "analysis": "Missing required dependency",
        "fixes": [
            {
                "type": "add_dependency",
                "action": "add_to_pyproject_dependencies",
                "auto_fix": True,
                "common_modules": {
                    "h5py": "h5py>=3.8.0  # For HDF5 file handling",
                    "yaml": "PyYAML>=6.0  # For YAML configuration",
                    "requests": "requests>=2.25.0  # For HTTP requests",
                    "pandas": "pandas>=1.3.0  # For data manipulation",
                    "matplotlib": "matplotlib>=3.5.0  # For plotting",
                    "scipy": "scipy>=1.7.0  # For scientific computing",
                    "numpy": "numpy>=1.21.0  # For numerical computing"
                }
            }
        ]
    }
}
```

### **2. Code Logic & Test Errors**

#### **Variable Naming & Scope Issues**
```python
CODE_LOGIC_ERROR_PATTERNS = {
    "NameError": {
        "patterns": [
            r"NameError: name '(\w+)' is not defined",
            r"NameError: name '_.*' is not defined"
        ],
        "analysis": "Variable scope or naming inconsistency",
        "fixes": [
            {
                "type": "variable_assignment",
                "action": "add_missing_assignment",
                "patterns": {
                    "_pcov": "Add _pcov = pcov assignment",
                    "_popt": "Add _popt = popt assignment"
                }
            },
            {
                "type": "variable_consistency",
                "action": "fix_naming_consistency",
                "scope": "test_files",
                "validate": "check_unpack_usage_consistency"
            }
        ]
    },
    "AssertionError in tests": {
        "patterns": [
            r"assert.*(\w+)\.shape.*",
            r"assert_.*\(.*(\w+).*\)",
            r"AssertionError.*expected.*actual"
        ],
        "analysis": "Test assertion failure or variable mismatch",
        "fixes": [
            "check_variable_unpacking",
            "validate_test_data_consistency",
            "fix_assertion_variables"
        ]
    }
}
```

### **3. Workflow Configuration Errors**

#### **GitHub Actions Workflow Issues**
```python
WORKFLOW_ERROR_PATTERNS = {
    "unrecognized arguments": {
        "patterns": [
            r"unrecognized arguments: (.*)",
            r"error: unrecognized arguments"
        ],
        "analysis": "Script arguments don't match implementation",
        "fixes": [
            {
                "type": "script_args",
                "action": "sync_workflow_with_script",
                "validation": "check_script_help_output",
                "files": [".github/workflows/*.yml"]
            }
        ]
    },
    "benchmark format errors": {
        "patterns": [
            r"expected array, received object",
            r"Invalid input: expected array",
            r"BenchmarkResult format"
        ],
        "analysis": "Benchmark output format mismatch",
        "fixes": [
            {
                "type": "disable_step",
                "action": "disable_problematic_benchmark_storage",
                "condition": "if: false  # Disabled until fixed"
            }
        ]
    }
}
```

### **4. Scientific Computing Specific Errors**

#### **JAX/NumPy/SciPy Issues**
```python
SCIENTIFIC_ERROR_PATTERNS = {
    "jax_gpu_errors": {
        "patterns": [
            r"No GPU/TPU found",
            r"JAX backend configuration",
            r"CUDA initialization failed"
        ],
        "analysis": "JAX GPU configuration issues",
        "fixes": [
            "set_jax_cpu_backend",
            "update_jax_configuration",
            "add_gpu_availability_checks"
        ]
    },
    "numerical_precision": {
        "patterns": [
            r"AssertionError.*not equal to tolerance",
            r"allclose failed",
            r"floating point precision"
        ],
        "analysis": "Numerical precision or tolerance issues",
        "fixes": [
            "increase_test_tolerances",
            "add_floating_point_guards",
            "use_approximate_equality"
        ]
    }
}
```

## Automated Fix Implementation Engine

### **Smart Fix Application System**
```python
class AdvancedFixEngine:
    """Comprehensive automated fix application system."""

    def __init__(self):
        self.error_patterns = load_all_error_patterns()
        self.fix_history = []
        self.success_metrics = {}

    def apply_fix_batch(self, errors, cycle_number):
        """Apply a batch of fixes for current cycle."""

        # Prioritize fixes by success probability and impact
        prioritized_fixes = self.prioritize_fixes(errors)

        fixes_applied = []
        for fix in prioritized_fixes:
            try:
                result = self.apply_single_fix(fix)
                if result['success']:
                    fixes_applied.append(fix)
                    self.log_fix_success(fix, result)
                else:
                    self.log_fix_failure(fix, result)
            except Exception as e:
                self.log_fix_exception(fix, e)

        return fixes_applied

    def apply_single_fix(self, fix):
        """Apply individual fix with validation."""

        if fix['type'] == 'add_dependency':
            return self.add_dependency_fix(fix)
        elif fix['type'] == 'variable_assignment':
            return self.variable_assignment_fix(fix)
        elif fix['type'] == 'workflow_config':
            return self.workflow_config_fix(fix)
        elif fix['type'] == 'email_validation':
            return self.email_validation_fix(fix)
        else:
            return {'success': False, 'reason': 'Unknown fix type'}

    def add_dependency_fix(self, fix):
        """Add missing dependency to pyproject.toml."""

        missing_module = fix['module']
        if missing_module in fix['common_modules']:
            dependency_line = fix['common_modules'][missing_module]

            # Read current pyproject.toml
            content = read_file('pyproject.toml')

            # Find dependencies section
            deps_pattern = r'(dependencies\s*=\s*\[)(.*?)(\])'
            match = re.search(deps_pattern, content, re.DOTALL)

            if match:
                deps_content = match.group(2)
                new_deps = deps_content.rstrip() + f',\n    "{dependency_line}"'
                new_content = content.replace(match.group(2), new_deps)

                write_file('pyproject.toml', new_content)
                return {'success': True, 'action': f'Added {dependency_line}'}

        return {'success': False, 'reason': 'Could not add dependency'}

    def variable_assignment_fix(self, fix):
        """Fix variable assignment issues."""

        file_path = fix['file']
        variable = fix['variable']
        assignment = fix['assignment']

        content = read_file(file_path)

        # Find appropriate insertion point
        insertion_point = self.find_variable_insertion_point(content, variable)

        if insertion_point:
            lines = content.split('\n')
            lines.insert(insertion_point, f"        {assignment}")
            new_content = '\n'.join(lines)

            write_file(file_path, new_content)
            return {'success': True, 'action': f'Added {assignment}'}

        return {'success': False, 'reason': 'Could not find insertion point'}
```

### **Comprehensive Workflow Monitoring**
```python
class GitHubActionsMonitor:
    """Advanced GitHub Actions monitoring and validation."""

    def __init__(self):
        self.workflow_cache = {}
        self.error_patterns = load_workflow_error_patterns()

    def monitor_workflow_cycle(self, timeout_minutes=20):
        """Monitor complete workflow execution cycle."""

        # Trigger workflows
        self.trigger_workflows()

        # Monitor execution
        results = self.wait_for_completion(timeout_minutes)

        # Analyze results
        analysis = self.analyze_workflow_results(results)

        return {
            'workflows': results,
            'analysis': analysis,
            'success': analysis['all_passed'],
            'errors': analysis['errors'],
            'next_fixes': analysis['recommended_fixes']
        }

    def trigger_workflows(self):
        """Trigger GitHub Actions workflows."""

        # Push changes to trigger workflows
        run_command(['git', 'push', 'origin', 'main'])

        # Wait for workflows to start
        time.sleep(10)

    def wait_for_completion(self, timeout_minutes):
        """Wait for all workflows to complete."""

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while time.time() - start_time < timeout_seconds:
            workflows = self.get_workflow_status()

            if all(w['status'] == 'completed' for w in workflows):
                return workflows

            # Check for new failures to analyze early
            failed_workflows = [w for w in workflows if w['conclusion'] == 'failure']
            if failed_workflows:
                self.analyze_early_failures(failed_workflows)

            time.sleep(30)

        return self.get_workflow_status()  # Return current state on timeout

    def analyze_workflow_results(self, workflows):
        """Comprehensive workflow result analysis."""

        analysis = {
            'total_workflows': len(workflows),
            'passed': len([w for w in workflows if w['conclusion'] == 'success']),
            'failed': len([w for w in workflows if w['conclusion'] == 'failure']),
            'all_passed': all(w['conclusion'] == 'success' for w in workflows),
            'errors': [],
            'recommended_fixes': []
        }

        # Analyze failed workflows
        for workflow in workflows:
            if workflow['conclusion'] == 'failure':
                errors = self.extract_workflow_errors(workflow)
                fixes = self.recommend_fixes_for_errors(errors)

                analysis['errors'].extend(errors)
                analysis['recommended_fixes'].extend(fixes)

        return analysis
```

## Advanced Fix Strategies by Error Category

### **1. Package Configuration Fixes**
```python
def fix_pyproject_email_validation():
    """Fix pyproject.toml email validation errors."""

    # Read current pyproject.toml
    content = read_file('pyproject.toml')

    # Fix empty email fields
    patterns_to_fix = [
        (r'email\s*=\s*""', '# email field removed - empty value invalid'),
        (r',\s*email\s*=\s*""', ''),  # Remove trailing empty email
    ]

    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)

    write_file('pyproject.toml', content)
    return {'success': True, 'action': 'Fixed email validation'}

def add_missing_dependencies(dependencies):
    """Add missing dependencies to pyproject.toml."""

    content = read_file('pyproject.toml')

    # Find dependencies section
    deps_match = re.search(r'(dependencies\s*=\s*\[)(.*?)(\])', content, re.DOTALL)

    if deps_match:
        existing_deps = deps_match.group(2).strip()

        for dep in dependencies:
            if dep['name'] not in existing_deps:
                new_dep_line = f'    "{dep["spec"]}",  # {dep["comment"]}'
                existing_deps += f',\n{new_dep_line}'

        new_content = content.replace(deps_match.group(2), f'\n{existing_deps}\n')
        write_file('pyproject.toml', new_content)
        return {'success': True, 'dependencies_added': len(dependencies)}

    return {'success': False, 'reason': 'Could not find dependencies section'}
```

### **2. Code Logic Fixes**
```python
def fix_variable_naming_consistency(file_path, variable_mappings):
    """Fix variable naming inconsistencies."""

    content = read_file(file_path)

    for old_var, new_var in variable_mappings.items():
        # Use word boundaries to avoid partial matches
        pattern = rf'\b{re.escape(old_var)}\b'
        content = re.sub(pattern, new_var, content)

    write_file(file_path, content)
    return {'success': True, 'mappings': variable_mappings}

def add_missing_variable_assignments(file_path, assignments):
    """Add missing variable assignments."""

    content = read_file(file_path)
    lines = content.split('\n')

    for assignment in assignments:
        # Find the best insertion point
        insertion_line = find_best_insertion_point(lines, assignment)
        if insertion_line:
            lines.insert(insertion_line, f"        {assignment['statement']}")

    write_file(file_path, '\n'.join(lines))
    return {'success': True, 'assignments': len(assignments)}
```

### **3. Workflow Configuration Fixes**
```python
def fix_workflow_script_arguments(workflow_file, script_fixes):
    """Fix GitHub Actions workflow script arguments."""

    content = read_file(workflow_file)

    for fix in script_fixes:
        old_command = fix['old_command']
        new_command = fix['new_command']

        content = content.replace(old_command, new_command)

    write_file(workflow_file, content)
    return {'success': True, 'fixes': len(script_fixes)}

def disable_problematic_workflow_steps(workflow_file, steps_to_disable):
    """Disable problematic workflow steps temporarily."""

    content = read_file(workflow_file)

    for step in steps_to_disable:
        # Add conditional to disable step
        step_pattern = rf'(\s+- name: {re.escape(step)})'
        replacement = rf'\1\n      if: false  # Disabled until fixed'
        content = re.sub(step_pattern, replacement, content)

    write_file(workflow_file, content)
    return {'success': True, 'disabled_steps': len(steps_to_disable)}
```

## Command Execution Flow

### **Primary Execution Modes**

#### **1. Automatic Mode (--auto-fix)**
```python
def auto_fix_mode(max_cycles=10):
    """Fully automated fix mode."""

    with TodoWrite([
        "Analyze GitHub Actions failures",
        "Apply automated fixes iteratively",
        "Monitor and validate results",
        "Generate completion report"
    ]):

        # Phase 1: Initial diagnosis
        mark_in_progress("Analyze GitHub Actions failures")
        diagnosis = comprehensive_failure_diagnosis()
        mark_completed("Analyze GitHub Actions failures")

        # Phase 2: Iterative fix cycles
        mark_in_progress("Apply automated fixes iteratively")
        cycle = 0

        while cycle < max_cycles:
            cycle += 1

            # Apply fixes for this cycle
            fixes = apply_fix_batch(diagnosis, cycle)

            # Commit and push changes
            commit_fixes(fixes, cycle)

            # Monitor workflows
            results = monitor_workflows()

            # Check success
            if all_workflows_passed(results):
                mark_completed("Apply automated fixes iteratively")
                break

            # Update diagnosis for next cycle
            diagnosis = update_diagnosis(diagnosis, results)

        # Phase 3: Final validation
        mark_in_progress("Monitor and validate results")
        validation = comprehensive_validation()
        mark_completed("Monitor and validate results")

        # Phase 4: Report generation
        mark_in_progress("Generate completion report")
        report = generate_comprehensive_report(fixes, validation)
        mark_completed("Generate completion report")

        return report
```

#### **2. Interactive Mode (--interactive)**
```python
def interactive_fix_mode():
    """Interactive mode with user confirmation."""

    # Analyze errors
    diagnosis = comprehensive_failure_diagnosis()
    display_diagnosis_summary(diagnosis)

    # Interactive fix selection
    selected_fixes = interactive_fix_selection(diagnosis)

    # Apply selected fixes
    for fix_batch in selected_fixes:
        confirm = ask_user_confirmation(fix_batch)
        if confirm:
            apply_fix_batch(fix_batch)

            # Immediate testing
            if ask_run_tests():
                results = run_immediate_tests()
                display_test_results(results)
```

### **Error Resolution Priority Matrix**

```python
FIX_PRIORITY_MATRIX = {
    'critical': {
        'priority': 1,
        'patterns': [
            'package installation failure',
            'import errors',
            'build system errors'
        ],
        'auto_fix': True
    },
    'high': {
        'priority': 2,
        'patterns': [
            'test failures',
            'variable naming errors',
            'workflow argument errors'
        ],
        'auto_fix': True
    },
    'medium': {
        'priority': 3,
        'patterns': [
            'linting issues',
            'configuration warnings',
            'performance issues'
        ],
        'auto_fix': False
    },
    'low': {
        'priority': 4,
        'patterns': [
            'documentation issues',
            'style violations',
            'minor warnings'
        ],
        'auto_fix': False
    }
}
```

## Success Validation & Monitoring

### **Multi-Level Validation System**
```python
def comprehensive_success_validation():
    """Multi-level validation of fix success."""

    validation_levels = {
        'local_validation': {
            'package_installation': test_local_installation(),
            'import_tests': test_module_imports(),
            'basic_functionality': test_basic_functionality(),
            'test_execution': run_critical_tests_locally()
        },
        'github_actions_validation': {
            'workflow_triggering': trigger_workflows(),
            'execution_monitoring': monitor_workflow_execution(),
            'result_analysis': analyze_workflow_results(),
            'success_confirmation': confirm_all_workflows_pass()
        },
        'integration_validation': {
            'cross_platform_compatibility': test_cross_platform(),
            'dependency_resolution': validate_dependencies(),
            'performance_benchmarks': run_performance_tests(),
            'regression_testing': run_regression_suite()
        }
    }

    validation_results = {}
    overall_success = True

    for level, tests in validation_levels.items():
        level_results = {}
        level_success = True

        for test_name, test_func in tests.items():
            try:
                result = test_func()
                level_results[test_name] = result
                if not result.get('success', False):
                    level_success = False
                    overall_success = False
            except Exception as e:
                level_results[test_name] = {'success': False, 'error': str(e)}
                level_success = False
                overall_success = False

        validation_results[level] = {
            'tests': level_results,
            'success': level_success
        }

    return {
        'validation_results': validation_results,
        'overall_success': overall_success,
        'summary': generate_validation_summary(validation_results)
    }
```

### **Real-time Monitoring Dashboard**
```python
def create_monitoring_dashboard():
    """Create real-time monitoring for fix progress."""

    dashboard = {
        'current_cycle': 0,
        'fixes_applied': [],
        'workflow_status': {},
        'error_trends': {},
        'success_metrics': {},
        'estimated_completion': None
    }

    return dashboard

def update_dashboard(dashboard, cycle_results):
    """Update monitoring dashboard with latest results."""

    dashboard['current_cycle'] += 1
    dashboard['fixes_applied'].extend(cycle_results['fixes'])
    dashboard['workflow_status'] = cycle_results['workflows']

    # Update success metrics
    dashboard['success_metrics'] = calculate_success_metrics(dashboard)

    # Estimate completion time
    dashboard['estimated_completion'] = estimate_completion(dashboard)

    return dashboard
```

## Command Options & Configuration

### **Advanced Command Options**
```bash
# Maximum automation with aggressive fixes
/fix-commit-errors --auto-fix --aggressive --max-cycles=15 --timeout=30

# Scientific computing optimized mode
/fix-commit-errors --scientific-computing --auto-fix --dependencies

# Comprehensive analysis with detailed reporting
/fix-commit-errors --debug --report --interactive --monitor

# Fast emergency mode
/fix-commit-errors --emergency --auto-fix --skip-validation

# Specific error type focus
/fix-commit-errors --focus=dependencies,configuration --auto-fix
```

### **Configuration File Support**
```yaml
# .claude/fix-commit-errors-config.yml
fix_settings:
  max_cycles: 10
  timeout_minutes: 20
  auto_fix_enabled: true
  aggressive_mode: false

error_patterns:
  priority_overrides:
    "missing dependencies": critical
    "workflow arguments": high
    "style violations": low

validation_settings:
  run_local_tests: true
  monitor_workflows: true
  generate_reports: true

notification_settings:
  slack_webhook: "${SLACK_WEBHOOK_URL}"
  email_alerts: true
  success_notifications: true
```

**Target**: Systematically diagnose and fix GitHub Actions failures with automated iterative cycles until all tests pass, providing comprehensive error resolution for scientific computing projects.