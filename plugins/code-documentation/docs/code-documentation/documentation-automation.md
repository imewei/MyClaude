# Documentation Automation

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: CI/CD pipelines, README generation, and documentation coverage automation

## Overview

Automate documentation generation, validation, and deployment using CI/CD pipelines, ensuring documentation stays synchronized with code changes.

## GitHub Actions Workflow

### Complete Documentation Pipeline

```yaml
name: Generate Documentation

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'api/**'
      - 'docs/**'
  pull_request:
    branches: [main]

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements-docs.txt
        npm install -g @redocly/cli

    - name: Generate API documentation
      run: |
        python scripts/generate_openapi.py > docs/api/openapi.json
        redocly build-docs docs/api/openapi.json -o docs/api/index.html

    - name: Generate code documentation
      run: |
        sphinx-build -b html docs/source docs/build
        sphinx-build -b linkcheck docs/source docs/build

    - name: Check documentation coverage
      run: |
        interrogate -v src/ --fail-under 80

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build
```

## README Generation

### Automated README Template

```python
def generate_readme(project_info):
    """Generate comprehensive README from project metadata"""

    template = f'''# {project_info['name']}

{generate_badges(project_info)}

{project_info['description']}

## Features

{generate_feature_list(project_info['features'])}

## Installation

### Prerequisites

{generate_prerequisites(project_info['requirements'])}

### Using {project_info['package_manager']}

```bash
{project_info['install_command']}
```

### From Source

```bash
git clone {project_info['repo_url']}
cd {project_info['name']}
{project_info['setup_commands']}
```

## Quick Start

```{project_info['language']}
{project_info['quickstart_code']}
```

## Documentation

Full documentation: {project_info['docs_url']}

## Configuration

{generate_config_table(project_info['config'])}

## Development

### Running Tests

```bash
{project_info['test_command']}
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

{project_info['license']}
'''

    return template
```

## Documentation Coverage Validation

### Python Docstring Coverage Check

```python
import ast
import glob

class DocCoverage:
    def check_coverage(self, codebase_path):
        """Check documentation coverage for codebase"""
        results = {
            'total_functions': 0,
            'documented_functions': 0,
            'total_classes': 0,
            'documented_classes': 0,
            'missing_docs': []
        }

        for file_path in glob.glob(f"{codebase_path}/**/*.py", recursive=True):
            module = ast.parse(open(file_path).read())

            for node in ast.walk(module):
                if isinstance(node, ast.FunctionDef):
                    results['total_functions'] += 1
                    if ast.get_docstring(node):
                        results['documented_functions'] += 1
                    else:
                        results['missing_docs'].append({
                            'type': 'function',
                            'name': node.name,
                            'file': file_path,
                            'line': node.lineno
                        })

                elif isinstance(node, ast.ClassDef):
                    results['total_classes'] += 1
                    if ast.get_docstring(node):
                        results['documented_classes'] += 1
                    else:
                        results['missing_docs'].append({
                            'type': 'class',
                            'name': node.name,
                            'file': file_path,
                            'line': node.lineno
                        })

        # Calculate coverage percentages
        results['function_coverage'] = (
            results['documented_functions'] / results['total_functions'] * 100
            if results['total_functions'] > 0 else 100
        )
        results['class_coverage'] = (
            results['documented_classes'] / results['total_classes'] * 100
            if results['total_classes'] > 0 else 100
        )

        return results
```

## Documentation Linting

### Markdown Linting Configuration

```yaml
# .markdownlint.json
{
  "default": true,
  "MD013": false,
  "MD033": {
    "allowed_elements": ["details", "summary", "img"]
  },
  "MD041": false
}
```

### Pre-commit Hook for Documentation

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/igorshubovych/markdownlint-cli
    rev: v0.37.0
    hooks:
      - id: markdownlint
        args: ['--fix']

  - repo: local
    hooks:
      - id: doc-coverage
        name: Check documentation coverage
        entry: interrogate
        args: ['-v', 'src/', '--fail-under', '80']
        language: system
        pass_filenames: false
```

## Continuous Documentation Deployment

### Netlify Configuration

```toml
# netlify.toml
[build]
  command = "sphinx-build -b html docs/source docs/build"
  publish = "docs/build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[context.production.environment]
  PYTHON_VERSION = "3.11"
```

### Read the Docs Configuration

```yaml
# .readthedocs.yml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/conf.py
  fail_on_warning: true

formats:
  - pdf
  - epub

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

## Usage Examples

### Automated Doc Generation Script

```bash
#!/bin/bash
# scripts/generate_docs.sh

echo "Generating API documentation..."
python scripts/generate_openapi.py > docs/api/openapi.json

echo "Building Sphinx documentation..."
cd docs && make clean && make html

echo "Checking coverage..."
interrogate -v src/ --fail-under 80

echo "Running link checker..."
cd docs && make linkcheck

echo "Documentation generated successfully!"
```
