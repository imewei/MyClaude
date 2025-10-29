#!/usr/bin/env python3
"""
Test suite for plugin documentation content extraction (Task Group 4.1)

Tests code example extraction, usage pattern identification, and RST generation
for plugin documentation pages.
"""

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sphinx_doc_generator import SphinxDocGenerator


class ContentExtractionTester:
    """Test suite for content extraction functionality"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results: List[Dict[str, Any]] = []

    def log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)

    def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        self.log(f"\n🧪 Running: {test_name}")
        try:
            test_func()
            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASS",
                "error": None
            })
            self.log(f"✅ PASS: {test_name}")
        except AssertionError as e:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            self.log(f"❌ FAIL: {test_name}")
            self.log(f"   Error: {e}")
        except Exception as e:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "ERROR",
                "error": str(e)
            })
            self.log(f"💥 ERROR: {test_name}")
            self.log(f"   Error: {e}")

    def test_code_example_extraction_julia(self):
        """Test 4.1.1: Test code example extraction from README.md files (Julia)"""
        # Create temporary test README with Julia code blocks
        readme_content = """# Test Plugin

## Quick Start

```julia
# Example Julia code
using DifferentialEquations

function lotka_volterra!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end
```

## Advanced Usage

```julia
# ODE solver
prob = ODEProblem(lotka_volterra!, u0, tspan, params)
sol = solve(prob, Tsit5())
```
"""

        # Test extraction
        code_blocks = self._extract_code_blocks(readme_content)

        assert len(code_blocks) == 2, f"Expected 2 code blocks, got {len(code_blocks)}"
        assert code_blocks[0]['language'] == 'julia', f"Expected julia, got {code_blocks[0]['language']}"
        assert 'DifferentialEquations' in code_blocks[0]['code'], "Expected DifferentialEquations in first block"
        assert 'ODEProblem' in code_blocks[1]['code'], "Expected ODEProblem in second block"

    def test_code_example_extraction_python(self):
        """Test 4.1.2: Test code example extraction from Python README"""
        readme_content = """# Python Plugin

## Installation

```bash
pip install fastapi uvicorn
```

## Usage

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```
"""

        code_blocks = self._extract_code_blocks(readme_content)

        assert len(code_blocks) == 2, f"Expected 2 code blocks, got {len(code_blocks)}"
        assert code_blocks[0]['language'] == 'bash', f"Expected bash, got {code_blocks[0]['language']}"
        assert code_blocks[1]['language'] == 'python', f"Expected python, got {code_blocks[1]['language']}"
        assert 'FastAPI' in code_blocks[1]['code'], "Expected FastAPI in Python code"

    def test_usage_pattern_identification(self):
        """Test 4.1.3: Test usage pattern identification"""
        readme_content = """# Test Plugin

## Quick Start

Quick start section content.

```bash
# Installation
npm install package
```

## Advanced Usage

Advanced usage content.

```javascript
const pkg = require('package');
pkg.run();
```

## Configuration

Configuration section.

```yaml
settings:
  enabled: true
```
"""

        patterns = self._identify_usage_patterns(readme_content)

        assert 'Quick Start' in patterns, "Expected Quick Start section"
        assert 'Advanced Usage' in patterns, "Expected Advanced Usage section"
        assert 'Configuration' in patterns, "Expected Configuration section"
        assert len(patterns['Quick Start']) == 1, "Expected 1 code block in Quick Start"
        assert patterns['Quick Start'][0]['language'] == 'bash', "Expected bash in Quick Start"

    def test_code_block_language_detection(self):
        """Test 4.1.4: Test code block language detection (bash, python, julia, etc.)"""
        test_cases = [
            ("```bash\nls -la\n```", 'bash'),
            ("```python\nprint('hello')\n```", 'python'),
            ("```julia\nprintln('hello')\n```", 'julia'),
            ("```yaml\nkey: value\n```", 'yaml'),
            ("```json\n{\"key\": \"value\"}\n```", 'json'),
            ("```javascript\nconsole.log('hi')\n```", 'javascript'),
            ("```\nno language\n```", ''),  # No language specified
        ]

        for code_md, expected_lang in test_cases:
            blocks = self._extract_code_blocks(code_md)
            assert len(blocks) == 1, f"Expected 1 block for {expected_lang}"
            assert blocks[0]['language'] == expected_lang, \
                f"Expected {expected_lang}, got {blocks[0]['language']}"

    def test_rst_code_block_directive_generation(self):
        """Test 4.1.5: Test RST code-block directive generation"""
        code_block = {
            'language': 'python',
            'code': 'def hello():\n    print("Hello, world!")',
            'context': 'Basic Python function'
        }

        rst = self._generate_rst_code_block(code_block)

        # Verify RST format
        assert '.. code-block:: python' in rst, "Expected code-block directive"
        assert 'def hello():' in rst, "Expected Python code"
        assert 'print("Hello, world!")' in rst, "Expected print statement"

        # Verify the structure - should have context, then directive, then indented code
        lines = rst.split('\n')
        assert lines[0] == 'Basic Python function', "Expected context on first line"
        assert lines[1] == '', "Expected blank line after context"
        assert lines[2] == '.. code-block:: python', "Expected directive"
        assert lines[3] == '', "Expected blank line after directive"
        # Code lines should be indented
        assert lines[4].startswith('   '), "Expected indented code"

    def test_complete_plugin_page_generation(self):
        """Test 4.1.6: Test complete plugin page generation for sample plugins"""
        # Create temporary plugin directory with plugin.json and README
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test-plugin"
            plugin_dir.mkdir()

            # Create plugin.json
            plugin_json = {
                "name": "test-plugin",
                "version": "1.0.0",
                "description": "Test plugin for validation",
                "author": "Test Author",
                "license": "MIT",
                "category": "development",
                "keywords": ["test", "example"],
                "agents": [
                    {
                        "name": "test-agent",
                        "description": "Test agent",
                        "status": "active"
                    }
                ],
                "commands": [
                    {
                        "name": "/test-command",
                        "description": "Test command",
                        "status": "active"
                    }
                ],
                "skills": [
                    {
                        "name": "test-skill",
                        "description": "Test skill",
                        "status": "active"
                    }
                ]
            }

            with open(plugin_dir / "plugin.json", 'w') as f:
                json.dump(plugin_json, f, indent=2)

            # Create README with code examples
            readme_content = """# Test Plugin

## Quick Start

```bash
# Install
npm install test-plugin
```

## Usage

```python
import test_plugin
test_plugin.run()
```
"""
            with open(plugin_dir / "README.md", 'w') as f:
                f.write(readme_content)

            # Generate documentation
            plugins_dir = Path(tmpdir)
            output_dir = Path(tmpdir) / "docs" / "plugins"
            output_dir.mkdir(parents=True)

            generator = SphinxDocGenerator(plugins_dir, verbose=self.verbose)
            generator.discover_plugins()
            generator.detect_integrations()
            generator.build_reverse_dependencies()
            generator.identify_integration_patterns()

            rst_content = generator.generate_plugin_rst(plugin_dir, output_dir)

            # Verify RST content
            assert "Test Plugin" in rst_content, "Expected plugin title"
            assert "Description" in rst_content, "Expected Description section"
            assert "Agents" in rst_content, "Expected Agents section"
            assert "Commands" in rst_content, "Expected Commands section"
            assert "Skills" in rst_content, "Expected Skills section"
            assert "Usage Examples" in rst_content, "Expected Usage Examples section"
            assert "Integration" in rst_content, "Expected Integration section"

            # Verify file was created
            output_file = output_dir / "test-plugin.rst"
            assert output_file.exists(), "Expected RST file to be created"

    def test_code_extraction_with_context(self):
        """Test 4.1.7: Test code example extraction preserves context"""
        readme_content = """# Plugin

## Installation Steps

Follow these steps to install:

```bash
pip install package
```

## Basic Configuration

Configure the settings:

```yaml
settings:
  enabled: true
  port: 8000
```
"""

        patterns = self._identify_usage_patterns(readme_content)

        # Verify context is preserved
        assert 'Installation Steps' in patterns
        install_block = patterns['Installation Steps'][0]
        assert install_block['language'] == 'bash'
        assert 'pip install' in install_block['code']

        assert 'Basic Configuration' in patterns
        config_block = patterns['Basic Configuration'][0]
        assert config_block['language'] == 'yaml'
        assert 'enabled: true' in config_block['code']

    def test_multi_language_extraction(self):
        """Test 4.1.8: Test extraction handles multiple languages in one README"""
        readme_content = """# Multi-Language Plugin

## Backend (Python)

```python
from fastapi import FastAPI
app = FastAPI()
```

## Frontend (JavaScript)

```javascript
const app = new Vue({
  el: '#app'
})
```

## Configuration (YAML)

```yaml
backend:
  host: localhost
  port: 8000
frontend:
  port: 3000
```

## Deployment (Bash)

```bash
docker-compose up -d
```
"""

        code_blocks = self._extract_code_blocks(readme_content)

        assert len(code_blocks) == 4, f"Expected 4 code blocks, got {len(code_blocks)}"

        languages = [block['language'] for block in code_blocks]
        assert 'python' in languages, "Expected python code"
        assert 'javascript' in languages, "Expected javascript code"
        assert 'yaml' in languages, "Expected yaml code"
        assert 'bash' in languages, "Expected bash code"

    # Helper methods

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content"""
        code_blocks = []

        # Pattern to match fenced code blocks with optional language
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) if match.group(1) else ''
            code = match.group(2).strip()

            code_blocks.append({
                'language': language,
                'code': code,
                'context': ''
            })

        return code_blocks

    def _identify_usage_patterns(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """Identify usage patterns by extracting code blocks with their section context"""
        patterns = {}

        # Split into sections based on headers
        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            # Detect markdown headers (## Header)
            if line.startswith('## '):
                # Save previous section
                if current_section and section_content:
                    section_text = '\n'.join(section_content)
                    code_blocks = self._extract_code_blocks(section_text)
                    if code_blocks:
                        patterns[current_section] = code_blocks

                # Start new section
                current_section = line[3:].strip()
                section_content = []
            else:
                section_content.append(line)

        # Save last section
        if current_section and section_content:
            section_text = '\n'.join(section_content)
            code_blocks = self._extract_code_blocks(section_text)
            if code_blocks:
                patterns[current_section] = code_blocks

        return patterns

    def _generate_rst_code_block(self, code_block: Dict[str, str]) -> str:
        """Generate RST code-block directive from code block data"""
        lines = []

        # Add context if available
        if code_block.get('context'):
            lines.append(code_block['context'])
            lines.append('')

        # Add code-block directive
        language = code_block.get('language', '')
        if language:
            lines.append(f".. code-block:: {language}")
        else:
            lines.append(".. code-block::")
        lines.append('')

        # Add code with proper indentation (3 spaces for RST)
        code = code_block['code']
        for code_line in code.split('\n'):
            lines.append(f"   {code_line}")

        return '\n'.join(lines)

    def print_summary(self):
        """Print test summary"""
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "=" * 70)
        print("Content Extraction Test Summary")
        print("=" * 70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print("=" * 70)

        if self.tests_failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if result['status'] != 'PASS':
                    print(f"  ❌ {result['name']}: {result['error']}")

        return self.tests_failed == 0


def main():
    """Run all content extraction tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test plugin content extraction")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    args = parser.parse_args()

    tester = ContentExtractionTester(verbose=args.verbose)

    # Run all tests (8 tests as per Task 4.1)
    tester.run_test("Code Example Extraction (Julia)",
                    tester.test_code_example_extraction_julia)
    tester.run_test("Code Example Extraction (Python)",
                    tester.test_code_example_extraction_python)
    tester.run_test("Usage Pattern Identification",
                    tester.test_usage_pattern_identification)
    tester.run_test("Code Block Language Detection",
                    tester.test_code_block_language_detection)
    tester.run_test("RST Code-Block Directive Generation",
                    tester.test_rst_code_block_directive_generation)
    tester.run_test("Complete Plugin Page Generation",
                    tester.test_complete_plugin_page_generation)
    tester.run_test("Code Extraction with Context",
                    tester.test_code_extraction_with_context)
    tester.run_test("Multi-Language Extraction",
                    tester.test_multi_language_extraction)

    # Print summary
    success = tester.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
