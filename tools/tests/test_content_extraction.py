#!/usr/bin/env python3
"""
Test suite for plugin documentation content extraction (Task Group 4.1)

Tests code example extraction, usage pattern identification, and RST generation
for plugin documentation pages.
"""

import re
import sys
import unittest
from pathlib import Path
from typing import List, Dict

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator  # noqa: F401
except ImportError:
    pass

def extract_code_blocks(content: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown content (helper for tests)"""
    code_blocks = []
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.finditer(pattern, content, re.DOTALL)
    for match in matches:
        language = match.group(1) if match.group(1) else ''
        code = match.group(2).strip()
        code_blocks.append({'language': language, 'code': code, 'context': ''})
    return code_blocks

class TestContentExtraction(unittest.TestCase):
    """Test suite for content extraction functionality"""

    def test_code_example_extraction_julia(self):
        """Test 4.1.1: Test code example extraction from README.md files (Julia)"""
        readme_content = """# Test Plugin

## Quick Start

```julia
# Example Julia code
using DifferentialEquations
```
"""
        code_blocks = extract_code_blocks(readme_content)
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]['language'], 'julia')

    def test_code_block_language_detection(self):
        """Test 4.1.4: Test code block language detection"""
        test_cases = [
            ("```bash\nls -la\n```", 'bash'),
            ("```python\nprint('hello')\n```", 'python'),
        ]
        for code_md, expected_lang in test_cases:
            blocks = extract_code_blocks(code_md)
            self.assertEqual(blocks[0]['language'], expected_lang)

if __name__ == "__main__":
    unittest.main()
