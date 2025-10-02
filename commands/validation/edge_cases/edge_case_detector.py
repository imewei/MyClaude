#!/usr/bin/env python3
"""
Edge Case Detector - Detects edge cases in validation projects.
"""

import os
from pathlib import Path
from typing import Any, Dict, List


class EdgeCaseDetector:
    """Detects edge cases in projects."""

    def detect(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect edge cases in project."""
        edge_cases = []

        # Large files
        edge_cases.extend(self._detect_large_files(project_path))

        # Deep nesting
        edge_cases.extend(self._detect_deep_nesting(project_path))

        # Complex dependencies
        edge_cases.extend(self._detect_complex_dependencies(project_path))

        # Mixed encodings
        edge_cases.extend(self._detect_encoding_issues(project_path))

        return edge_cases

    def _detect_large_files(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect very large files."""
        edge_cases = []
        threshold_lines = 10000

        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = sum(1 for _ in f)

                        if lines > threshold_lines:
                            edge_cases.append({
                                'type': 'large_file',
                                'file': str(file_path.relative_to(project_path)),
                                'lines': lines,
                                'severity': 'high' if lines > 20000 else 'medium'
                            })
                    except Exception:
                        continue

        return edge_cases

    def _detect_deep_nesting(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect deeply nested directory structures."""
        edge_cases = []
        max_depth = 10

        for root, _, _ in os.walk(project_path):
            depth = len(Path(root).relative_to(project_path).parts)
            if depth > max_depth:
                edge_cases.append({
                    'type': 'deep_nesting',
                    'path': str(Path(root).relative_to(project_path)),
                    'depth': depth,
                    'severity': 'medium'
                })

        return edge_cases[:10]  # Limit results

    def _detect_complex_dependencies(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect complex dependency patterns."""
        edge_cases = []

        # Check for requirements.txt with many dependencies
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    deps = [l.strip() for l in f if l.strip() and not l.startswith('#')]

                if len(deps) > 50:
                    edge_cases.append({
                        'type': 'many_dependencies',
                        'file': 'requirements.txt',
                        'count': len(deps),
                        'severity': 'high' if len(deps) > 100 else 'medium'
                    })
            except Exception:
                pass

        return edge_cases

    def _detect_encoding_issues(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect files with encoding issues."""
        edge_cases = []

        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read()
                    except UnicodeDecodeError:
                        edge_cases.append({
                            'type': 'encoding_issue',
                            'file': str(file_path.relative_to(project_path)),
                            'severity': 'medium'
                        })
                    except Exception:
                        continue

        return edge_cases[:10]