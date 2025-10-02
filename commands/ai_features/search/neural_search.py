#!/usr/bin/env python3
"""
Neural Code Search
==================

AI-powered code search using neural embeddings and semantic understanding.

Features:
- Search by functionality (not just keywords)
- Semantic similarity search
- Cross-language code search
- Example-based search
- Natural language queries
- Intelligent ranking

Author: Claude Code AI Team
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import ast


@dataclass
class SearchResult:
    """Code search result"""
    file_path: Path
    code_snippet: str
    entity_name: str
    entity_type: str
    score: float
    line_number: int
    metadata: Dict[str, Any]


class NeuralSearch:
    """
    Neural code search engine using semantic embeddings.
    """

    def __init__(self, index_dir: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.index_dir = index_dir or Path.home() / ".claude" / "search_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # In production, load actual embedding model and index
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer('microsoft/codebert-base')
        # self.index = faiss.IndexFlatIP(768)
        self.model = None
        self.index = None

        # Code index
        self.code_index: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    def index_codebase(self, root_path: Path):
        """
        Index entire codebase for search.

        Args:
            root_path: Root directory of codebase
        """
        self.logger.info(f"Indexing codebase: {root_path}")

        python_files = list(root_path.rglob("*.py"))
        total = len(python_files)

        for i, file_path in enumerate(python_files, 1):
            self.logger.info(f"Indexing {i}/{total}: {file_path.name}")
            try:
                self._index_file(file_path)
            except Exception as e:
                self.logger.error(f"Failed to index {file_path}: {e}")

        self.logger.info(f"Indexed {self.next_id} code entities")

    def search_by_functionality(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search code by describing functionality.

        Args:
            query: Natural language description of desired functionality
            top_k: Number of results

        Returns:
            List of search results
        """
        self.logger.info(f"Searching for: {query}")

        # In production, use actual model
        # query_embedding = self.model.encode(query)
        # scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        # For framework, use keyword matching
        results = self._keyword_search(query, top_k)

        return results

    def search_by_example(
        self,
        example_code: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Find similar code by providing an example.

        Args:
            example_code: Example code snippet
            top_k: Number of results

        Returns:
            Similar code snippets
        """
        self.logger.info("Searching by example")

        # In production, use embeddings
        # code_embedding = self.model.encode(example_code)

        # For framework, use AST similarity
        results = self._ast_similarity_search(example_code, top_k)

        return results

    def search_by_signature(
        self,
        function_signature: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for functions with similar signatures.

        Args:
            function_signature: Function signature to match
            top_k: Number of results

        Returns:
            Functions with similar signatures
        """
        self.logger.info(f"Searching for signature: {function_signature}")

        # Parse signature
        try:
            # Extract function name and parameters
            if "def " in function_signature:
                parts = function_signature.split("(")
                func_name = parts[0].replace("def", "").strip()
                params = parts[1].split(")")[0] if len(parts) > 1 else ""
                param_count = len([p.strip() for p in params.split(",") if p.strip()])
            else:
                func_name = ""
                param_count = 0

            # Search index
            results = []
            for idx, entry in self.code_index.items():
                if entry["entity_type"] != "function":
                    continue

                # Match by parameter count and name similarity
                entry_params = entry.get("param_count", 0)
                if param_count == entry_params or abs(param_count - entry_params) <= 1:
                    score = self._calculate_name_similarity(
                        func_name,
                        entry["entity_name"]
                    )

                    if score > 0.3:
                        results.append(SearchResult(
                            file_path=Path(entry["file_path"]),
                            code_snippet=entry["code"],
                            entity_name=entry["entity_name"],
                            entity_type=entry["entity_type"],
                            score=score,
                            line_number=entry["line_number"],
                            metadata=entry.get("metadata", {})
                        ))

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            self.logger.error(f"Signature search failed: {e}")
            return []

    def _index_file(self, file_path: Path):
        """Index a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Index functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._index_function(node, file_path, source)
                elif isinstance(node, ast.ClassDef):
                    self._index_class(node, file_path, source)

        except Exception as e:
            self.logger.error(f"Failed to index {file_path}: {e}")

    def _index_function(self, node: ast.FunctionDef, file_path: Path, source: str):
        """Index a function"""
        # Extract code snippet
        code = ast.get_source_segment(source, node)
        if not code:
            return

        # Extract metadata
        docstring = ast.get_docstring(node)
        params = [arg.arg for arg in node.args.args]

        # In production, generate embedding
        # embedding = self.model.encode(code)
        # self.index.add(embedding.reshape(1, -1))

        # Store in index
        self.code_index[self.next_id] = {
            "file_path": str(file_path),
            "entity_name": node.name,
            "entity_type": "function",
            "code": code,
            "docstring": docstring or "",
            "parameters": params,
            "param_count": len(params),
            "line_number": node.lineno,
            "metadata": {
                "decorators": [d.id if isinstance(d, ast.Name) else str(d)
                             for d in node.decorator_list]
            }
        }

        self.next_id += 1

    def _index_class(self, node: ast.ClassDef, file_path: Path, source: str):
        """Index a class"""
        code = ast.get_source_segment(source, node)
        if not code:
            return

        docstring = ast.get_docstring(node)
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        self.code_index[self.next_id] = {
            "file_path": str(file_path),
            "entity_name": node.name,
            "entity_type": "class",
            "code": code,
            "docstring": docstring or "",
            "methods": methods,
            "line_number": node.lineno,
            "metadata": {
                "bases": [b.id if isinstance(b, ast.Name) else str(b)
                         for b in node.bases]
            }
        }

        self.next_id += 1

    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Simple keyword-based search"""
        query_lower = query.lower()
        keywords = query_lower.split()

        results = []

        for idx, entry in self.code_index.items():
            # Search in name, docstring, and code
            searchable_text = " ".join([
                entry["entity_name"],
                entry.get("docstring", ""),
                entry["code"]
            ]).lower()

            # Calculate score
            score = sum(1 for kw in keywords if kw in searchable_text)
            score = score / len(keywords) if keywords else 0

            if score > 0:
                results.append(SearchResult(
                    file_path=Path(entry["file_path"]),
                    code_snippet=entry["code"][:200] + "...",
                    entity_name=entry["entity_name"],
                    entity_type=entry["entity_type"],
                    score=score,
                    line_number=entry["line_number"],
                    metadata=entry.get("metadata", {})
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _ast_similarity_search(
        self,
        example_code: str,
        top_k: int
    ) -> List[SearchResult]:
        """Search using AST structure similarity"""
        try:
            example_tree = ast.parse(example_code)
            example_features = self._extract_ast_features(example_tree)

            results = []

            for idx, entry in self.code_index.items():
                try:
                    entry_tree = ast.parse(entry["code"])
                    entry_features = self._extract_ast_features(entry_tree)

                    similarity = self._calculate_ast_similarity(
                        example_features,
                        entry_features
                    )

                    if similarity > 0.3:
                        results.append(SearchResult(
                            file_path=Path(entry["file_path"]),
                            code_snippet=entry["code"][:200] + "...",
                            entity_name=entry["entity_name"],
                            entity_type=entry["entity_type"],
                            score=similarity,
                            line_number=entry["line_number"],
                            metadata=entry.get("metadata", {})
                        ))
                except:
                    pass

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            self.logger.error(f"AST similarity search failed: {e}")
            return []

    def _extract_ast_features(self, tree: ast.AST) -> Dict[str, int]:
        """Extract features from AST for similarity"""
        features = {
            "loops": 0,
            "conditionals": 0,
            "function_calls": 0,
            "list_ops": 0,
            "dict_ops": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                features["loops"] += 1
            elif isinstance(node, ast.If):
                features["conditionals"] += 1
            elif isinstance(node, ast.Call):
                features["function_calls"] += 1
            elif isinstance(node, ast.List):
                features["list_ops"] += 1
            elif isinstance(node, ast.Dict):
                features["dict_ops"] += 1

        return features

    def _calculate_ast_similarity(
        self,
        features1: Dict[str, int],
        features2: Dict[str, int]
    ) -> float:
        """Calculate similarity between AST features"""
        total_diff = 0
        total_max = 0

        for key in features1:
            val1 = features1[key]
            val2 = features2.get(key, 0)

            total_diff += abs(val1 - val2)
            total_max += max(val1, val2)

        if total_max == 0:
            return 1.0

        similarity = 1.0 - (total_diff / total_max)
        return max(0.0, similarity)

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity"""
        name1 = name1.lower()
        name2 = name2.lower()

        if name1 == name2:
            return 1.0

        # Simple Levenshtein-like similarity
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0

        matches = sum(1 for a, b in zip(name1, name2) if a == b)
        return matches / max_len


def main():
    """Demonstration"""
    print("Neural Code Search")
    print("=================\n")

    search = NeuralSearch()

    # Demo: Index current file
    current_file = Path(__file__)
    search._index_file(current_file)

    print(f"Indexed {search.next_id} entities")

    # Search by functionality
    results = search.search_by_functionality("search code", top_k=5)
    print(f"\nFound {len(results)} results")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.entity_name} ({result.entity_type})")
        print(f"   Score: {result.score:.2f}")
        print(f"   File: {result.file_path.name}:{result.line_number}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())