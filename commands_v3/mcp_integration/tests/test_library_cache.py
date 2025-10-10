"""
Unit Tests for LibraryCache

Tests library ID caching, detection, and fallback mechanisms.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_integration import LibraryCache, LibraryInfo, DetectionPattern, DetectionType


@pytest.fixture
def sample_libraries():
    """Sample library definitions."""
    return {
        'numpy': LibraryInfo(
            name='numpy',
            id='/numpy/numpy',
            aliases=['np'],
            category='scientific'
        ),
        'pytorch': LibraryInfo(
            name='pytorch',
            id='/pytorch/pytorch',
            aliases=['torch'],
            category='ml'
        ),
        'react': LibraryInfo(
            name='react',
            id='/facebook/react',
            aliases=[],
            category='frontend'
        ),
    }


@pytest.fixture
def detection_patterns():
    """Sample detection patterns."""
    return [
        DetectionPattern(
            pattern=r'import\s+numpy|from\s+numpy',
            library='numpy',
            detection_type=DetectionType.IMPORT
        ),
        DetectionPattern(
            pattern=r'import\s+torch|from\s+torch',
            library='pytorch',
            detection_type=DetectionType.IMPORT
        ),
        DetectionPattern(
            pattern=r'@jax\.jit',
            library='jax',
            detection_type=DetectionType.DECORATOR
        ),
    ]


@pytest.fixture
def library_cache(sample_libraries, detection_patterns):
    """Create LibraryCache instance."""
    return LibraryCache(
        libraries=sample_libraries,
        detection_patterns=detection_patterns,
        enable_fallback=True
    )


class TestLibraryCache:
    """Test LibraryCache functionality."""

    @pytest.mark.asyncio
    async def test_get_library_id_cache_hit(self, library_cache):
        """Test cache hit for known library."""
        lib_id = await library_cache.get_library_id("numpy")
        assert lib_id == "/numpy/numpy"
        assert library_cache.stats["cache_hits"] == 1
        assert library_cache.stats["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_get_library_id_alias(self, library_cache):
        """Test library lookup via alias."""
        lib_id = await library_cache.get_library_id("torch")
        assert lib_id == "/pytorch/pytorch"
        assert library_cache.stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_get_library_id_cache_miss(self, library_cache):
        """Test cache miss for unknown library."""
        lib_id = await library_cache.get_library_id("unknown-lib", use_fallback=False)
        assert lib_id is None
        assert library_cache.stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_get_library_id_fallback(self, library_cache):
        """Test fallback to context7 API."""
        # Mock context7 MCP
        mock_context7 = AsyncMock()
        mock_context7.resolve_library_id = AsyncMock(return_value="/owner/unknown-lib")
        library_cache.context7_mcp = mock_context7

        lib_id = await library_cache.get_library_id("unknown-lib", use_fallback=True)

        assert lib_id == "/owner/unknown-lib"
        assert library_cache.stats["fallback_calls"] == 1
        mock_context7.resolve_library_id.assert_called_once_with("unknown-lib")

    def test_detect_libraries_import(self, library_cache):
        """Test library detection from imports."""
        code = """
import numpy as np
import torch
from sklearn import datasets
        """
        detected = library_cache.detect_libraries(code)

        detected_names = [lib.name for lib in detected]
        assert 'numpy' in detected_names
        assert 'pytorch' in detected_names

    def test_detect_libraries_decorator(self, library_cache):
        """Test library detection from decorators."""
        code = """
@jax.jit
def compute(x):
    return x ** 2
        """
        detected = library_cache.detect_libraries(code)

        detected_names = [lib.name for lib in detected]
        assert 'jax' in detected_names

    def test_detect_libraries_no_match(self, library_cache):
        """Test detection with no matching patterns."""
        code = """
def hello():
    print("Hello, World!")
        """
        detected = library_cache.detect_libraries(code)
        assert len(detected) == 0

    def test_get_by_category(self, library_cache):
        """Test getting libraries by category."""
        scientific = library_cache.get_by_category("scientific")
        assert len(scientific) == 1
        assert scientific[0].name == "numpy"

        ml = library_cache.get_by_category("ml")
        assert len(ml) == 1
        assert ml[0].name == "pytorch"

    def test_get_common_libraries(self, library_cache):
        """Test getting common libraries."""
        common = library_cache.get_common_libraries(limit=2)
        assert len(common) <= 2
        assert all(isinstance(lib, LibraryInfo) for lib in common)

    def test_stats_tracking(self, library_cache):
        """Test statistics tracking."""
        stats = library_cache.get_stats()

        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "total_libraries" in stats
        assert stats["total_libraries"] == 3

    def test_reset_stats(self, library_cache):
        """Test stats reset."""
        library_cache.stats["cache_hits"] = 10
        library_cache.stats["cache_misses"] = 5

        library_cache.reset_stats()

        assert library_cache.stats["cache_hits"] == 0
        assert library_cache.stats["cache_misses"] == 0


class TestLibraryInfo:
    """Test LibraryInfo dataclass."""

    def test_matches_name(self):
        """Test matching by name."""
        lib = LibraryInfo(name="numpy", id="/numpy/numpy", aliases=["np"])
        assert lib.matches("numpy")
        assert lib.matches("NUMPY")  # Case insensitive

    def test_matches_alias(self):
        """Test matching by alias."""
        lib = LibraryInfo(name="numpy", id="/numpy/numpy", aliases=["np"])
        assert lib.matches("np")
        assert lib.matches("NP")  # Case insensitive

    def test_no_match(self):
        """Test no match."""
        lib = LibraryInfo(name="numpy", id="/numpy/numpy", aliases=["np"])
        assert not lib.matches("torch")


class TestDetectionPattern:
    """Test DetectionPattern dataclass."""

    def test_compiled_pattern(self):
        """Test regex pattern compilation."""
        pattern = DetectionPattern(
            pattern=r'import\s+numpy',
            library='numpy',
            detection_type=DetectionType.IMPORT
        )

        compiled = pattern.compiled_pattern
        assert compiled.search("import numpy as np") is not None
        assert compiled.search("import torch") is None

    def test_pattern_caching(self):
        """Test that pattern is compiled once and cached."""
        pattern = DetectionPattern(
            pattern=r'import\s+numpy',
            library='numpy',
            detection_type=DetectionType.IMPORT
        )

        compiled1 = pattern.compiled_pattern
        compiled2 = pattern.compiled_pattern
        assert compiled1 is compiled2  # Same object


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
