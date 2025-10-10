"""
Unit Tests for KnowledgeHierarchy

Tests three-tier knowledge retrieval with authority rules and caching.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_integration import (
    KnowledgeHierarchy,
    Knowledge,
    KnowledgeSource,
    AuthorityRule,
)


@pytest.fixture
def mock_memory_bank():
    """Mock memory-bank MCP."""
    mock = AsyncMock()
    mock.fetch = AsyncMock(return_value=None)
    mock.store = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_serena():
    """Mock serena MCP."""
    mock = AsyncMock()
    mock.fetch = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_context7():
    """Mock context7 MCP."""
    mock = AsyncMock()
    mock.fetch = AsyncMock(return_value=None)
    return mock


@pytest.fixture
async def hierarchy(mock_memory_bank, mock_serena, mock_context7):
    """Create KnowledgeHierarchy instance."""
    return await KnowledgeHierarchy.create(
        memory_bank=mock_memory_bank,
        serena=mock_serena,
        context7=mock_context7
    )


class TestKnowledgeHierarchy:
    """Test KnowledgeHierarchy functionality."""

    @pytest.mark.asyncio
    async def test_fetch_from_memory_bank(self, hierarchy, mock_memory_bank):
        """Test successful fetch from memory-bank."""
        mock_memory_bank.fetch.return_value = {"data": "cached"}

        result = await hierarchy.fetch("test_query", context_type="general")

        assert result.success
        assert result.source == KnowledgeSource.MEMORY_BANK
        assert result.content == {"data": "cached"}
        assert result.cached

    @pytest.mark.asyncio
    async def test_fetch_from_serena(self, hierarchy, mock_memory_bank, mock_serena):
        """Test fallback to serena."""
        mock_memory_bank.fetch.return_value = None
        mock_serena.fetch.return_value = {"code": "def main(): pass"}

        result = await hierarchy.fetch("test_query", context_type="project_code")

        assert result.success
        assert result.source == KnowledgeSource.SERENA
        assert result.content == {"code": "def main(): pass"}

    @pytest.mark.asyncio
    async def test_fetch_from_context7(self, hierarchy, mock_memory_bank, mock_serena, mock_context7):
        """Test fallback to context7."""
        mock_memory_bank.fetch.return_value = None
        mock_serena.fetch.return_value = None
        mock_context7.fetch.return_value = {"docs": "API documentation"}

        result = await hierarchy.fetch("test_query", context_type="library_api")

        assert result.success
        assert result.source == KnowledgeSource.CONTEXT7
        assert result.content == {"docs": "API documentation"}

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, hierarchy, mock_memory_bank, mock_serena, mock_context7):
        """Test knowledge not found in any source."""
        mock_memory_bank.fetch.return_value = None
        mock_serena.fetch.return_value = None
        mock_context7.fetch.return_value = None

        result = await hierarchy.fetch("unknown_query")

        assert not result.success
        assert result.source == KnowledgeSource.NONE
        assert result.content is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_authority_rule_library_api(self, hierarchy, mock_context7):
        """Test authority rule for library API queries."""
        mock_context7.fetch.return_value = {"docs": "API docs"}

        result = await hierarchy.fetch(
            "numpy.array",
            context_type="library_api",
            authority_rule=AuthorityRule.LIBRARY_API
        )

        # context7 should be checked first for library APIs
        assert result.source == KnowledgeSource.CONTEXT7
        mock_context7.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_authority_rule_project_code(self, hierarchy, mock_serena):
        """Test authority rule for project code queries."""
        mock_serena.fetch.return_value = {"code": "project code"}

        result = await hierarchy.fetch(
            "main.py",
            context_type="project_code",
            authority_rule=AuthorityRule.PROJECT_CODE
        )

        # serena should be checked first for project code
        assert result.source == KnowledgeSource.SERENA
        mock_serena.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_authority_rule_auto(self, hierarchy, mock_context7):
        """Test automatic authority rule selection."""
        mock_context7.fetch.return_value = {"docs": "docs"}

        result = await hierarchy.fetch(
            "api query",
            context_type="api",
            authority_rule=AuthorityRule.AUTO
        )

        # AUTO + context_type="api" should use LIBRARY_API rule
        assert result.source == KnowledgeSource.CONTEXT7

    @pytest.mark.asyncio
    async def test_caching_behavior(self, hierarchy, mock_memory_bank, mock_serena):
        """Test that results are cached in memory-bank."""
        mock_memory_bank.fetch.return_value = None
        mock_serena.fetch.return_value = {"code": "data"}

        result = await hierarchy.fetch("query", context_type="code")

        # Should cache in memory-bank
        assert mock_memory_bank.store.called
        store_call = mock_memory_bank.store.call_args
        assert "code" in store_call[1]["value"]["content"]

    @pytest.mark.asyncio
    async def test_no_mcp_available(self):
        """Test behavior when no MCPs are available."""
        empty_hierarchy = await KnowledgeHierarchy.create()

        result = await empty_hierarchy.fetch("query")

        assert not result.success
        assert result.source == KnowledgeSource.NONE
        assert "No MCP servers available" in result.error

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, hierarchy, mock_memory_bank):
        """Test statistics tracking."""
        mock_memory_bank.fetch.return_value = {"cached": True}

        # Multiple queries
        await hierarchy.fetch("query1")
        await hierarchy.fetch("query2")

        stats = hierarchy.get_stats()

        assert stats["total_queries"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_hit_rate"] == 1.0
        assert stats["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_reset_stats(self, hierarchy, mock_memory_bank):
        """Test stats reset."""
        mock_memory_bank.fetch.return_value = {"data": "test"}

        await hierarchy.fetch("query")
        assert hierarchy.stats["total_queries"] == 1

        hierarchy.reset_stats()
        assert hierarchy.stats["total_queries"] == 0
        assert hierarchy.stats["cache_hits"] == 0


class TestKnowledge:
    """Test Knowledge dataclass."""

    def test_success_property(self):
        """Test success property."""
        # Successful knowledge
        knowledge = Knowledge(
            content={"data": "value"},
            source=KnowledgeSource.SERENA,
            latency_ms=100
        )
        assert knowledge.success

        # Failed knowledge (no content)
        knowledge = Knowledge(
            content=None,
            source=KnowledgeSource.NONE,
            latency_ms=50,
            error="Not found"
        )
        assert not knowledge.success

        # Failed knowledge (error)
        knowledge = Knowledge(
            content={"data": "value"},
            source=KnowledgeSource.SERENA,
            latency_ms=100,
            error="Some error"
        )
        assert not knowledge.success


class TestAuthorityRule:
    """Test AuthorityRule enum."""

    def test_authority_rule_values(self):
        """Test authority rule values."""
        assert AuthorityRule.LIBRARY_API.value == "library_api"
        assert AuthorityRule.PROJECT_CODE.value == "project_code"
        assert AuthorityRule.PATTERNS.value == "patterns"
        assert AuthorityRule.AUTO.value == "auto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
