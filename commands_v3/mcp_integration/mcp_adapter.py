"""
MCP Adapter Layer

Adapts actual Claude Code MCP servers to the MCPInterface protocol.
Provides a bridge between the integration system and real MCP implementations.

Example:
    >>> # Adapt existing MCP servers
    >>> memory_bank = MCPAdapter.create_memory_bank(memory_bank_mcp)
    >>> serena = MCPAdapter.create_serena(serena_mcp)
    >>> context7 = MCPAdapter.create_context7(context7_mcp)
    >>>
    >>> # Use with hierarchy
    >>> hierarchy = await KnowledgeHierarchy.create(
    ...     memory_bank=memory_bank,
    ...     serena=serena,
    ...     context7=context7
    ... )
"""

import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MCPAdapter:
    """
    Base adapter for MCP servers.

    Wraps actual MCP implementations to conform to MCPInterface protocol.
    """

    mcp_instance: Any
    mcp_type: str

    @staticmethod
    def create_memory_bank(memory_bank_mcp: Any) -> "MemoryBankAdapter":
        """Create adapter for memory-bank MCP."""
        return MemoryBankAdapter(memory_bank_mcp)

    @staticmethod
    def create_serena(serena_mcp: Any) -> "SerenaAdapter":
        """Create adapter for serena MCP."""
        return SerenaAdapter(serena_mcp)

    @staticmethod
    def create_context7(context7_mcp: Any) -> "Context7Adapter":
        """Create adapter for context7 MCP."""
        return Context7Adapter(context7_mcp)

    @staticmethod
    def create_github(github_mcp: Any) -> "GitHubAdapter":
        """Create adapter for github MCP."""
        return GitHubAdapter(github_mcp)

    @staticmethod
    def create_playwright(playwright_mcp: Any) -> "PlaywrightAdapter":
        """Create adapter for playwright MCP."""
        return PlaywrightAdapter(playwright_mcp)

    @staticmethod
    def create_sequential_thinking(seq_thinking_mcp: Any) -> "SequentialThinkingAdapter":
        """Create adapter for sequential-thinking MCP."""
        return SequentialThinkingAdapter(seq_thinking_mcp)


class MemoryBankAdapter:
    """
    Adapter for allPepper-memory-bank MCP.

    Maps memory-bank operations to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "memory-bank"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from memory-bank.

        Args:
            query: Cache key or search query
            context_type: Type of knowledge
            **kwargs: Additional parameters

        Returns:
            Cached data or None
        """
        # Extract project name from kwargs or use default
        project_name = kwargs.get('project_name', 'default')

        # Determine filename based on context_type
        filename_map = {
            'error': 'errors.json',
            'solution': 'solutions.json',
            'test_stability': 'test_stability.json',
            'code_smell': 'code_smells.json',
            'quality_baseline': 'quality_baseline.json',
            'general': f'{context_type}.json',
        }
        filename = filename_map.get(context_type, f'{context_type}.json')

        try:
            # Use memory_bank_read tool
            result = await self.mcp.memory_bank_read(
                projectName=project_name,
                fileName=filename
            )

            if result:
                # Parse JSON content
                content = json.loads(result) if isinstance(result, str) else result

                # Search for query in content if it's a dict
                if isinstance(content, dict) and query in content:
                    return content[query]

                return content

            return None
        except Exception as e:
            # Silent fail for cache misses
            return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in memory-bank.

        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live (not directly supported by memory-bank)
            **kwargs: Additional parameters (project_name, tags, etc.)

        Returns:
            True if stored successfully
        """
        project_name = kwargs.get('project_name', 'default')
        context_type = kwargs.get('context_type', 'general')
        tags = kwargs.get('tags', [])

        # Determine filename
        filename = f'{context_type}.json'

        try:
            # Read existing content
            existing = await self.mcp.memory_bank_read(
                projectName=project_name,
                fileName=filename
            )

            # Parse or create new content
            if existing:
                content = json.loads(existing) if isinstance(existing, str) else existing
            else:
                content = {}

            # Update with new value
            content[key] = {
                'value': value,
                'ttl': ttl,
                'tags': tags,
                'stored_at': __import__('time').time(),
            }

            # Write back
            await self.mcp.memory_bank_update(
                projectName=project_name,
                fileName=filename,
                content=json.dumps(content, indent=2)
            )

            return True
        except Exception as e:
            print(f"Failed to store in memory-bank: {e}")
            return False


class SerenaAdapter:
    """
    Adapter for serena MCP.

    Maps serena code analysis to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "serena"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from serena (code analysis).

        Args:
            query: Symbol, file path, or search query
            context_type: Type of query (code, symbol, file)
            **kwargs: Additional parameters

        Returns:
            Code analysis results or None
        """
        # Serena doesn't have a direct fetch operation
        # This is a placeholder for when serena adds search/fetch capabilities
        # For now, return None and rely on other MCPs
        return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in serena (not applicable).

        Serena is read-only for code analysis.
        """
        return False


class Context7Adapter:
    """
    Adapter for context7 MCP.

    Maps context7 library documentation to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "context7"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from context7 (library documentation).

        Args:
            query: Library name, API query, or documentation search
            context_type: Type of query (library, api, docs)
            **kwargs: Additional parameters (library_id, topic, tokens)

        Returns:
            Library documentation or None
        """
        library_id = kwargs.get('library_id')
        topic = kwargs.get('topic', query)
        tokens = kwargs.get('tokens', 5000)

        try:
            # If library_id not provided, resolve it
            if not library_id:
                # Extract library name from query
                # e.g., "numpy.array" -> "numpy"
                lib_name = query.split('.')[0].split('(')[0].strip()

                # Resolve library ID
                resolved = await self.mcp.resolve_library_id(libraryName=lib_name)
                if not resolved or 'libraries' not in resolved:
                    return None

                # Get first match
                libraries = resolved['libraries']
                if not libraries:
                    return None

                library_id = libraries[0]['id']

            # Fetch documentation
            docs = await self.mcp.get_library_docs(
                context7CompatibleLibraryID=library_id,
                topic=topic,
                tokens=tokens
            )

            return {
                'library_id': library_id,
                'topic': topic,
                'documentation': docs,
                'source': 'context7',
            }
        except Exception as e:
            print(f"Failed to fetch from context7: {e}")
            return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in context7 (not applicable).

        Context7 is read-only for library documentation.
        """
        return False

    async def resolve_library_id(self, library_name: str) -> Optional[str]:
        """
        Resolve library name to Context7 library ID.

        Args:
            library_name: Library name (e.g., 'numpy', 'react')

        Returns:
            Library ID (e.g., '/numpy/numpy') or None
        """
        try:
            result = await self.mcp.resolve_library_id(libraryName=library_name)

            if result and 'libraries' in result and result['libraries']:
                return result['libraries'][0]['id']

            return None
        except Exception:
            return None


class GitHubAdapter:
    """
    Adapter for github MCP.

    Maps GitHub operations to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "github"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from GitHub.

        Args:
            query: PR number, issue number, or search query
            context_type: Type of query (pr, issue, workflow)
            **kwargs: Additional parameters (owner, repo)

        Returns:
            GitHub data or None
        """
        # GitHub operations are mostly write-based
        # For read operations, this would query GitHub API
        # Placeholder for now
        return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in GitHub (not applicable for caching).

        GitHub is used for operations, not caching.
        """
        return False


class PlaywrightAdapter:
    """
    Adapter for playwright MCP.

    Maps browser automation to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "playwright"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from browser (snapshot, screenshot, etc.).

        Args:
            query: URL or browser action
            context_type: Type of fetch (snapshot, screenshot, content)
            **kwargs: Additional parameters

        Returns:
            Browser data or None
        """
        # Playwright is primarily for actions, not data fetching
        # Could return snapshots or screenshots as data
        # Placeholder for now
        return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in playwright (not applicable).

        Playwright is for browser automation, not storage.
        """
        return False


class SequentialThinkingAdapter:
    """
    Adapter for sequential-thinking MCP.

    Maps reasoning/thinking to MCPInterface protocol.
    """

    def __init__(self, mcp: Any):
        self.mcp = mcp
        self.mcp_type = "sequential-thinking"

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch from sequential thinking (reasoning results).

        Args:
            query: Thinking query or problem
            context_type: Type of reasoning
            **kwargs: Additional parameters

        Returns:
            Reasoning results or None
        """
        # Sequential thinking is procedural, not data-fetching
        # Would need to execute thinking and return results
        # Placeholder for now
        return None

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store in sequential-thinking (not applicable).

        Sequential thinking is for reasoning, not storage.
        """
        return False


# Factory function for easy adapter creation
async def create_mcp_adapters(
    memory_bank_mcp: Optional[Any] = None,
    serena_mcp: Optional[Any] = None,
    context7_mcp: Optional[Any] = None,
    github_mcp: Optional[Any] = None,
    playwright_mcp: Optional[Any] = None,
    sequential_thinking_mcp: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create all MCP adapters.

    Args:
        *_mcp: MCP server instances

    Returns:
        Dictionary of MCP name -> adapter

    Example:
        >>> adapters = await create_mcp_adapters(
        ...     memory_bank_mcp=memory_bank,
        ...     serena_mcp=serena,
        ...     context7_mcp=context7
        ... )
        >>>
        >>> hierarchy = await KnowledgeHierarchy.create(
        ...     memory_bank=adapters['memory-bank'],
        ...     serena=adapters['serena'],
        ...     context7=adapters['context7']
        ... )
    """
    adapters = {}

    if memory_bank_mcp:
        adapters['memory-bank'] = MCPAdapter.create_memory_bank(memory_bank_mcp)

    if serena_mcp:
        adapters['serena'] = MCPAdapter.create_serena(serena_mcp)

    if context7_mcp:
        adapters['context7'] = MCPAdapter.create_context7(context7_mcp)

    if github_mcp:
        adapters['github'] = MCPAdapter.create_github(github_mcp)

    if playwright_mcp:
        adapters['playwright'] = MCPAdapter.create_playwright(playwright_mcp)

    if sequential_thinking_mcp:
        adapters['sequential-thinking'] = MCPAdapter.create_sequential_thinking(sequential_thinking_mcp)

    return adapters
