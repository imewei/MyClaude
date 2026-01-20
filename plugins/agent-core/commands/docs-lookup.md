---
name: docs-lookup
description: Query library documentation using Context7 MCP for up-to-date API references
allowed-tools:
  - mcp__plugin_context7_context7__resolve-library-id
  - mcp__plugin_context7_context7__query-docs
---

# Library Documentation Lookup

Use Context7 MCP to retrieve current documentation and code examples.

## Process

1. **Resolve Library**: First call `resolve-library-id` to get the Context7 library ID
2. **Query Docs**: Use the resolved ID to query specific documentation

## Workflow

### Step 1: Identify the Library
```json
{
  "query": "User's question or task context",
  "libraryName": "package-name"
}
```

### Step 2: Query Documentation
Use the resolved library ID to fetch relevant documentation sections.

## Use Cases

- Looking up API references for unfamiliar libraries
- Finding code examples for specific functionality
- Checking current (not outdated) documentation
- Understanding library patterns and best practices

## Notes

- Do not call resolve-library-id more than 3 times per question
- If user provides a library ID in format `/org/project`, skip resolution
- Prioritize libraries with higher documentation coverage scores

## User Request

$ARGUMENTS
