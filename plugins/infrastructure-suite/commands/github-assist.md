---
version: "2.2.0"
name: github-assist
description: GitHub operations using GitHub MCP for issues, PRs, repos, and code search
argument-hint: "[query] [--type=issue|pr|code] [--org=name] [--label=name]"
color: blue
allowed-tools:
  - mcp__plugin_github_github__get_me
  - mcp__plugin_github_github__search_issues
  - mcp__plugin_github_github__search_pull_requests
  - mcp__plugin_github_github__search_code
  - mcp__plugin_github_github__list_issues
  - mcp__plugin_github_github__list_pull_requests
  - mcp__plugin_github_github__issue_read
  - mcp__plugin_github_github__pull_request_read
  - mcp__plugin_github_github__create_pull_request
  - mcp__plugin_github_github__add_issue_comment
---

# GitHub Operations with GitHub MCP

Use GitHub MCP for comprehensive repository interactions.

## Common Operations

### Get Current User Context
Always start with `get_me` to understand permissions and context.

### Search Operations
Use `search_*` tools for targeted queries with specific criteria:
- `search_issues`: Find issues by keywords, labels, assignee
- `search_pull_requests`: Find PRs by author, status, labels
- `search_code`: Find code containing specific patterns

### List Operations
Use `list_*` tools for broad retrieval with basic filtering:
- `list_issues`: All issues with state/label filters
- `list_pull_requests`: All PRs with state filters

## Query Tips

- Use separate `sort` and `order` parameters (not in query string)
- Query strings: `org:name language:python label:bug`
- Use `minimal_output: true` when full details aren't needed
- Paginate with batches of 5-10 items

## PR Workflow

1. Search for existing PR templates in `.github/PULL_REQUEST_TEMPLATE`
2. Use template content to structure PR description
3. Create PR with `create_pull_request`

## User Request

$ARGUMENTS
