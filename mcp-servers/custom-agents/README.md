# Custom Agents MCP Server

MCP server that enables Claude to autonomously discover and suggest custom agents based on conversation context.

## Features

- **Autonomous Discovery**: Claude can discover available agents without manual invocation
- **Context-Aware Suggestions**: Proactively suggests relevant agents based on conversation
- **Semantic Search**: Find agents by keyword, technology, or task description
- **4 MCP Tools**: Complete programmatic access to agent registry

## Tools

### 1. `list_agents`
Lists all available custom agents with descriptions and capabilities.

**Usage:**
```typescript
mcp__custom-agents__list_agents()
```

**Returns:**
```json
{
  "total": 20,
  "agents": [
    {
      "name": "jax-pro",
      "description": "Core JAX programming specialist...",
      "tools": ["Read", "Write", "Edit", ...],
      "model": "inherit",
      "keywords": ["jax", "optimization", "flax", ...]
    },
    ...
  ]
}
```

### 2. `get_agent`
Get detailed information about a specific agent including its system prompt.

**Parameters:**
- `name` (string, required): Exact name of the agent

**Usage:**
```typescript
mcp__custom-agents__get_agent({ name: "jax-pro" })
```

**Returns:**
```json
{
  "agent": {
    "name": "jax-pro",
    "description": "...",
    "tools": [...],
    "model": "inherit",
    "keywords": [...],
    "triggerPatterns": [...],
    "systemPrompt": "Full system prompt content..."
  }
}
```

### 3. `search_agents`
Search for agents matching a query, returns ranked results with confidence scores.

**Parameters:**
- `query` (string, required): Search query (e.g., "JAX optimization", "database design")
- `limit` (number, optional): Max results (default: 5)
- `threshold` (number, optional): Min confidence score 0-1 (default: 0.1)

**Usage:**
```typescript
mcp__custom-agents__search_agents({
  query: "optimize JAX neural network",
  limit: 3
})
```

**Returns:**
```json
{
  "query": "optimize JAX neural network",
  "total_matches": 3,
  "matches": [
    {
      "agent": {
        "name": "jax-pro",
        "description": "..."
      },
      "confidence": 0.85,
      "reasons": ["keyword_match:75%", "trigger:\"JAX optimization\""],
      "systemPrompt": "..."
    },
    ...
  ]
}
```

### 4. `suggest_agent`
Get the best matching agent for a given context. Returns recommendation with explanation.

**Parameters:**
- `context` (string, required): Task or conversation context

**Usage:**
```typescript
mcp__custom-agents__suggest_agent({
  context: "I need to implement a transformer model in JAX using Flax NNX"
})
```

**Returns:**
```json
{
  "suggestion": "agent_recommended",
  "confidence": 0.92,
  "agent": {
    "name": "jax-pro",
    "description": "...",
    "systemPrompt": "..."
  },
  "reasons": ["name_match", "keyword_match:85%"],
  "usage": "Use /agent --use jax-pro to activate this agent"
}
```

## Architecture

```
custom-agents/
├── src/
│   └── index.ts          # MCP server implementation
├── dist/                 # Compiled JavaScript (generated)
│   ├── index.js
│   └── index.d.ts
├── package.json
├── tsconfig.json
└── README.md
```

**Backend:** Wraps existing `~/.claude/scripts/agent-registry.mjs`
**Protocol:** MCP (Model Context Protocol) over stdio
**Runtime:** Node.js

## How It Works

1. **Registration:** Server registered in `~/.claude.json` mcpServers section
2. **Discovery:** Claude can call MCP tools to discover available agents
3. **Matching:** TF-IDF keyword matching + trigger pattern detection
4. **Suggestion:** Returns best matching agent with confidence score
5. **Activation:** User can activate with `/agent --use <name>` command

## Example Workflow

```
User: "I need to optimize my JAX training loop"

Claude internally calls:
  mcp__custom-agents__suggest_agent({
    context: "optimize JAX training loop"
  })

Response:
  {
    suggestion: "agent_recommended",
    confidence: 0.85,
    agent: { name: "jax-pro", ... },
    usage: "Use /agent --use jax-pro to activate this agent"
  }

Claude to user:
  "I notice you're working on JAX optimization. The jax-pro agent
   specializes in JAX performance optimization with jit/vmap/pmap.
   Would you like me to use it? (Run /agent --use jax-pro)"
```

## Development

```bash
# Install dependencies
cd ~/.claude/mcp-servers/custom-agents
npm install

# Build TypeScript
npm run build

# Watch mode (auto-rebuild)
npm run watch

# Test server manually
node dist/index.js
```

## Troubleshooting

**Server not loading:**
1. Check `~/.claude.json` has correct path to `dist/index.js`
2. Verify build succeeded: `ls -la dist/`
3. Test registry: `node ~/.claude/scripts/agent-cli.mjs list`

**No agents found:**
1. Ensure `~/.claude/agents/` has agent .md files
2. Run `/agent --refresh` to rebuild cache
3. Check agent frontmatter is valid YAML

**Import errors:**
1. Verify `agent-registry.mjs` exists at `~/.claude/scripts/`
2. Check Node.js version (requires v18+)

## Version

1.0.0 - Initial release with 4 core tools

## License

MIT
