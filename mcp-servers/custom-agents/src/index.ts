#!/usr/bin/env node
/**
 * Custom Agents MCP Server
 *
 * Enables Claude to autonomously discover and suggest custom agents
 * based on conversation context.
 *
 * Tools exposed:
 * - list_agents: List all available custom agents
 * - get_agent: Get specific agent details by name
 * - search_agents: Search for agents matching a query
 * - suggest_agent: Get best matching agent for given context
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { join } from 'path';

// Dynamic import of AgentRegistry
let AgentRegistry: any;
let registry: any;
let registryLoaded = false;

async function initRegistry() {
  if (!AgentRegistry) {
    const module = await import(join(process.env.HOME || process.env.USERPROFILE || '', '.claude/scripts/agent-registry.mjs'));
    AgentRegistry = module.default;
    registry = new AgentRegistry();
  }
}

/**
 * Ensure registry is loaded before operations
 */
async function ensureRegistryLoaded() {
  await initRegistry();
  if (!registryLoaded) {
    await registry.load();
    registryLoaded = true;
  }
}

/**
 * MCP Server setup
 */
const server = new Server(
  {
    name: "custom-agents",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Tool: list_agents
 * Lists all available custom agents with basic metadata
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_agents",
        description: "List all available custom agents with their descriptions and capabilities",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_agent",
        description: "Get detailed information about a specific agent by name",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "The exact name of the agent to retrieve",
            },
          },
          required: ["name"],
        },
      },
      {
        name: "search_agents",
        description: "Search for agents that match a query string, returns ranked results with confidence scores",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query describing the task or technology (e.g., 'JAX optimization', 'database design')",
            },
            limit: {
              type: "number",
              description: "Maximum number of results to return (default: 5)",
              default: 5,
            },
            threshold: {
              type: "number",
              description: "Minimum confidence score threshold (0-1, default: 0.1)",
              default: 0.1,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "suggest_agent",
        description: "Get the best matching agent for a given context or task description. Returns the most relevant agent with explanation.",
        inputSchema: {
          type: "object",
          properties: {
            context: {
              type: "string",
              description: "Description of the task, problem, or conversation context",
            },
          },
          required: ["context"],
        },
      },
    ],
  };
});

/**
 * Tool handlers
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  await ensureRegistryLoaded();

  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_agents": {
        const agents = registry.list();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                total: agents.length,
                agents: agents.map((agent: any) => ({
                  name: agent.name,
                  description: agent.description,
                  tools: agent.tools,
                  model: agent.model,
                  keywords: agent.keywords,
                })),
              }, null, 2),
            },
          ],
        };
      }

      case "get_agent": {
        const agentName = (args as any).name;
        const agent = registry.get(agentName);

        if (!agent) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  error: `Agent '${agentName}' not found`,
                  suggestion: "Use list_agents to see all available agents",
                }, null, 2),
              },
            ],
            isError: true,
          };
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                agent: {
                  name: agent.name,
                  description: agent.description,
                  tools: agent.tools,
                  model: agent.model,
                  keywords: agent.keywords,
                  triggerPatterns: agent.triggerPatterns,
                  systemPrompt: agent.systemPrompt,
                },
              }, null, 2),
            },
          ],
        };
      }

      case "search_agents": {
        const { query, limit = 5, threshold = 0.1 } = args as any;
        const matches = registry.match(query, { limit, threshold });

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                query,
                total_matches: matches.length,
                matches: matches.map((match: any) => ({
                  agent: match.agent,
                  confidence: match.score,
                  reasons: match.reasons,
                  systemPrompt: match.systemPrompt,
                })),
              }, null, 2),
            },
          ],
        };
      }

      case "suggest_agent": {
        const { context } = args as any;
        const matches = registry.match(context, { limit: 1, threshold: 0.1 });

        if (matches.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  suggestion: "no_match",
                  message: "No suitable agent found for this context. You can handle this task with general capabilities.",
                  context,
                }, null, 2),
              },
            ],
          };
        }

        const bestMatch = matches[0];
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                suggestion: "agent_recommended",
                confidence: bestMatch.score,
                agent: {
                  name: bestMatch.agent.name,
                  description: bestMatch.agent.description,
                  tools: bestMatch.agent.tools,
                  model: bestMatch.agent.model,
                  systemPrompt: bestMatch.systemPrompt,
                },
                reasons: bestMatch.reasons,
                usage: `Use /agent --use ${bestMatch.agent.name} to activate this agent`,
                context,
              }, null, 2),
            },
          ],
        };
      }

      default:
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: `Unknown tool: ${name}`,
              }, null, 2),
            },
          ],
          isError: true,
        };
    }
  } catch (error: any) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            error: error.message,
            stack: error.stack,
          }, null, 2),
        },
      ],
      isError: true,
    };
  }
});

/**
 * Start server
 */
async function main() {
  await ensureRegistryLoaded();

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("Custom Agents MCP Server running on stdio");
  console.error(`Loaded ${registry.agents?.size || 0} agents`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
