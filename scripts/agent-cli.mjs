#!/usr/bin/env node
/**
 * Command-line interface for agent registry
 * Used by slash commands to query agents
 *
 * Commands:
 * - list: Show all available agents
 * - match <query>: Find best matching agents for query
 * - get <name>: Retrieve specific agent by name
 * - refresh: Force rebuild of agent cache
 */

import AgentRegistry from './agent-registry.mjs';

const registry = new AgentRegistry();

async function main() {
  const command = process.argv[2];
  const args = process.argv.slice(3);

  try {
    await registry.load();

    switch (command) {
      case 'list':
        await handleList();
        break;

      case 'match':
        await handleMatch(args.join(' '));
        break;

      case 'get':
        await handleGet(args[0]);
        break;

      case 'refresh':
        await handleRefresh();
        break;

      case 'stats':
        await handleStats();
        break;

      default:
        showUsage();
        process.exit(1);
    }
  } catch (err) {
    console.error('❌ Error:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

async function handleList() {
  const agents = registry.list();
  console.log(JSON.stringify(agents, null, 2));
}

async function handleMatch(query) {
  if (!query || query.trim().length === 0) {
    console.error('❌ Query cannot be empty');
    process.exit(1);
  }

  const matches = registry.match(query, { limit: 5, threshold: 0.1 });
  console.log(JSON.stringify(matches, null, 2));
}

async function handleGet(name) {
  if (!name) {
    console.error('❌ Agent name required');
    process.exit(1);
  }

  const agent = registry.get(name);
  if (!agent) {
    console.error(`❌ Agent not found: ${name}`);
    process.exit(1);
  }

  console.log(JSON.stringify(agent, null, 2));
}

async function handleRefresh() {
  await registry.fullScan();
  await registry.saveCache();
  console.log(JSON.stringify({
    success: true,
    count: registry.agents.size,
    message: `Refreshed ${registry.agents.size} agents`
  }, null, 2));
}

async function handleStats() {
  const stats = {
    totalAgents: registry.agents.size,
    agentNames: Array.from(registry.agents.keys()),
    lastScan: registry.metadata.lastScan,
    cacheFile: registry.cacheFile,
    scanPaths: registry.agentPaths
  };

  console.log(JSON.stringify(stats, null, 2));
}

function showUsage() {
  console.error(`
Agent CLI - Dynamic Agent Registry Interface

Usage:
  agent-cli.mjs <command> [arguments]

Commands:
  list              List all available agents
  match <query>     Find best matching agents for query
  get <name>        Get specific agent by name
  refresh           Force rebuild of agent cache
  stats             Show registry statistics

Examples:
  agent-cli.mjs list
  agent-cli.mjs match "optimize JAX code"
  agent-cli.mjs get jax-pro
  agent-cli.mjs refresh

Output:
  All commands output JSON to stdout
  Logging messages go to stderr
  Exit code 0 on success, 1 on error
`);
}

main().catch(err => {
  console.error('❌ Fatal error:', err.message);
  process.exit(1);
});
