import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import AgentRegistry from '../../agent-registry.mjs';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const fixturesPath = path.join(__dirname, '../fixtures/sample-agents');
const tempCachePath = path.join(__dirname, '../fixtures/.temp-cache');

describe('AgentRegistry - Integration Tests', () => {
  let registry;

  beforeEach(async () => {
    registry = new AgentRegistry({
      agentPaths: [fixturesPath]
    });
    // Override cache paths for testing
    registry.cacheDir = tempCachePath;
    registry.cacheFile = path.join(tempCachePath, 'test-cache.json');

    // Ensure cache directory exists
    await fs.mkdir(tempCachePath, { recursive: true });
  });

  afterEach(async () => {
    // Clean up temp cache
    try {
      await fs.rm(tempCachePath, { recursive: true, force: true });
    } catch (err) {
      // Ignore errors
    }
  });

  describe('parseAgentFile', () => {
    it('should parse valid agent file', async () => {
      const agentPath = path.join(fixturesPath, 'test-minimal.md');
      const agent = await registry.parseAgentFile(agentPath);

      expect(agent).toBeDefined();
      expect(agent.name).toBe('test-minimal');
      expect(agent.description).toBe('Minimal test agent for unit testing');
      expect(agent.tools).toEqual(['Read', 'Write']);
      expect(agent.model).toBe('inherit');
      expect(agent.systemPrompt).toContain('Minimal Test Agent');
      expect(agent.keywords).toContain('minimal');
      expect(agent.triggerPatterns).toBeDefined();  // May be empty array if no triggers
    });

    it('should parse full-featured agent file', async () => {
      const agentPath = path.join(fixturesPath, 'test-full.md');
      const agent = await registry.parseAgentFile(agentPath);

      expect(agent).toBeDefined();
      expect(agent.name).toBe('test-full-featured');
      expect(agent.tools).toContain('Bash');
      expect(agent.tools).toContain('Grep');
      expect(agent.model).toBe('sonnet');
      expect(agent.keywords).toContain('testing');
      expect(agent.triggerPatterns.length).toBeGreaterThan(2);
    });

    it('should return null for malformed agent file', async () => {
      const agentPath = path.join(fixturesPath, 'test-malformed.md');
      const agent = await registry.parseAgentFile(agentPath);

      expect(agent).toBeNull();
    });

    it('should handle non-existent file', async () => {
      const agentPath = path.join(fixturesPath, 'non-existent.md');
      const agent = await registry.parseAgentFile(agentPath);

      expect(agent).toBeNull();
    });
  });

  describe('fullScan', () => {
    it('should scan directory and load all valid agents', async () => {
      await registry.fullScan();

      expect(registry.agents.size).toBeGreaterThan(0);
      expect(registry.agents.has('test-minimal')).toBe(true);
      expect(registry.agents.has('test-full-featured')).toBe(true);
    });

    it('should skip malformed agents without crashing', async () => {
      await registry.fullScan();

      // Should load valid agents but skip malformed
      expect(registry.agents.has('test-minimal')).toBe(true);
      expect(registry.agents.has('test-malformed')).toBe(false);
    });

    it('should update metadata after scan', async () => {
      await registry.fullScan();

      expect(registry.metadata.lastScan).toBeGreaterThan(0);
      expect(registry.metadata.agentCount).toBe(registry.agents.size);
    });
  });

  describe('cache operations', () => {
    it('should save and load cache', async () => {
      // Perform full scan
      await registry.fullScan();
      const agentCountBeforeSave = registry.agents.size;

      // Save cache
      await registry.saveCache();

      // Create new registry instance
      const newRegistry = new AgentRegistry({
        agentPaths: [fixturesPath]
      });
      newRegistry.cacheDir = registry.cacheDir;
      newRegistry.cacheFile = registry.cacheFile;

      // Load from cache
      const cacheValid = await newRegistry.loadFromCache();

      expect(cacheValid).toBe(true);
      expect(newRegistry.agents.size).toBe(agentCountBeforeSave);
      expect(newRegistry.agents.has('test-minimal')).toBe(true);
    });

    it('should invalidate cache on version mismatch', async () => {
      await registry.fullScan();
      await registry.saveCache();

      // Manually modify cache version
      const cacheContent = await fs.readFile(registry.cacheFile, 'utf-8');
      const cache = JSON.parse(cacheContent);
      cache.version = '0.9.0';  // Old version
      await fs.writeFile(registry.cacheFile, JSON.stringify(cache, null, 2));

      // Try to load
      const cacheValid = await registry.loadFromCache();

      expect(cacheValid).toBe(false);
    });

    it('should invalidate cache when too old', async () => {
      await registry.fullScan();
      await registry.saveCache();

      // Manually modify cache timestamp to 25 hours ago
      const cacheContent = await fs.readFile(registry.cacheFile, 'utf-8');
      const cache = JSON.parse(cacheContent);
      cache.timestamp = Date.now() - (25 * 60 * 60 * 1000);  // 25 hours ago
      await fs.writeFile(registry.cacheFile, JSON.stringify(cache, null, 2));

      // Try to load
      const cacheValid = await registry.loadFromCache();

      expect(cacheValid).toBe(false);
    });

    it('should invalidate cache on directory modification', async () => {
      await registry.fullScan();
      await registry.saveCache();

      // Manually modify dirMtimes to simulate old scan
      const cacheContent = await fs.readFile(registry.cacheFile, 'utf-8');
      const cache = JSON.parse(cacheContent);

      // Set directory mtime to a time in the past
      for (const key in cache.dirMtimes) {
        cache.dirMtimes[key] = Date.now() - (60 * 60 * 1000);  // 1 hour ago
      }

      await fs.writeFile(registry.cacheFile, JSON.stringify(cache, null, 2));

      // Try to load
      const cacheValid = await registry.loadFromCache();

      // Should invalidate because directory mtime is newer than cached mtime
      expect(cacheValid).toBe(false);
    });
  });

  describe('load() with caching', () => {
    it('should perform full scan on first load', async () => {
      const count = await registry.load();

      expect(count).toBeGreaterThan(0);
      expect(registry.agents.size).toBeGreaterThan(0);
    });

    it('should use cache on second load', async () => {
      // First load (creates cache)
      await registry.load();

      // Create new registry
      const newRegistry = new AgentRegistry({
        agentPaths: [fixturesPath]
      });
      newRegistry.cacheDir = registry.cacheDir;
      newRegistry.cacheFile = registry.cacheFile;

      // Second load (should use cache)
      const count = await newRegistry.load();

      expect(count).toBeGreaterThan(0);
      expect(newRegistry.agents.size).toBeGreaterThan(0);
    });
  });
});
