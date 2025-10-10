# Claude Code Documentation

Central documentation hub for the Claude Code slash commands system.

## 📚 Core Documentation

### System Architecture & Design

#### [MCP Integration](MCP_INTEGRATION.md)
**Complete MCP integration system guide** - Production-ready optimization layer for Claude Code

- Three-tier knowledge hierarchy (memory-bank → serena → context7)
- Library ID caching with 40+ pre-cached libraries (82% cache hit rate)
- MCP profiles for command types (5x faster parallel loading)
- Smart triggering with pattern-based activation
- Learning system with adaptive pattern recognition
- Predictive preloading (60-80% latency reduction)
- Comprehensive monitoring and alerting
- **Performance:** 73% latency reduction, 70% cost savings
- **Deployment guides and API reference included**

#### [Agent System](AGENT_SYSTEM.md)
**Multi-agent orchestration system** - Coordinate multiple specialized agents

- Agent coordination and communication patterns
- Resource allocation strategies
- Task delegation and result aggregation
- Multi-agent workflows
- Best practices for agent orchestration

### Reference & Usage

#### [Quick Reference](QUICK_REFERENCE.md)
**Command syntax and usage guide** - Fast lookup for daily use

- Command overview and categories
- Common usage patterns
- Command syntax reference
- Best practices and tips
- Frequently used examples

### Project History

#### [Changelog](CHANGELOG.md)
**Version history and changes** - Track project evolution

- Release notes
- Feature additions
- Bug fixes
- Breaking changes
- Migration guides

## 📁 Additional Resources

### Archive
- [archive/](archive/) - Archived documentation and superseded versions
  - `MCP_IMPLEMENTATION_SUMMARY.md.OLD` - Phase 1 summary (superseded by MCP_INTEGRATION.md)
  - `MCP_PHASE2_SUMMARY.md.OLD` - Phase 2 summary (superseded by MCP_INTEGRATION.md)
  - `MCP_PHASE3_SUMMARY.md.OLD` - Phase 3 summary (superseded by MCP_INTEGRATION.md)

## 🔗 Related Documentation

### Code & Examples
- **Main command directory:** `/Users/b80985/.claude/commands/`
- **MCP integration package:** `/Users/b80985/.claude/commands/mcp_integration/`
- **MCP integration examples:** `/Users/b80985/.claude/commands/mcp_integration/examples/`
- **MCP tests:** `/Users/b80985/.claude/commands/mcp_integration/tests/`

### Configuration Files
- **MCP config:** `/Users/b80985/.claude/commands/mcp-config.yaml`
- **MCP profiles:** `/Users/b80985/.claude/commands/mcp-profiles.yaml`
- **Library cache:** `/Users/b80985/.claude/commands/library-cache.yaml`

### External Resources
- **MCP deployment guide:** See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) Section 6
- **API quick reference:** See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) Section 7
- **Performance metrics:** See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) Section 8

## 🚀 Quick Start

### For Command Users
1. Start with [Quick Reference](QUICK_REFERENCE.md) for command syntax
2. Check [Changelog](CHANGELOG.md) for recent updates
3. Explore specific features in detailed docs

### For Developers
1. Read [MCP Integration](MCP_INTEGRATION.md) for system architecture
2. Review [Agent System](AGENT_SYSTEM.md) for multi-agent patterns
3. Check deployment guides for integration
4. Explore code examples in `/mcp_integration/examples/`

### For Contributors
1. Review [Changelog](CHANGELOG.md) to understand project history
2. Read system architecture docs
3. Follow coding patterns in existing commands
4. Test thoroughly before submitting changes

---

**Last Updated:** 2025-10-04
**Documentation Version:** 1.0
