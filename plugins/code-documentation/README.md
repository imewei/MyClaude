# Code Documentation

**Version:** 2.1.0 | **Category:** documentation | **License:** MIT

Comprehensive code documentation, explanation, and generation with AST-based analysis, visual diagrams, and automated documentation workflows. Features hub-and-spoke architecture achieving 54% command file reduction with comprehensive external documentation.


## What's New in v2.1.0

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Agents (3)

All agents updated to v1.0.3 with enhanced metadata, capabilities, and trigger scenarios.

### 🔍 code-reviewer (v1.0.3)

**Status:** active | **Maturity:** 92%

Elite code review expert specializing in modern AI-powered code analysis, security vulnerabilities, performance optimization, and production reliability.

**Capabilities**:
- Security vulnerability detection (XSS, CSRF, SQL injection)
- Performance optimization analysis
- Code quality assessment with static analysis
- Architectural review and design pattern validation
- Production reliability checks
- Best practices enforcement

**Triggers**: Code reviews, security audits, performance optimization, pre-production checks

---

### 📚 docs-architect (v1.0.3)

**Status:** active | **Maturity:** 92%

Creates comprehensive technical documentation from existing codebases. Analyzes architecture, design patterns, and implementation details to produce long-form technical manuals.

**Capabilities**:
- AST-based code analysis for documentation extraction
- System architecture documentation
- API documentation generation
- Sphinx/MkDocs/Hugo documentation setup
- README and onboarding guide creation
- Documentation coverage analysis

**Triggers**: System documentation, API docs, README generation, Sphinx setup, post-refactoring updates

---

### 🎓 tutorial-engineer (v1.0.3)

**Status:** active | **Maturity:** 92%

Creates step-by-step tutorials and educational content from code. Transforms complex concepts into progressive learning experiences with hands-on examples.

**Capabilities**:
- Progressive learning path creation
- Hands-on exercise design
- Concept explanation with examples
- Common pitfall identification
- Interactive code playground setup
- Multi-level content (beginner/intermediate/advanced)

**Triggers**: Feature tutorials, onboarding docs, educational content, algorithm explanations, training materials

---

## Commands (4)

### 🔍 `/code-explain` (v1.0.3)

**Status:** active | **Maturity:** 95% | **Optimization:** 73.7% reduction (2.1.0 → 265 lines)

Detailed explanation of code structure, functionality, and design patterns with scientific computing support.

**Execution Modes**:
- **quick** (5-10 min): Simple code walkthrough, urgent explanations
- **standard** (15-25 min): Comprehensive explanation with diagrams (default)
- **comprehensive** (30-45 min): Full analysis with learning resources

**External Docs** (~1,600 lines):
- code-analysis-framework.md - AST parsing, complexity analysis
- visualization-techniques.md - Mermaid diagrams, algorithm viz
- learning-resources.md - Design patterns, tutorials
- scientific-code-explanation.md - NumPy/JAX/Julia/ML patterns

---

### 📄 `/doc-generate` (v1.0.3)

**Status:** active | **Maturity:** 95% | **Optimization:** 50.2% reduction (653 → 325 lines)

Generate comprehensive, maintainable documentation from code with AI-powered analysis.

**Execution Modes**:
- **quick** (10-15 min): README + basic API docs
- **standard** (20-30 min): Complete docs with examples (default)
- **comprehensive** (40-60 min): Full doc site with CI/CD

**External Docs** (~850 lines):
- api-documentation-templates.md - OpenAPI specs, endpoint extraction
- documentation-automation.md - CI/CD, README generation

---

### 📝 `/update-claudemd` (v1.0.3)

**Status:** active | **Maturity:** 90%

Automatically update CLAUDE.md file based on recent code changes.

**Execution Modes**:
- **quick** (5-10 min): Last 5 commits, quick updates
- **standard** (10-15 min): Last 10 commits (default)
- **force** (15-20 min): Full rebuild with complete history

---

### 📚 `/update-docs` (v1.0.3)

**Status:** active | **Maturity:** 95% | **Optimization:** 43.9% reduction (683 → 383 lines)

Comprehensively update and optimize Sphinx docs, README, and related codebase documentation with AST-based content extraction.

**Execution Modes**:
- **quick** (15-20 min): README + critical updates
- **standard** (30-45 min): Full AST analysis + docs (default)
- **comprehensive** (60-90 min): Everything + CI/CD

**External Docs** (~1,600 lines):
- ast-parsing-implementation.md - Multi-language AST extraction
- sphinx-optimization.md - Sphinx config, autodoc setup
- api-documentation-templates.md - OpenAPI templates
- documentation-automation.md - CI/CD automation

---

## Quick Start

### Installation

```bash
# Install via Claude Code marketplace
claude-code plugins install code-documentation
```

### Basic Usage

```bash
# Code review with security analysis
"Review this authentication component for security vulnerabilities"
# Invokes code-reviewer with chain-of-thought reasoning

# Generate comprehensive documentation
"Create comprehensive documentation for this microservices system"
# Invokes docs-architect with architecture analysis

# Create educational tutorial
"Create a tutorial for building a REST API with Express.js"
# Invokes tutorial-engineer with progressive learning design

# Quick code explanation
/code-explain path/to/file.js
# Detailed code explanation with design patterns

# Update project documentation
/update-docs --full --sphinx --readme
# Comprehensive documentation update
```

---

## Examples

### Example 1: React Authentication Component Security Review

**Request:** "Review this LoginForm component for security vulnerabilities and performance issues"

**code-reviewer v1.0.2 Response:**
```
## Step 1: Code Assessment & Context Understanding
- Purpose: User authentication component
- Language: React/JavaScript
- Complexity: Medium (100 lines, 3 state hooks)
- Risk Level: CRITICAL (handles user credentials)
- Deployment: Production web application
- Blast Radius: All users (authentication affects entire app)

## Step 2: Automated Analysis Execution
Running automated tools:
- ESLint: 2 warnings (unused variables)
- SonarQube: Code smells detected (complexity score: 8)
- Snyk: 1 dependency vulnerability (react-dom@17.0.1)
- npm audit: 0 critical vulnerabilities

## Step 3: Manual Code Review
Logic Analysis:
✓ Form submission properly handled
✗ Missing input validation before submission
✗ Error handling incomplete (no network error recovery)

## Step 4: Security & Performance Deep Dive

CRITICAL Security Issues:
1. XSS-Vulnerable Token Storage
   - Location: line 42: localStorage.setItem('token', response.token)
   - Risk: Accessible to JavaScript, vulnerable to XSS attacks
   - Fix: Use httpOnly cookies instead

2. Missing Input Validation
   - Location: handleSubmit function (line 30)
   - Risk: No validation before API call
   - Fix: Validate email format and password requirements

3. Missing CSRF Protection
   - Location: API call (line 38)
   - Risk: Cross-site request forgery vulnerability
   - Fix: Implement CSRF token in headers

Performance Issues:
- Unnecessary re-renders on every keystroke (no debouncing)
- Memory leak potential (missing cleanup in useEffect)

## Step 5: Actionable Feedback Generation

BLOCKING (Must fix before merge):
1. Fix XSS token storage vulnerability
   [Complete code example with httpOnly cookie implementation]

2. Add input validation
   [Complete code example with validation logic]

HIGH PRIORITY:
3. Add CSRF protection
   [Complete code example with CSRF token]

NICE-TO-HAVE:
4. Optimize re-renders
   [Complete code example with useMemo/useCallback]

## Step 6: Review Validation & Completeness Check
✓ All security areas covered (OWASP Top 10 checked)
✓ Performance analyzed (rendering, memory)
✓ Code examples provided for each issue
✓ Severity levels assigned
⚠️ TODO: Add accessibility review in follow-up

Final Summary:
- 2 BLOCKING security issues (estimated fix: 2-3 hours)
- 1 HIGH priority security issue (estimated fix: 1-2 hours)
- 3 NICE-TO-HAVE improvements (estimated fix: 1 hour)
- Total estimated effort: 4-6 hours

[~650 lines of complete code examples provided]
```

---

### Example 2: Microservices API Gateway Documentation

**Request:** "Create comprehensive documentation for our API Gateway system with service discovery and rate limiting"

**docs-architect v1.0.2 Response:**
```
## Step 1: Codebase Discovery
Analyzing structure...
- Entry point: src/gateway/server.js
- Core components: 8 identified (routing, auth, rate-limiting, service-discovery, etc.)
- Dependencies: Express.js, Consul, Redis, JWT
- Integrations: 12 backend microservices

## Step 2: Architecture Analysis
Key patterns identified:
- API Gateway pattern (centralized entry point)
- Service Registry pattern (Consul for discovery)
- Circuit Breaker pattern (Hystrix-like implementation)
- Rate Limiting (Redis-based token bucket)

Design decisions extracted:
- Why API Gateway? (centralized authentication, rate limiting, single entry point)
- Why Consul? (vs. Eureka, etcd - dynamic service registration, health checks)
- Why Redis? (vs. Memcached - atomic operations for rate limiting)

## Step 3: Documentation Planning
Target audience: 5 personas
1. Backend Developers (implementation details, API contracts)
2. Frontend Developers (API usage, authentication)
3. DevOps Engineers (deployment, scaling, monitoring)
4. Security Team (authentication, authorization, threat model)
5. Management (architecture overview, capabilities)

Planned structure: 80-90 pages
- Executive Summary (2 pages)
- Architecture Overview (15 pages)
- Design Decisions (12 pages)
- Component Deep-Dives (35 pages)
- Security Model (10 pages)
- Deployment & Operations (8 pages)
- Appendices (8 pages)

## Step 4: Content Creation (Excerpt)

# API Gateway System Documentation

## Executive Summary
This document provides comprehensive technical documentation for the API Gateway system serving as the central entry point for all client requests...

[Target Audience: Technical teams, architects, and stakeholders]

### Key Capabilities
- Unified API access point for 12 backend microservices
- JWT-based authentication with OAuth2 support
- Redis-powered rate limiting (10,000 requests/second capacity)
- Dynamic service discovery with Consul
- Circuit breaker pattern for resilience
- Comprehensive request/response logging

## Architecture Overview

### System Context
```
┌─────────────┐
│   Clients   │
│  (Web/Mobile) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      API Gateway (Port 3000)     │
│  ┌──────────┬────────────────┐  │
│  │  Auth    │ Rate Limiting  │  │
│  └──────────┴────────────────┘  │
└───────┬─────────────────────────┘
        │
   ┌────┴─────┐
   ▼          ▼
[Service Registry] [Backend Services]
```

### Design Decisions

#### Decision 1: API Gateway Pattern
**Context**: Multiple microservices needed centralized authentication and rate limiting

**Rationale**:
- Centralized authentication reduces code duplication across 12 services
- Single point for rate limiting prevents distributed state issues
- Simplified client integration (one endpoint vs. 12)

**Trade-offs**:
- ✓ Pros: Centralized control, simplified client code, better security
- ✗ Cons: Single point of failure, potential bottleneck, added latency

**Alternatives Considered**:
- Backend for Frontend (BFF): Rejected due to multiple client types
- Direct service access: Rejected due to security and rate limiting concerns

[Complete 80-90 page documentation with all sections provided]

## Step 5: Integration & Cross-Reference
- 45 internal cross-references added
- 87 code references with file:line format
- Consistent terminology (23-term glossary)
- 5 reading paths defined for different audiences

## Step 6: Validation (Self-Critique)
Completeness Check:
✓ All 8 core components documented
✓ All design decisions explained with rationale
✓ Security model comprehensive (OAuth2, JWT, RBAC)
✓ Multi-audience reading paths provided
⚠️ Performance benchmarks could be expanded (add in v2)

Accuracy Check:
✓ All code references verified
✓ Architecture diagrams match implementation
✓ Configuration examples tested

Final Assessment: 85/90 pages completed
```

---

### Example 3: WebSocket Real-Time Chat Tutorial

**Request:** "Create a beginner tutorial for building a real-time chat application with WebSockets"

**tutorial-engineer v1.0.2 Response:**
```
## Step 1: Learning Objective Definition
What learners will achieve:
- Understand WebSocket protocol and bidirectional communication
- Build complete real-time chat server and client
- Handle connection lifecycle (connect, disconnect, reconnect)
- Implement broadcasting to multiple clients
- Debug common WebSocket issues

Prerequisites:
- Node.js installed (v14+)
- Basic JavaScript knowledge (functions, callbacks)
- HTML/CSS fundamentals
- Terminal/command line comfort

Measurable Outcome: Working chat application with 3+ concurrent users

## Step 2: Concept Decomposition
Atomic concepts in learning order:
1. HTTP vs WebSocket (request-response vs persistent connection)
2. WebSocket server setup (ws library, event handlers)
3. WebSocket client connection (browser API, message sending)
4. Broadcasting pattern (maintaining client list, iterating connections)
5. Message metadata (usernames, timestamps, formatting)
6. Error handling (connection loss, server crashes, edge cases)

## Step 3: Exercise Design
Progressive hands-on exercises:
- Section 2: Modify echo server to uppercase messages
- Section 3: Add button to send messages (guided)
- Section 4: Filter messages by keyword (challenge)
- Section 5: Add emoji support (advanced)
- Final: Implement private messaging (synthesis)

## Step 4: Content Creation

# Tutorial: Build a Real-Time Chat Application with WebSockets

**Time Estimate**: 45-60 minutes
**Difficulty**: Beginner
**Final Result**: Multi-user chat with usernames and timestamps

## Section 1: Understanding WebSockets

**Key Concept**: WebSockets enable real-time, bidirectional communication

HTTP (Traditional):
- Client asks, server answers
- New connection for each request
- Higher latency (connection overhead)

WebSocket:
- Persistent connection
- Both sides can send anytime
- Lower latency (single connection)

## Section 2: Building a Minimal Echo Server

Let's start with the simplest WebSocket server (15 lines):

```javascript
// server.js
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  console.log('Client connected');

  ws.on('message', (message) => {
    console.log('Received:', message);
    ws.send(`Echo: ${message}`); // Send back to same client
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

console.log('WebSocket server running on ws://localhost:8080');
```

**Run it**: `node server.js`
**Checkpoint**: You should see "WebSocket server running..."

## Section 3: Building WebSocket Client

[HTML + JavaScript client implementation]

## Section 4: Broadcasting to All Clients

[Broadcasting pattern with client list]

## Section 5: Adding Usernames and Timestamps

[Metadata and formatting]

## Section 6: Error Handling and Reconnection

[Robust connection management]

## Common Errors and Troubleshooting

**Error 1: "WebSocket is not defined"**
- Symptom: Browser console error
- Root Cause: Older browser without WebSocket support
- Fix: Add polyfill or check for WebSocket support

[4 more common errors with complete solutions]

## Practice Exercises

**Exercise 1: Add Message Timestamps (Guided)**
Modify the client to display when each message was sent.

Hint: Use `new Date().toLocaleTimeString()`

<details>
<summary>Solution</summary>

```javascript
// In message event handler:
const timestamp = new Date().toLocaleTimeString();
const div = document.createElement('div');
div.textContent = `[${timestamp}] ${data}`;
messages.appendChild(div);
```
</details>

[4 more exercises with solutions]

## Summary
Concepts learned:
- WebSocket protocol and persistent connections
- Server-side event handling (connection, message, close)
- Client-side WebSocket API
- Broadcasting pattern for multi-client communication
- Error handling and reconnection strategies

Skills acquired:
- Build WebSocket servers with Node.js
- Create WebSocket clients with browser API
- Debug common connection issues
- Handle edge cases gracefully

Next Steps:
- Deploy to Heroku/AWS
- Add user authentication
- Implement message history with database
- Scale horizontally with Redis pub/sub

## Step 5: Error Anticipation
Identified 5 common mistakes:
- Forgetting to listen on port 8080
- Not handling client disconnections
- Sending to closed connections
- Mixing up client-side and server-side code
- Not implementing reconnection

## Step 6: Validation (Self-Critique)
Beginner Accessibility: ✓ Prerequisites clearly stated
Progressive Complexity: ✓ 6 sections building incrementally
Hands-On Practice: ✓ 5 exercises with solutions
Error-Embracing: ✓ 5 common errors documented
Measurable Outcomes: ✓ Working chat app as validation

Constitutional Principles Validated:
✓ All 5 principles met
```

---

## Key Features

### Chain-of-Thought Reasoning
All agents provide transparent, step-by-step reasoning for their outputs:
- **Systematic Analysis**: Structured thinking process ensures comprehensive coverage
- **Decision Documentation**: Rationale for choices clearly explained
- **Self-Validation**: Quality checks built into every process
- **Transparent Thinking**: Users see how conclusions are reached

### Constitutional AI Principles
Each agent has 5 core principles that guide decision-making:

**code-reviewer**:
- Security-First, Constructive Feedback, Actionable Guidance, Context-Aware Analysis, Production Reliability

**docs-architect**:
- Comprehensiveness, Progressive Disclosure, Accuracy & Precision, Audience-Aware Communication, Long-term Maintainability

**tutorial-engineer**:
- Beginner-Friendly, Progressive Complexity, Hands-On Practice, Error-Embracing, Measurable Outcomes

### Comprehensive Examples
Every agent includes production-ready examples:
- **code-reviewer**: 650+ lines of code fixes with security vulnerabilities and performance optimizations
- **docs-architect**: 80-90 page API Gateway documentation with architecture diagrams and design decisions
- **tutorial-engineer**: Complete WebSocket tutorial with 6 sections, 5 errors, 5 practice exercises

---

## Integration

### Compatible Plugins
- **cicd-automation**: Integrate code reviews into CI/CD pipelines
- **backend-development**: Documentation for API and microservices architecture
- **frontend-mobile-development**: UI component documentation and tutorials

### Slash Commands
- `/code-explain`: Detailed code explanations with design patterns
- `/update-claudemd`: Automatic CLAUDE.md updates
- `/update-docs`: Comprehensive documentation updates with Sphinx integration

---

## Documentation

### Full Documentation
For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/code-documentation.html)

### Changelog
See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.

### Agent Documentation
Each agent has detailed documentation with examples:
- [code-reviewer.md](./agents/code-reviewer.md) - Elite code review with security and performance analysis
- [docs-architect.md](./agents/docs-architect.md) - Comprehensive technical documentation creation
- [tutorial-engineer.md](./agents/tutorial-engineer.md) - Educational content and step-by-step tutorials

---

## Support

### Reporting Issues
Report issues at: https://github.com/anthropics/claude-code/issues

### Contributing
Contributions are welcome! Please see the individual agent documentation for contribution guidelines.

### License
MIT License - See [LICENSE](./LICENSE) for details

---

**Author:** Wei Chen
**Version:** 2.1.0
**Category:** Documentation
**Last Updated:** 2026-01-18
