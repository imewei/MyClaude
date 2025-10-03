# Glossary

Comprehensive glossary of terms used in the Claude Code Command Executor Framework.

## A

**Agent**
A specialized AI persona with expertise in specific domains. The framework has 23 agents covering scientific computing, engineering, quality, and specialized domains.

**Agent Capability**
A specific technical skill or knowledge area that an agent possesses (e.g., code analysis, performance optimization, ML/AI).

**Agent Category**
Grouping of agents by primary domain: orchestration, scientific, engineering, quality, domain, scientific-domain.

**Agent Coordination**
The process of managing multiple agents working together on a task, including task assignment, dependency management, and result synthesis.

**Agent Orchestration**
Advanced coordination of multiple agents with features like parallel execution, load balancing, and conflict resolution. Enabled with `--orchestrate` flag.

**Agent Profile**
Complete specification of an agent including name, capabilities, specializations, languages, frameworks, and priority.

**Agent Selection**
The process of choosing which agents to use for a task. Can be automatic (`--agents=auto`) or explicit (`--agents=scientific`).

**AST (Abstract Syntax Tree)**
A tree representation of source code structure used for deep code analysis, particularly in `clean-codebase` command.

**Auto-Fix**
Automatic application of code fixes without manual intervention. Used with `--auto-fix` flag.

## B

**Backup**
Automatic creation of code backup before modifications. Enabled with `--backup` flag. Stored in `~/.claude/backups/`.

**Baseline**
Initial state of code before optimizations or modifications, used for comparison and validation.

**Benchmark**
Performance measurement used to compare code before and after optimizations.

**Breakthrough Mode**
Advanced analysis mode that discovers novel optimization patterns through cross-domain agent collaboration. Enabled with `--breakthrough` flag.

## C

**Cache**
Multi-level caching system with three levels:
- AST cache (24-hour TTL)
- Analysis cache (7-day TTL)
- Agent cache (7-day TTL)

**Capability Matching**
Algorithm for matching agent capabilities to task requirements, weighted at 40% in the intelligent agent matcher.

**CI/CD (Continuous Integration/Continuous Deployment)**
Automated testing and deployment pipeline. Set up with `/ci-setup` command.

**Command Executor**
The base class that implements the execution pipeline for all commands.

**Context**
Execution context containing command arguments, work directory, agent selection, and execution flags.

## D

**Dead Code**
Code that is never executed or is unreachable. Detected and removed by `/clean-codebase --dead-code`.

**Dry-Run**
Preview mode that shows what would happen without making actual changes. Enabled with `--dry-run` flag. Always recommended before first execution.

**Duplicate Code**
Identical or similar code blocks that can be consolidated. Detected with `/clean-codebase --duplicates`.

## E

**Execution Pipeline**
Six-phase command execution process:
1. Initialization
2. Validation
3. Pre-execution
4. Execution
5. Post-execution
6. Finalization

**Execution Result**
Standardized output from command execution including success status, duration, summary, details, warnings, and errors.

## F

**Framework**
The Claude Code Command Executor Framework - the complete system including 14 commands and 23-agent system.

## G

**GPU Optimization**
Optimization for GPU-accelerated computing, particularly for JAX and deep learning frameworks. Handled by `jax-pro` agent.

## H

**Hierarchical Execution**
Orchestration pattern where an orchestrator agent coordinates specialized agent teams for complex workflows.

## I

**Implement**
Flag (`--implement`) that applies recommendations automatically rather than just reporting them.

**Intelligent Selection**
Enhanced auto-selection mode using deeper codebase analysis and pattern recognition. Enabled with `--intelligent` flag.

**Interactive Mode**
Mode that prompts for confirmation before each change. Enabled with `--interactive` flag. Recommended for critical files.

## J

**JAX**
Google's library for high-performance numerical computing and machine learning. Supported by `jax-pro` agent.

## L

**Load Balancing**
Distribution of tasks across agents to optimize resource usage and execution time.

## M

**Multi-Agent System**
System where multiple agents collaborate on tasks through coordination and communication.

**Multi-Level Cache**
Three-tier caching system for AST, analysis, and agent results with different TTLs.

## O

**Orchestration**
Advanced multi-agent coordination with parallel execution, load balancing, and conflict resolution. Enabled with `--orchestrate`.

**Orchestrator Agent**
Special agent (`multi-agent-orchestrator`) that coordinates workflows across all 23 agents.

## P

**Parallel Execution**
Simultaneous execution of agents or tasks. Enabled with `--parallel` flag.

**Performance Profiling**
Analysis of code execution time, memory usage, and bottlenecks. Enabled with `--profile` flag.

**Personal Agent System**
The 23-agent ecosystem providing specialized expertise across all development domains.

**Priority**
Agent ranking (1-10) used in selection decisions. Higher priority agents are preferred when multiple agents match requirements.

## Q

**Quality Score**
Numerical assessment of code quality (0-100) based on multiple factors including complexity, style, documentation, and best practices.

## R

**Rollback**
Ability to revert code changes if something goes wrong. Enabled with `--rollback` flag.

## S

**Safety Features**
Framework features ensuring safe code modification:
- Dry-run mode
- Automatic backups
- Rollback capability
- Interactive confirmation
- Validation

**Scientific Computing**
Numerical and computational science applications. Supported by 8 specialized scientific agents.

**Shared Knowledge**
Central repository where agents store and share discoveries during multi-agent execution.

**Specialization**
Specific area of expertise within an agent's capabilities (e.g., "GPU optimization", "quantum algorithms").

**Synthesis**
Process of combining results from multiple agents into a coherent final output.

## T

**Task**
Unit of work assigned to an agent with specific goals, context, and dependencies.

**TTL (Time To Live)**
Cache expiration time. Different for each cache level (24 hours for AST, 7 days for analysis/agent).

## U

**Ultrathink**
Advanced analysis mode using sophisticated reasoning and cross-file analysis. Enabled with `--analysis=ultrathink`.

**Unused Imports**
Import statements that aren't used in the code. Detected and removed by `/clean-codebase --imports`.

## V

**Validation**
Verification of prerequisites, arguments, and results to ensure correctness and safety.

**Validation Engine**
Component that validates execution context against predefined rules before command execution.

## W

**Workflow**
Sequence of commands for accomplishing a specific goal (e.g., quality improvement, performance optimization).

## Common Abbreviations

| Abbreviation | Full Term |
|--------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AST | Abstract Syntax Tree |
| CI | Continuous Integration |
| CD | Continuous Deployment |
| GPU | Graphics Processing Unit |
| HPC | High-Performance Computing |
| JAX | Just After eXecution (Google's ML library) |
| ML | Machine Learning |
| TTL | Time To Live |
| UX | User Experience |

## Command-Specific Terms

### check-code-quality
- **Auto-fix**: Automatic fixing of code quality issues
- **Quality score**: Numerical assessment (0-100)
- **Severity levels**: HIGH, MEDIUM, LOW

### optimize
- **Profiling**: Performance measurement
- **Hot spot**: Performance bottleneck
- **Complexity**: Algorithmic complexity (O(n), O(n²), etc.)
- **Category**: Optimization type (algorithm, memory, io, concurrency)

### clean-codebase
- **AST analysis**: Abstract Syntax Tree-based code analysis
- **Dead code**: Unreachable or unused code
- **Duplicate**: Identical or similar code blocks
- **Unused import**: Import statement not referenced in code

### generate-tests
- **Coverage**: Percentage of code tested
- **Test type**: unit, integration, performance
- **Framework**: Testing library (pytest, jest, junit)

### run-all-tests
- **Benchmark**: Performance measurement
- **Coverage report**: Test coverage analysis
- **Auto-fix**: Automatic test fixing

## Flag Reference

### Universal Flags

| Flag | Purpose |
|------|---------|
| `--dry-run` | Preview without changes |
| `--agents=<type>` | Select agent group |
| `--backup` | Create backup |
| `--rollback` | Enable rollback |
| `--interactive` | Confirm each change |
| `--parallel` | Parallel execution |
| `--validate` | Validate results |

### Analysis Flags

| Flag | Purpose |
|------|---------|
| `--profile` | Performance profiling |
| `--detailed` | Detailed analysis |
| `--analysis=<level>` | Analysis depth |
| `--report` | Generate report |

### Modification Flags

| Flag | Purpose |
|------|---------|
| `--implement` | Apply changes |
| `--auto-fix` | Automatic fixes |
| `--backup` | Create backup |

### Agent Flags

| Flag | Purpose |
|------|---------|
| `--agents=auto` | Smart selection |
| `--agents=core` | 5 core agents |
| `--agents=all` | All 23 agents |
| `--orchestrate` | Enable orchestration |
| `--intelligent` | Enhanced selection |

## See Also

- **[Command Reference](command-options.md)** - All command options
- **[Agent Capabilities](agent-capabilities.md)** - Agent capability matrix
- **[Configuration](configuration.md)** - Configuration options

---

**Looking for something?** → Use Ctrl+F / Cmd+F to search this page