# Executor System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                           │
│                  (Main Entry Point)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Command Dispatcher                              │
│  • Route commands to executors or prompts                    │
│  • Load executor modules dynamically                         │
│  • Handle executor not found cases                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │   Has Executor   │      │  No Executor     │
    │   (18 commands)  │      │  (Run as Prompt) │
    └─────────┬────────┘      └──────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Base Executor Class                         │
│  • Argument parsing                                          │
│  • Execution workflow                                        │
│  • Result formatting                                         │
│  • Error handling                                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Phase 1     │  │  Phase 2     │  │  Phase 3     │
│  Critical    │  │  Quality     │  │  Advanced    │
│  (4 cmds)    │  │  (5 cmds)    │  │  (7 cmds)    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Utility Modules                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  git_utils   │  │github_utils  │  │ test_runner  │     │
│  │              │  │              │  │              │     │
│  │ • status     │  │ • issues     │  │ • pytest     │     │
│  │ • commit     │  │ • PRs        │  │ • jest       │     │
│  │ • push       │  │ • workflows  │  │ • cargo      │     │
│  │ • branch     │  │ • releases   │  │ • go test    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │code_modifier │  │ ast_analyzer │                        │
│  │              │  │              │                        │
│  │ • backup     │  │ • functions  │                        │
│  │ • modify     │  │ • classes    │                        │
│  │ • restore    │  │ • imports    │                        │
│  │ • format     │  │ • complexity │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Executor Inheritance Hierarchy

```
CommandExecutor (base_executor.py)
    ├─ Abstract Methods:
    │   ├─ get_parser() → ArgumentParser
    │   └─ execute(args: Dict) → Dict[str, Any]
    │
    ├─ Concrete Methods:
    │   ├─ run(argv: List[str]) → int
    │   ├─ format_output(result: Dict) → None
    │   └─ handle_error(error: Exception) → Dict
    │
    └─ Implementations (16 new + 2 existing):
        │
        ├─ Existing Executors (2)
        │   ├─ ThinkUltraExecutor
        │   └─ DoubleCheckExecutor
        │
        ├─ Phase 1: Critical Automation (4)
        │   ├─ CommitExecutor
        │   ├─ RunAllTestsExecutor
        │   ├─ FixGitHubIssueExecutor
        │   └─ AdoptCodeExecutor
        │
        ├─ Phase 2: Code Quality & Testing (5)
        │   ├─ FixCommitErrorsExecutor
        │   ├─ GenerateTestsExecutor
        │   ├─ CheckCodeQualityExecutor
        │   ├─ CleanCodebaseExecutor
        │   └─ RefactorCleanExecutor
        │
        └─ Phase 3: Advanced Features (7)
            ├─ OptimizeExecutor
            ├─ MultiAgentOptimizeExecutor
            ├─ CiSetupExecutor
            ├─ DebugExecutor
            ├─ UpdateDocsExecutor
            ├─ ReflectionExecutor
            └─ ExplainCodeExecutor
```

## Data Flow Architecture

### Standard Executor Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ Parse Arguments │
│  get_parser()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute Command │
│   execute()     │
└────────┬────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│ Call Utilities  │                  │ Agent           │
│                 │                  │ Orchestration   │
│ • git_utils     │                  │                 │
│ • github_utils  │                  │ • Select agents │
│ • test_runner   │                  │ • Distribute    │
│ • code_modifier │                  │ • Synthesize    │
│ • ast_analyzer  │                  │ • Coordinate    │
└────────┬────────┘                  └────────┬────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Format Result │
                │ • success     │
                │ • summary     │
                │ • details     │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Print Output  │
                │ Return Code   │
                └───────────────┘
```

### Commit Executor Flow (Example)

```
/commit --all --ai-message --push
           │
           ▼
    ┌──────────────┐
    │ Parse Args   │
    │ all: true    │
    │ ai_msg: true │
    │ push: true   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Get Status   │──────► git_utils.get_status()
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Stage Files  │──────► git_utils.add_all()
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Get Diff     │──────► git_utils.get_diff(staged=True)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Generate Msg │──────► AI analysis of diff
    └──────┬───────┘              │
           │                      ▼
           │              ┌──────────────┐
           │              │ Analyze Type │
           │              │ Analyze Scope│
           │              │ Apply Template│
           │              └──────┬───────┘
           │                     │
           ◄─────────────────────┘
           │
           ▼
    ┌──────────────┐
    │ Commit       │──────► git_utils.commit(message)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Push         │──────► git_utils.push()
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Return Result│
    │ • hash       │
    │ • files      │
    │ • success    │
    └──────────────┘
```

### Run-All-Tests with Auto-Fix Flow

```
/run-all-tests --auto-fix --coverage
           │
           ▼
    ┌──────────────┐
    │ Detect       │──────► test_runner.detect_framework()
    │ Framework    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Run Tests    │──────► test_runner.run_tests()
    │ (Attempt 1)  │
    └──────┬───────┘
           │
           ├─── Passed? ───► Return Success
           │
           ▼ Failed
    ┌──────────────┐
    │ Create       │──────► code_modifier.create_backup()
    │ Backup       │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Analyze      │
    │ Failures     │
    │              │
    │ • ImportError│
    │ • TypeError  │
    │ • AssertError│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Apply Fixes  │──────► code_modifier.modify_file()
    │              │
    │ • Add imports│
    │ • Fix types  │
    │ • Fix syntax │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Run Tests    │──────► test_runner.run_tests()
    │ (Attempt 2)  │
    └──────┬───────┘
           │
           ├─── Passed? ───► Cleanup Backup → Success
           │
           ├─── Failed? ───► Continue (max 5 attempts)
           │
           └─── No Progress? ──► Restore Backup → Failure
```

## File Organization

```
.claude/commands/executors/
│
├── Shared Utilities (5 files)
│   ├── git_utils.py              [✅ Complete]
│   ├── github_utils.py           [✅ Complete]
│   ├── test_runner.py            [✅ Complete]
│   ├── code_modifier.py          [✅ Complete]
│   └── ast_analyzer.py           [✅ Complete]
│
├── Base Infrastructure (2 files)
│   ├── base_executor.py          [✅ Existing]
│   └── command_dispatcher.py     [✅ Existing, needs update]
│
├── Existing Executors (2 files)
│   ├── think_ultra_executor.py   [✅ Complete]
│   └── double_check_executor.py  [✅ Complete]
│
├── Phase 1: Critical (4 files)
│   ├── commit_executor.py        [✅ Complete]
│   ├── run_all_tests_executor.py [✅ Complete]
│   ├── fix_github_issue_executor.py    [✅ Complete]
│   └── adopt_code_executor.py          [✅ Complete]
│
├── Phase 2: Quality (5 files)
│   ├── fix_commit_errors_executor.py   [🔴 Pending]
│   ├── generate_tests_executor.py      [🔴 Pending]
│   ├── check_code_quality_executor.py  [🔴 Pending]
│   ├── clean_codebase_executor.py      [🔴 Pending]
│   └── refactor_clean_executor.py      [🔴 Pending]
│
├── Phase 3: Advanced (7 files)
│   ├── optimize_executor.py            [🔴 Pending]
│   ├── multi_agent_optimize_executor.py[🔴 Pending]
│   ├── ci_setup_executor.py            [🔴 Pending]
│   ├── debug_executor.py               [🔴 Pending]
│   ├── update_docs_executor.py         [🔴 Pending]
│   ├── reflection_executor.py          [🔴 Pending]
│   └── explain_code_executor.py        [🔴 Pending]
│
└── Documentation (2 files)
    ├── IMPLEMENTATION_PLAN.md    [✅ Complete]
    └── ARCHITECTURE.md           [✅ This file]

Total: 27 files
   ✅ Complete: 15 files (56%)
   🔴 Pending: 12 files (44%)
```

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    All Executors                             │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────┐                           │
│              │ base_executor.py │                           │
│              └────────┬─────────┘                           │
└───────────────────────┼──────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  git_utils   │ │ github_utils │ │ test_runner  │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        │               │               │
┌───────┴───────────────┴───────────────┴───────┐
│         Used by Multiple Executors            │
├───────────────────────────────────────────────┤
│ • commit_executor                             │
│ • fix_github_issue_executor                   │
│ • fix_commit_errors_executor                  │
│ • run_all_tests_executor                      │
│ • ci_setup_executor                           │
│ • generate_tests_executor                     │
└───────────────────────────────────────────────┘

┌──────────────┐ ┌──────────────┐
│code_modifier │ │ ast_analyzer │
└──────────────┘ └──────────────┘
        │               │
        │               │
┌───────┴───────────────┴───────────────────────┐
│         Used by Code Analysis Commands        │
├───────────────────────────────────────────────┤
│ • optimize_executor                           │
│ • check_code_quality_executor                 │
│ • clean_codebase_executor                     │
│ • refactor_clean_executor                     │
│ • generate_tests_executor                     │
│ • adopt_code_executor                         │
│ • update_docs_executor                        │
│ • explain_code_executor                       │
└───────────────────────────────────────────────┘
```

## Error Handling Strategy

```
┌──────────────────────────────────────────────────┐
│          Executor Error Hierarchy                │
├──────────────────────────────────────────────────┤
│                                                  │
│  Exception                                       │
│    │                                            │
│    ├─ GitError (git_utils.py)                   │
│    │    ├─ Repository not found                 │
│    │    ├─ Nothing to commit                    │
│    │    └─ Push rejected                        │
│    │                                            │
│    ├─ GitHubError (github_utils.py)             │
│    │    ├─ API rate limit                       │
│    │    ├─ Authentication failed                │
│    │    └─ Resource not found                   │
│    │                                            │
│    ├─ ModificationError (code_modifier.py)      │
│    │    ├─ Backup failed                        │
│    │    ├─ Restore failed                       │
│    │    └─ File not found                       │
│    │                                            │
│    └─ ExecutorError (base_executor.py)          │
│         ├─ Invalid arguments                    │
│         ├─ Execution failed                     │
│         └─ Validation failed                    │
│                                                  │
└──────────────────────────────────────────────────┘

Error Handling Flow:
    try:
        result = executor.execute(args)
    except GitError as e:
        # Rollback git operations
        # Show helpful message
    except ModificationError as e:
        # Restore from backup
        # Report what failed
    except Exception as e:
        # Log unexpected error
        # Safe cleanup
        # User-friendly error message
```

## Multi-Agent Orchestration

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Orchestrator                              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Agent Selection                                     │    │
│  │ • auto: Detect optimal agents from codebase        │    │
│  │ • core: Foundational reasoning agents               │    │
│  │ • scientific: Scientific computing agents           │    │
│  │ • engineering: Software engineering agents          │    │
│  │ • all: Complete 23-agent system                     │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │ Task Distribution                                   │    │
│  │ • Parallel analysis streams                         │    │
│  │ • Sequential dependencies                           │    │
│  │ • Load balancing                                    │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │ Synthesis & Coordination                            │    │
│  │ • Cross-agent insights                              │    │
│  │ • Conflict resolution                               │    │
│  │ • Priority ranking                                  │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │ Implementation                                      │    │
│  │ • Domain-specific changes                           │    │
│  │ • Multi-track execution                             │    │
│  │ • Quality gates                                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

Used by:
  • multi_agent_optimize_executor
  • think_ultra_executor (existing)
  • double_check_executor (existing)
  • run_all_tests_executor (with --agents=all)
  • update_docs_executor (with --agents=all)
  • optimize_executor (with --agents=all)
```

## Performance Considerations

### Optimization Strategies

```
┌──────────────────────────────────────────────────┐
│          Performance Optimization                │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Caching                                      │
│     • Git status results (TTL: 1s)              │
│     • GitHub API responses (TTL: 60s)           │
│     • AST parsing results (per file hash)       │
│     • Test results (per code hash)              │
│                                                  │
│  2. Parallelization                              │
│     • Multi-agent analysis (ThreadPoolExecutor) │
│     • Test execution (framework-specific)       │
│     • File processing (ProcessPoolExecutor)     │
│                                                  │
│  3. Lazy Loading                                 │
│     • Import utilities only when needed         │
│     • Parse files on-demand                     │
│     • Load agents selectively                   │
│                                                  │
│  4. Resource Limits                              │
│     • Max concurrent agents: 10                 │
│     • Max file size for AST: 1MB                │
│     • API rate limit respect                    │
│     • Timeout for long operations: 5min         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Security Considerations

```
┌──────────────────────────────────────────────────┐
│          Security Measures                       │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Input Validation                             │
│     • Sanitize all command arguments            │
│     • Validate file paths (no traversal)        │
│     • Check git repo boundaries                 │
│                                                  │
│  2. Safe Operations                              │
│     • Never git push --force without confirm    │
│     • Create backups before modifications       │
│     • Validate before destructive ops           │
│                                                  │
│  3. Credential Handling                          │
│     • Use gh CLI auth (no direct tokens)        │
│     • No credential storage                     │
│     • Respect gitignore for secrets             │
│                                                  │
│  4. Code Execution Safety                        │
│     • No eval() or exec() of user input         │
│     • Sandboxed test execution where possible   │
│     • Limit subprocess permissions              │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                External System Integration                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Git (via git CLI)                                          │
│    • status, add, commit, push, branch, merge               │
│    • Used by: commit, fix-github-issue, fix-commit-errors   │
│                                                              │
│  GitHub (via gh CLI)                                        │
│    • issues, PRs, workflows, releases                       │
│    • Used by: fix-github-issue, fix-commit-errors, ci-setup │
│                                                              │
│  Test Frameworks                                            │
│    • pytest, Jest, Cargo, Go, Julia, CTest                  │
│    • Used by: run-all-tests, generate-tests                 │
│                                                              │
│  Code Formatters                                            │
│    • black, prettier, rustfmt                               │
│    • Used by: check-code-quality, refactor-clean            │
│                                                              │
│  Linters                                                    │
│    • pylint, flake8, eslint, mypy                          │
│    • Used by: check-code-quality, clean-codebase            │
│                                                              │
│  AI/ML Services (future)                                    │
│    • Claude API for advanced commit messages                │
│    • Code analysis and suggestions                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Complete Command Mapping

### Existing Executors (2)
| Command | Executor | Status | Category |
|---------|----------|--------|----------|
| think-ultra | ThinkUltraExecutor | ✅ Complete | Analysis |
| double-check | DoubleCheckExecutor | ✅ Complete | Verification |

### Phase 1: Critical Automation (4)
| Command | Executor | Status | Category |
|---------|----------|--------|----------|
| commit | CommitExecutor | ✅ Complete | Git |
| run-all-tests | RunAllTestsExecutor | ✅ Complete | Testing |
| fix-github-issue | FixGitHubIssueExecutor | ✅ Complete | GitHub |
| adopt-code | AdoptCodeExecutor | ✅ Complete | Modernization |

### Phase 2: Code Quality & Testing (5)
| Command | Executor | Status | Category |
|---------|----------|--------|----------|
| fix-commit-errors | FixCommitErrorsExecutor | 🔴 Pending | CI/CD |
| generate-tests | GenerateTestsExecutor | 🔴 Pending | Testing |
| check-code-quality | CheckCodeQualityExecutor | 🔴 Pending | Quality |
| clean-codebase | CleanCodebaseExecutor | 🔴 Pending | Quality |
| refactor-clean | RefactorCleanExecutor | 🔴 Pending | Quality |

### Phase 3: Advanced Features (7)
| Command | Executor | Status | Category |
|---------|----------|--------|----------|
| optimize | OptimizeExecutor | 🔴 Pending | Performance |
| multi-agent-optimize | MultiAgentOptimizeExecutor | 🔴 Pending | Optimization |
| ci-setup | CiSetupExecutor | 🔴 Pending | CI/CD |
| debug | DebugExecutor | 🔴 Pending | Debugging |
| update-docs | UpdateDocsExecutor | 🔴 Pending | Documentation |
| reflection | ReflectionExecutor | 🔴 Pending | Analysis |
| explain-code | ExplainCodeExecutor | 🔴 Pending | Documentation |

---

## Key Takeaways

1. **Modular Design**: Shared utilities eliminate code duplication
2. **Phased Implementation**: Critical features first, then quality, then advanced
3. **Consistent Patterns**: All executors follow base_executor template
4. **Safety First**: Backup/rollback for all modifications
5. **Extensible**: Easy to add new executors following established patterns
6. **Multi-Agent Ready**: Orchestration support built-in for complex operations
7. **Streamlined Scope**: 18 commands focused on core development workflows

**Status**: 56% complete (15/27 files)
**Next Step**: Complete Phase 2 quality executors (5 remaining)