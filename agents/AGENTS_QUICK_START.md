# 🤖 Claude Code Agents Quick Start & Cheatsheet

> **Complete reference for all available specialized agents in Claude Code**

## 🚀 **Quick Start**

### Basic Agent Usage
```bash
# Launch a single agent
Task: agent-name "Your task description here"

# Launch multiple agents in parallel
Task: agent-1 "Task 1" + agent-2 "Task 2"

# Common workflow
Task: code-reviewer "Review my authentication system" + test-automator "Create comprehensive tests"
```

### Best Practices
- **Be specific** in task descriptions
- **Combine agents** for complex workflows
- **Use proactively** - agents work best when given clear objectives
- **Check agent descriptions** to match the right expert to your task

---

## 🎯 **Development & Engineering Agents**

### **Frontend Development**
| Agent | Use When | Example |
|-------|----------|---------|
| `frontend-developer` | Building React components, UI features, scalable frontend | "Create a responsive dashboard component with TypeScript" |
| `ui-designer` | Designing interfaces, design systems, visual components | "Design a modern login form with accessibility features" |
| `ux-designer` | User experience optimization, usability testing, interaction design | "Improve the onboarding flow for better user engagement" |

### **Backend Development**
| Agent | Use When | Example |
|-------|----------|---------|
| `backend-developer` | APIs, microservices, server-side logic | "Build a REST API for user management with authentication" |
| `fullstack-developer` | End-to-end features, database to UI integration | "Implement complete user registration with email verification" |
| `api-designer` | API architecture, REST/GraphQL design, documentation | "Design a scalable API structure for e-commerce platform" |

### **Programming Language Experts**
| Agent | Use When | Example |
|-------|----------|---------|
| `python-expert` | Python development, optimization, modern features | "Optimize this data processing script using Python 3.11 features" |
| `rust-pro` | Rust development, async patterns, memory safety | "Build a high-performance HTTP server in Rust with async" |
| `cpp-pro` | Modern C++ development, performance optimization | "Refactor this C++ code to use modern C++20 features" |
| `c-expert` | Systems programming, embedded development | "Optimize this C function for embedded system constraints" |

### **Scientific Computing & AI/ML**
| Agent | Use When | Example |
|-------|----------|---------|
| `ml-engineer` | Complete ML lifecycle, model deployment, production | "Deploy a recommendation system with monitoring and scaling" |
| `pytorch-expert` | Deep learning, neural networks, GPU acceleration | "Build a CNN for image classification with data augmentation" |
| `scikit-learn-expert` | Traditional ML, model selection, evaluation | "Create a customer churn prediction model with hyperparameter tuning" |
| `numpy-expert` | Array operations, numerical computing optimization | "Optimize these matrix operations for better performance" |
| `numerical-computing-expert` | Mathematical computation, scientific algorithms | "Implement numerical solver for differential equations" |
| `visualization-expert` | Data visualization, scientific plotting | "Create interactive dashboard for real-time data visualization" |
| `statistics-expert` | Statistical analysis, hypothesis testing | "Perform A/B test analysis with statistical significance" |
| `gpu-computing-expert` | CUDA programming, parallel computing | "Accelerate this computation using GPU parallelization" |

---

## 🔧 **Infrastructure & Operations Agents**

### **DevOps & Infrastructure**
| Agent | Use When | Example |
|-------|----------|---------|
| `devops-engineer` | CI/CD, automation, infrastructure management | "Set up complete DevOps pipeline with automated testing" |
| `sre-engineer` | System reliability, monitoring, operational excellence | "Implement SLOs and monitoring for production system" |
| `deployment-engineer` | Release automation, deployment strategies | "Create blue-green deployment pipeline with rollback" |
| `terraform-engineer` | Infrastructure as code, cloud provisioning | "Design Terraform modules for multi-environment AWS setup" |
| `build-engineer` | Build optimization, compilation strategies | "Optimize build pipeline to reduce CI/CD time by 50%" |

### **Security & Compliance**
| Agent | Use When | Example |
|-------|----------|---------|
| `security-engineer` | DevSecOps, cloud security, compliance | "Implement security scanning in CI/CD pipeline" |
| `security-auditor` | Security assessments, compliance validation | "Conduct comprehensive security audit for web application" |
| `penetration-tester` | Ethical hacking, vulnerability assessment | "Perform penetration testing on API endpoints" |
| `compliance-auditor` | Regulatory compliance, GDPR, HIPAA | "Ensure GDPR compliance for user data processing" |

### **Monitoring & Performance**
| Agent | Use When | Example |
|-------|----------|---------|
| `performance-monitor` | System metrics, real-time monitoring | "Set up comprehensive monitoring for microservices" |
| `performance-engineer` | System optimization, bottleneck identification | "Identify and fix performance bottlenecks in web app" |
| `incident-responder` | Incident management, rapid resolution | "Create incident response plan for production outages" |

---

## 🧪 **Quality Assurance & Testing Agents**

### **Testing & QA**
| Agent | Use When | Example |
|-------|----------|---------|
| `qa-expert` | Quality assurance strategy, test planning | "Design comprehensive QA strategy for mobile app" |
| `test-automator` | Test automation, CI/CD integration | "Create automated test suite with 90% coverage" |
| `accessibility-tester` | WCAG compliance, inclusive design | "Audit website for accessibility compliance" |

### **Code Quality & Review**
| Agent | Use When | Example |
|-------|----------|---------|
| `code-reviewer` | Code quality assessment, security review | "Review this authentication system for security issues" |
| `refactoring-specialist` | Code improvement, design patterns | "Refactor legacy codebase to modern architecture" |
| `debugger` | Complex debugging, root cause analysis | "Debug intermittent memory leak in production system" |

---

## 📚 **Documentation & Knowledge Agents**

### **Documentation**
| Agent | Use When | Example |
|-------|----------|---------|
| `documentation-expert` | Technical documentation, API docs | "Create comprehensive API documentation with examples" |
| `api-documenter` | OpenAPI/Swagger, interactive docs | "Generate interactive API documentation portal" |
| `tutorial-engineer` | Educational content, step-by-step guides | "Create tutorial series for new developers" |

### **Knowledge Management**
| Agent | Use When | Example |
|-------|----------|---------|
| `knowledge-synthesizer` | Extract insights, build collective intelligence | "Analyze team's knowledge gaps and create learning plan" |
| `research-analyst` | Information gathering, market research | "Research competitor APIs and best practices" |

---

## 🛠️ **Specialized Development Tools**

### **CLI & Tooling**
| Agent | Use When | Example |
|-------|----------|---------|
| `cli-developer` | Command-line tools, developer utilities | "Build CLI tool for database migrations" |
| `tooling-engineer` | Developer productivity tools | "Create VS Code extension for team workflows" |
| `command-creator` | Claude Code custom commands | "Create custom slash command for deployment" |

### **Database & Storage**
| Agent | Use When | Example |
|-------|----------|---------|
| `postgres-pro` | PostgreSQL optimization, administration | "Optimize PostgreSQL queries and setup replication" |
| `database-optimizer` | Query optimization, performance tuning | "Optimize slow database queries and indexing strategy" |

### **Git & Version Control**
| Agent | Use When | Example |
|-------|----------|---------|
| `git-workflow-manager` | Git workflows, branching strategies | "Design Git workflow for 20-person development team" |
| `github-actions-expert` | GitHub Actions, CI/CD workflows | "Create GitHub Actions workflow with security scanning" |

---

## 🔍 **Analysis & Optimization Agents**

### **System Analysis**
| Agent | Use When | Example |
|-------|----------|---------|
| `architect-reviewer` | Architecture validation, design review | "Review microservices architecture for scalability" |
| `dx-optimizer` | Developer experience, build optimization | "Improve developer workflow and reduce build times" |
| `legacy-modernizer` | Legacy system migration, modernization | "Plan migration strategy from monolith to microservices" |

### **Error & Issue Management**
| Agent | Use When | Example |
|-------|----------|---------|
| `error-detective` | Complex error analysis, distributed debugging | "Investigate cascading failures in microservices" |
| `error-coordinator` | Error handling, system resilience | "Design fault-tolerant error handling strategy" |

### **Search & Discovery**
| Agent | Use When | Example |
|-------|----------|---------|
| `search-specialist` | Advanced search, information retrieval | "Find all instances of deprecated API usage in codebase" |
| `general-purpose` | Complex research, multi-step tasks | "Research and implement OAuth2 with PKCE flow" |

---

## 🤖 **AI & Language Model Agents**

### **LLM & AI**
| Agent | Use When | Example |
|-------|----------|---------|
| `llm-architect` | LLM deployment, optimization | "Deploy and scale LLM inference service" |
| `nlp-engineer` | Natural language processing, transformers | "Build text classification system with BERT" |
| `prompt-engineer` | Prompt optimization, LLM interactions | "Optimize prompts for better AI assistant responses" |
| `mcp-developer` | Model Context Protocol development | "Build MCP server for custom tool integration" |

---

## 🎯 **Project Management & Coordination**

### **Workflow & Coordination**
| Agent | Use When | Example |
|-------|----------|---------|
| `multi-agent-coordinator` | Complex workflows, agent orchestration | "Coordinate deployment across multiple services" |
| `agent-organizer` | Team assembly, workflow optimization | "Organize development team for new feature delivery" |
| `task-distributor` | Work allocation, load balancing | "Distribute tasks across development team efficiently" |
| `workflow-orchestrator` | Process design, business workflows | "Design approval workflow for code changes" |

### **Context & State Management**
| Agent | Use When | Example |
|-------|----------|---------|
| `context-manager` | Information storage, state synchronization | "Manage shared state across distributed system" |
| `trend-analyst` | Pattern identification, forecasting | "Analyze development trends and predict technology needs" |

---

## 🛠️ **Configuration & Setup Agents**

### **Environment Setup**
| Agent | Use When | Example |
|-------|----------|---------|
| `statusline-setup` | Configure Claude Code status line | "Customize status line for project information" |
| `output-style-setup` | Create Claude Code output styles | "Create custom output style for code reviews" |
| `dependency-manager` | Package management, version conflicts | "Resolve dependency conflicts and security issues" |

### **Quality Control**
| Agent | Use When | Example |
|-------|----------|---------|
| `pre-commit-fixer` | Code quality, pre-commit hooks | "Fix all linting and formatting issues before commit" |

---

## 🎯 **Common Workflows**

### **Full-Stack Feature Development**
```bash
Task: frontend-developer "Build user dashboard component" + backend-developer "Create dashboard API endpoints" + test-automator "Create comprehensive tests"
```

### **Code Quality & Security Review**
```bash
Task: code-reviewer "Review authentication system" + security-auditor "Security assessment" + refactoring-specialist "Improve code structure"
```

### **ML Model Development**
```bash
Task: ml-engineer "Build recommendation model" + data-engineer "Create data pipeline" + performance-engineer "Optimize inference speed"
```

### **Infrastructure & Deployment**
```bash
Task: devops-engineer "Setup CI/CD pipeline" + terraform-engineer "Infrastructure as code" + security-engineer "Security scanning"
```

### **Documentation & Knowledge**
```bash
Task: documentation-expert "API documentation" + tutorial-engineer "Developer tutorials" + api-documenter "Interactive docs"
```

---

## 🚀 **Pro Tips**

### **Agent Selection Guide**
- **New Features:** `frontend-developer`, `backend-developer`, `fullstack-developer`
- **Code Quality:** `code-reviewer`, `refactoring-specialist`, `debugger`
- **Testing:** `qa-expert`, `test-automator`, `accessibility-tester`
- **Performance:** `performance-engineer`, `gpu-computing-expert`, `database-optimizer`
- **Documentation:** `documentation-expert`, `api-documenter`, `tutorial-engineer`
- **Infrastructure:** `devops-engineer`, `sre-engineer`, `terraform-engineer`
- **Security:** `security-engineer`, `penetration-tester`, `compliance-auditor`
- **AI/ML:** `ml-engineer`, `pytorch-expert`, `scikit-learn-expert`
- **Analysis:** `architect-reviewer`, `error-detective`, `research-analyst`

### **Best Practices**
1. **Be Specific:** Provide detailed task descriptions
2. **Combine Agents:** Use multiple agents for complex workflows
3. **Use Proactively:** Agents work best with clear objectives
4. **Match Expertise:** Choose the right agent for your specific need
5. **Parallel Execution:** Launch multiple agents simultaneously for efficiency

### **Common Patterns**
- **Review + Fix:** `code-reviewer` → `refactoring-specialist`
- **Build + Test:** `frontend-developer` → `test-automator`
- **Deploy + Monitor:** `deployment-engineer` → `performance-monitor`
- **Research + Implement:** `research-analyst` → `domain-expert`
- **Design + Build:** `architect-reviewer` → `implementation-expert`

---

## 📝 **Quick Reference**

### **When to Use Which Agent**
- **🏗️ Building:** `frontend-developer`, `backend-developer`, `fullstack-developer`
- **🔍 Analyzing:** `code-reviewer`, `architect-reviewer`, `error-detective`
- **🧪 Testing:** `qa-expert`, `test-automator`, `accessibility-tester`
- **⚡ Optimizing:** `performance-engineer`, `dx-optimizer`, `database-optimizer`
- **📚 Documenting:** `documentation-expert`, `tutorial-engineer`, `api-documenter`
- **🛡️ Securing:** `security-engineer`, `penetration-tester`, `compliance-auditor`
- **🚀 Deploying:** `devops-engineer`, `deployment-engineer`, `sre-engineer`
- **🤖 AI/ML:** `ml-engineer`, `pytorch-expert`, `nlp-engineer`

### **Emergency Agents**
- **🚨 Production Issues:** `incident-responder`, `error-detective`, `debugger`
- **🔧 Quick Fixes:** `pre-commit-fixer`, `refactoring-specialist`
- **📊 Analysis:** `performance-monitor`, `error-coordinator`

---

*Save this file and refer to it when choosing the right agent for your task!*