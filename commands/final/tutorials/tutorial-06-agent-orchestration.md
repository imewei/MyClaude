# Tutorial 06: Agent System Deep Dive

**Duration**: 45 minutes | **Level**: Intermediate

---

## Learning Objectives

- Master all 23 agents
- Optimize agent selection strategies
- Implement multi-agent coordination
- Create custom agents

---

## Part 1: Understanding the 23-Agent System

### Agent Categories

**Core Agents (6)**:
- Multi-Agent Orchestrator
- Command Systems Engineer
- Code Quality Master
- Documentation Architect
- Systems Architect
- DevOps Security Engineer

**Scientific Computing (8)**:
- Scientific Computing Master
- Research Intelligence Master
- JAX Pro
- Neural Networks Master
- Quantum Computing Expert
- (+ 3 domain specialists)

**Domain Specialists (9)**:
- Data Professional
- Visualization Interface Master
- Database Workflow Engineer
- Scientific Code Adopter
- (+ 5 specialized agents)

---

## Part 2: Agent Selection Strategies (15 minutes)

### Auto Selection
```bash
# Intelligent auto-selection
/optimize --agents=auto src/

# System analyzes code and selects:
# → Scientific Computing Master (detected NumPy usage)
# → Performance Agent (performance issues found)
# → Systems Architect (architecture review)
```

### Manual Selection
```bash
# Select specific agents
/optimize --agents=scientific,engineering src/

# Use all agents
/multi-agent-optimize src/ --agents=all --orchestrate
```

### Strategic Selection
```bash
# For scientific code
/optimize --agents=scientific src/simulation.py

# For web applications
/optimize --agents=engineering,quality src/api.py

# For research projects
/optimize --agents=scientific,research src/paper_code.py
```

---

## Part 3: Multi-Agent Coordination (15 minutes)

### Parallel Agent Execution
```bash
# Agents work in parallel
/multi-agent-optimize --orchestrate --parallel src/

# Execution plan:
# Phase 1 (Parallel):
#   - Scientific Computing Master analyzes algorithms
#   - Code Quality Master checks code quality
#   - Security Agent scans for vulnerabilities
#
# Phase 2 (Parallel, depends on Phase 1):
#   - Systems Architect reviews architecture
#   - Performance Agent optimizes hotspots
#
# Phase 3 (Synthesis):
#   - Multi-Agent Orchestrator synthesizes results
```

### Agent Collaboration
```bash
# Agents collaborate on complex problems
/think-ultra --agents=all --orchestrate \
  "Optimize this GPU-accelerated ML pipeline"

# Collaboration flow:
# 1. JAX Pro analyzes GPU code
# 2. Neural Networks Master reviews ML architecture
# 3. Performance Agent profiles execution
# 4. Scientific Computing Master validates numerics
# 5. Orchestrator synthesizes recommendations
```

---

## Part 4: Custom Agent Creation (15 minutes)

### Define Custom Agent
```python
from agents.core import BaseAgent

class ProjectSpecificAgent(BaseAgent):
    name = "project-expert"
    expertise = ["domain-patterns", "business-logic"]

    def analyze(self, code, context):
        findings = {
            "patterns": self.detect_patterns(code),
            "violations": self.check_rules(code),
            "suggestions": self.generate_recommendations(code)
        }
        return findings
```

### Register Agent
```bash
# Register custom agent
/agent register ./project-specific-agent.py

# Use in commands
/optimize --agents=project-expert,scientific src/
```

---

## Practice Projects

**Project 1**: Compare agent strategies on same codebase
**Project 2**: Build domain-specific agent
**Project 3**: Implement multi-agent workflow

---

## Summary

✅ 23-agent system mastered
✅ Selection strategies optimized
✅ Multi-agent coordination understood
✅ Custom agent creation learned

**Next**: [Tutorial 07: Scientific Computing →](tutorial-07-scientific-computing.md)