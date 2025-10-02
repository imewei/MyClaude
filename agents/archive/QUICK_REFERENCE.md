# Claude Code Agent Quick Reference Card
**Version 1.0** | **Total Agents: 23** | **Print-Friendly Single Page**

---

## 🎯 5-Second Selection Guide

| Your Task | Recommended Agent | Alternative |
|-----------|------------------|-------------|
| **Web App Development** | fullstack-developer | systems-architect (if planning) |
| **ML Model Training** | ai-ml-specialist | neural-networks-master (architecture) |
| **AI Platform Design** | ai-systems-architect | ai-ml-specialist (if hands-on) |
| **Scientific Computing** | scientific-computing-master | jax-pro (if JAX-only) |
| **Database Design** | database-workflow-engineer | systems-architect (if architecture) |
| **CLI Tool** | command-systems-engineer | fullstack-developer (if web-like) |
| **Code Quality** | code-quality-master | (always use with any agent) |
| **Deployment** | devops-security-engineer | (always use after development) |
| **Documentation** | documentation-architect | (always use after development) |
| **Data Pipeline** | data-professional | database-workflow-engineer |
| **Quantum Computing** | advanced-quantum-computing-expert | jax-scientific-domains |
| **Physics Simulation** | Domain Specialist | scientific-computing-master |
| **Multi-Agent Coordination** | multi-agent-orchestrator | (use for 5+ agents) |

---

## 🌳 Decision Tree

```
START: What are you building?
│
├─ SOFTWARE APPLICATION
│  ├─ Need architecture? → systems-architect
│  ├─ Need full-stack? → fullstack-developer
│  ├─ CLI tool only? → command-systems-engineer
│  └─ Database-heavy? → database-workflow-engineer
│
├─ AI/ML SYSTEM
│  ├─ Training models? → ai-ml-specialist
│  ├─ Platform/infrastructure? → ai-systems-architect
│  └─ Neural architecture? → neural-networks-master
│
├─ SCIENTIFIC COMPUTING
│  ├─ Multi-language? → scientific-computing-master
│  ├─ JAX-only? → jax-pro
│  ├─ Domain-specific (Quantum/CFD/MD)? → jax-scientific-domains
│  └─ Legacy code migration? → scientific-code-adoptor
│
├─ PHYSICS/CHEMISTRY
│  ├─ Quantum computing? → advanced-quantum-computing-expert
│  ├─ Neutron scattering? → neutron-soft-matter-expert
│  ├─ X-ray scattering? → xray-soft-matter-expert
│  ├─ Correlation functions? → correlation-function-expert
│  └─ Non-equilibrium? → nonequilibrium-stochastic-expert
│
├─ DATA & VISUALIZATION
│  ├─ Data engineering? → data-professional
│  └─ Visualization/UI? → visualization-interface-master
│
├─ SUPPORT TASKS
│  ├─ Quality/Testing? → code-quality-master
│  ├─ Deployment? → devops-security-engineer
│  ├─ Documentation? → documentation-architect
│  └─ Literature review? → research-intelligence-master
│
└─ COMPLEX PROJECT (5+ agents)? → multi-agent-orchestrator
```

---

## 📚 Agent Categories at a Glance

### Engineering Core (7)
**systems-architect** - High-level design
**fullstack-developer** - Full-stack implementation
**code-quality-master** - Testing & quality
**command-systems-engineer** - CLI tools
**devops-security-engineer** - Deployment & security
**database-workflow-engineer** - Database & workflows
**documentation-architect** - Documentation

### AI/ML Core (3)
**ai-ml-specialist** - ML model development
**ai-systems-architect** - AI infrastructure
**neural-networks-master** - Neural architecture

### Scientific Computing (3)
**scientific-computing-master** - Multi-language HPC
**jax-pro** - JAX framework
**jax-scientific-domains** - Domain-specific JAX

### Domain Specialists (7)
**advanced-quantum-computing-expert** - Quantum algorithms
**correlation-function-expert** - Correlation analysis
**neutron-soft-matter-expert** - Neutron scattering
**xray-soft-matter-expert** - X-ray scattering
**nonequilibrium-stochastic-expert** - Non-equilibrium
**scientific-code-adoptor** - Legacy modernization
*(Note: Only 6 listed - 7th category has sub-specialists)*

### Support Specialists (3)
**data-professional** - Data engineering
**visualization-interface-master** - Visualization
**research-intelligence-master** - Research review
**multi-agent-orchestrator** - Multi-agent coordination

---

## ⚡ Common Workflows (Copy-Paste Ready)

### Workflow 1: Build Web App
```
1. systems-architect (2h) - Design architecture
2. database-workflow-engineer (3h) - Design schema
3. fullstack-developer (40h) - Implement
4. code-quality-master (8h) - Testing
5. devops-security-engineer (4h) - Deploy
6. documentation-architect (4h) - Document
Total: ~60h
```

### Workflow 2: ML Pipeline
```
1. data-professional (8h) - Data pipeline
2. ai-ml-specialist (20h) - Model training
3. ai-systems-architect (6h) - Infrastructure
4. devops-security-engineer (4h) - Deploy
5. documentation-architect (3h) - Document
Total: ~40h
```

### Workflow 3: Scientific Simulation
```
1. research-intelligence-master (3h) - Literature
2. scientific-computing-master (8h) - Algorithm
3. jax-pro (12h) - JAX implementation
4. visualization-interface-master (6h) - Visualization
5. documentation-architect (4h) - Document
Total: ~33h
```

---

## 🚨 Emergency Troubleshooting (Top 5 Issues)

### Issue 1: "Which agent should I use?"
**Solution**: Use decision tree above OR check agent's "When to Invoke" section

### Issue 2: "Agent doesn't fit my task exactly"
**Solution**: Pick closest match, agents are flexible. Use multi-agent if needed.

### Issue 3: "Need multiple agents, overwhelmed"
**Solution**: Use multi-agent-orchestrator for 5+ agents, otherwise sequential invocation

### Issue 4: "Validation failed"
**Solution**: Run `./validate-agents.sh` and check specific errors

### Issue 5: "Two agents seem similar"
**Solution**: Check "Differentiation from similar agents" in each agent file

---

## 🔍 Agent Differentiation Quick Lookup

| If considering... | Also consider... | Key Difference |
|-------------------|------------------|----------------|
| systems-architect | fullstack-developer | Planning vs implementation |
| ai-ml-specialist | ai-systems-architect | Model training vs infrastructure |
| ai-ml-specialist | neural-networks-master | Full ML vs pure architecture |
| scientific-computing-master | jax-pro | Multi-language vs JAX-only |
| jax-pro | jax-scientific-domains | General JAX vs domain-specific |
| fullstack-developer | command-systems-engineer | Web app vs CLI tool |
| code-quality-master | devops-security-engineer | Code quality vs infrastructure |

---

## 📖 When to Use Multiple Agents

**1-2 Agents**: Invoke directly, sequential
**3-5 Agents**: Sequential invocation, plan workflow
**5+ Agents**: Use multi-agent-orchestrator

**Always Include**:
- code-quality-master (after development)
- devops-security-engineer (for deployment)
- documentation-architect (for documentation)

---

## 🎓 Selection Best Practices

✅ **DO**:
- Read agent's "When to Invoke" section
- Check differentiation if multiple agents seem relevant
- Use AGENT_CATEGORIES.md for detailed info
- Start with closest match, adjust if needed

❌ **DON'T**:
- Overthink selection (agents are flexible)
- Use domain specialist for general tasks
- Skip quality/deployment/documentation agents
- Use multi-agent-orchestrator for simple tasks

---

## 📊 Agent Capabilities Matrix

| Capability | Primary Agents | Support Agents |
|------------|---------------|----------------|
| **Architecture** | systems-architect, ai-systems-architect | database-workflow-engineer |
| **Implementation** | fullstack-developer, ai-ml-specialist | scientific-computing-master |
| **Quality Assurance** | code-quality-master | devops-security-engineer |
| **Deployment** | devops-security-engineer | code-quality-master |
| **Documentation** | documentation-architect | research-intelligence-master |
| **Data** | data-professional | database-workflow-engineer |
| **Visualization** | visualization-interface-master | data-professional |
| **Scientific** | scientific-computing-master | jax-pro, jax-scientific-domains |
| **AI/ML** | ai-ml-specialist | neural-networks-master, ai-systems-architect |
| **Research** | research-intelligence-master | domain specialists |

---

## 🔗 Quick Links

- **Full Documentation**: AGENT_CATEGORIES.md
- **Workflows**: AGENT_COMPATIBILITY_MATRIX.md
- **Installation**: INSTALLATION_GUIDE.md
- **Validation**: `./validate-agents.sh`
- **Template**: AGENT_TEMPLATE.md

---

## 💡 Pro Tips

1. **Start broad, narrow down**: systems-architect → fullstack-developer
2. **Quality is non-negotiable**: Always use code-quality-master
3. **Documentation last**: documentation-architect after development
4. **Domain specialists are precise**: Use only if task matches exactly
5. **When in doubt, ask**: Check agent's "When to Invoke" section

---

**Version**: 1.0.0
**Last Updated**: 2025-09-29
**Total Agents**: 23 across 5 categories
**Validation**: All agents pass automated quality checks

**Need More Help?**
- Read full agent descriptions in `.claude/agents/`
- Check AGENT_CATEGORIES.md for category details
- Review AGENT_COMPATIBILITY_MATRIX.md for workflows
- Run `./validate-agents.sh` for quality checks

---

*Print this page and keep handy for quick agent selection!*