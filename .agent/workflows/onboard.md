---
description: Orchestrate complete onboarding for new team members with customized
  30/60/90 day plans
triggers:
- /onboard
- orchestrate complete onboarding for
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<role> [--level=junior|mid|senior] [--location=remote|hybrid|onsite]`
The agent should parse these arguments from the user's request.

# New Team Member Onboarding

Orchestrate complete onboarding experience from pre-arrival through first 90 days.

## Context

$ARGUMENTS

Parse: Role title, level, start date, location, team context, technical requirements

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 15-20 min | Day 1 checklist + Week 1 plan |
| Standard (default) | 30-45 min | Complete 30/60/90 day plan |
| Comprehensive | 60-90 min | Full plan + buddy program + metrics |

---

## Phase 1: Pre-Onboarding

### Access & Accounts
- Create accounts (email, Slack, GitHub, AWS)
- Configure SSO and 2FA
- Ship hardware with tracking
- Generate credentials and password manager setup

### Documentation
- Compile role-specific docs package
- Update team roster and org chart
- Prepare personalized checklist
- Welcome packet (handbook, benefits)

---

## Phase 2: Day 1 Orientation

### Morning
| Time | Activity |
|------|----------|
| 30 min | Manager welcome and role expectations |
| 45 min | Company mission, values, culture |
| 30 min | Team introductions |
| 30 min | First-week schedule review |

### Afternoon
| Time | Activity |
|------|----------|
| 90 min | IT-guided laptop setup |
| 30 min | Password manager and security tools |
| 30 min | Slack and communication setup |
| 45 min | HR paperwork and benefits |

---

## Phase 3: Week 1 (Codebase Immersion)

### Repository Orientation
- Architecture overview with tech lead
- Development workflow and branching
- Code style guides
- Testing philosophy

### First Contributions
- Identify "good first issues"
- Pair programming on simple fix
- Submit first PR with buddy guidance
- First code review participation

---

## Phase 4: Milestone Tracking

### 30-Day Checkpoint

| Metric | Target |
|--------|--------|
| Commits merged | 10+ |
| Training modules | 100% complete |
| Bug fix deployed | At least 1 |

### 60-Day Checkpoint

| Metric | Target |
|--------|--------|
| Feature shipped | 1 small feature |
| Code reviews | Actively giving feedback |
| On-call shadow | Complete |

### 90-Day Checkpoint

| Metric | Target |
|--------|--------|
| Independent projects | Leading 1+ |
| Mentoring | Helping newer members |
| Process improvements | 1+ proposed |

---

## Phase 5: Team Integration

### Buddy System
| Week | Frequency | Focus |
|------|-----------|-------|
| 1 | Daily | Questions, pair programming |
| 2-3 | 3x/week | Code review, architecture |
| 4 | 2x/week | Project collaboration |
| 5-8 | Weekly | Career development |

### Communication Norms
- Slack etiquette and channel purposes
- Meeting culture and documentation
- Async expectations and core hours
- Escalation paths

---

## Phase 6: Learning Path

### Technical
- Domain-specific courses
- Internal tech talks library
- Recommended reading
- Hands-on labs

### Product
- Product demos and user journeys
- Customer personas
- Roadmap and vision
- Feature flag experiments

---

## Role-Specific Plans

| Role | Week 1 Focus | 30-Day Goal | 90-Day Goal |
|------|--------------|-------------|-------------|
| Software Engineer | Dev setup, first PR | 10+ commits, training complete | Independent features |
| Senior/Lead | Architecture deep-dive | Technical assessment | Technical roadmap |
| Remote | Virtual integration, async norms | Timezone-aware workflow | Full team collaboration |

---

## Feedback & Metrics

### Collection
- Weekly pulse surveys (5 questions)
- Buddy feedback forms
- Manager 1:1 structured questions

### KPIs
| Metric | Target |
|--------|--------|
| Time to first commit | < 3 days |
| Time to first deploy | < 2 weeks |
| Integration satisfaction | > 8/10 |

---

## Success Criteria

- ✅ Pre-boarding complete before Day 1
- ✅ Day 1 agenda executed
- ✅ Week 1 codebase orientation done
- ✅ Buddy assigned and active
- ✅ 30/60/90 day milestones defined
- ✅ Feedback loops established

---

## Best Practices

1. **Customize** based on role, seniority, team needs
2. **Document** artifacts for future onboarding
3. **Measure** metrics and gather feedback
4. **Prioritize connection** over pure technical skills
5. **Maintain momentum** with daily engagement
