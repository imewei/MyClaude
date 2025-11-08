# Onboarding Playbook

> **Reference**: Comprehensive onboarding strategies, role-specific plans, and 30-60-90 day milestones

## Pre-Boarding (2 Weeks Before Start)

### IT & Access Setup
- [ ] Create company email account
- [ ] Set up Slack/Teams workspace access
- [ ] Create GitHub/GitLab account
- [ ] Provision AWS/Cloud access with appropriate IAM roles
- [ ] Set up VPN credentials
- [ ] Configure 2FA for all accounts
- [ ] Send password manager invitation (1Password/LastPass)
- [ ] Order hardware (laptop, monitors, keyboard, mouse)
- [ ] Prepare development environment setup guide

### Documentation Package
- [ ] Employee handbook
- [ ] Engineering team handbook
- [ ] Code of conduct
- [ ] Git workflow documentation
- [ ] Architecture overview documents
- [ ] API documentation links
- [ ] Team roster with photos and roles
- [ ] Office/remote work policies
- [ ] Benefits enrollment guide

### Workspace Preparation
**For Remote**:
- [ ] Home office setup checklist sent
- [ ] Home office stipend amount confirmed
- [ ] Shipping address verified for equipment
- [ ] Timezone and working hours documented

**For On-site**:
- [ ] Desk assigned and labeled
- [ ] Access badge ordered
- [ ] Parking access arranged
- [ ] Desk supplies ordered

### Team Communication
- [ ] Announcement in team channel (1 week before)
- [ ] Welcome video from team members recorded
- [ ] Buddy/mentor assigned
- [ ] Manager 1:1 scheduled for Day 1
- [ ] Team lunch scheduled for Week 1

---

## Day 1: Welcome & Orientation

### Morning (9am-12pm)

**9:00am - Welcome Meeting (Manager)**
- Welcome and introductions
- Review first week agenda
- Set expectations and goals
- Answer initial questions
- Tour (if on-site) or virtual office tour

**10:00am - IT Setup (Self-paced with IT support)**
- Laptop and monitor setup
- Email client configuration
- Slack/Teams installation and channels
- VPN setup and testing
- Password manager setup
- 2FA configuration for all services
- GitHub/GitLab access verification
- Development tools installation

**11:00am - Development Environment Setup**
```bash
# Clone main repository
git clone git@github.com:company/main-repo.git
cd main-repo

# Run setup script
./scripts/setup.sh

# Verify installation
npm test
docker-compose up

# Expected: All tests passing, services running
```

### Afternoon (1pm-5pm)

**1:00pm - Team Introduction Meeting**
- Meet entire team (15-20 minutes each)
- Learn about each person's role
- Understand team structure
- Get communication preferences

**2:30pm - Codebase Overview (Tech Lead)**
- Architecture walkthrough
- Key repositories and their purposes
- Technology stack explanation
- Development workflow overview
- Deployment process overview
- Testing strategy explanation

**4:00pm - First Task Assignment (Buddy)**
- Assign "good first issue" ticket
- Pair programming session setup
- Code review process explanation
- How to ask for help

---

## Week 1: Foundations

### Learning Objectives
- Understand company mission and product
- Set up complete development environment
- Make first code contribution
- Learn team communication patterns
- Understand Git workflow

### Daily Schedule Template

**Daily Standup (9:30am)**
- What you worked on yesterday
- What you're working on today
- Any blockers

**Pairing Sessions (2-3 hours/day)**
- Day 2: Codebase exploration with buddy
- Day 3: Work on first ticket together
- Day 4: Submit first PR with guidance
- Day 5: Review process and feedback

**Learning Time (2 hours/day)**
- Read documentation
- Watch recorded tech talks
- Explore codebase independently
- Complete training modules

### Week 1 Milestones
- [ ] Development environment fully functional
- [ ] Completed 1-2 "good first issue" tickets
- [ ] Submitted first PR and received feedback
- [ ] Attended all team meetings
- [ ] Met with all team members 1:1
- [ ] Read key documentation
- [ ] Understand deployment process

### Good First Issues Criteria
- Well-documented with clear acceptance criteria
- Small scope (2-4 hours)
- Touches core codebase but low risk
- Good learning opportunity
- Has automated tests
- Recent and relevant

Example good first issues:
```
- Add unit tests for utility functions
- Update outdated documentation
- Refactor component to use hooks
- Add error handling to API endpoint
- Implement new UI component from design
```

---

## Week 2-4: Ramp Up

### Learning Objectives
- Contribute independently to non-critical features
- Understand testing and CI/CD practices
- Learn debugging and troubleshooting
- Participate actively in code reviews
- Understand monitoring and alerting

### Weekly Goals

**Week 2**:
- [ ] Work on 2-3 small features independently
- [ ] Review 3-5 PRs from teammates
- [ ] Participate in sprint planning
- [ ] Shadow on-call engineer (if applicable)
- [ ] Complete security training

**Week 3**:
- [ ] Take ownership of medium-complexity feature
- [ ] Write feature documentation
- [ ] Present work in team demo
- [ ] Conduct code review (with mentor oversight)
- [ ] Contribute to technical design discussion

**Week 4**:
- [ ] Complete first significant feature end-to-end
- [ ] Submit feature for product review
- [ ] Fix bugs from QA testing
- [ ] Deploy feature to production
- [ ] Monitor feature performance

### Skill Development Focus

**Technical Skills**:
- Git workflow mastery
- Testing (unit, integration, E2E)
- Debugging with DevTools
- API development patterns
- Database query optimization
- Frontend component architecture

**Soft Skills**:
- Asking effective questions
- Time estimation
- Task prioritization
- Collaboration in PRs
- Documentation writing
- Meeting participation

---

## 30-60-90 Day Plan

### Days 1-30: Foundation & Learning

**Technical Goals**:
- [ ] Complete onboarding checklist 100%
- [ ] Merge 5-10 PRs (mix of fixes and small features)
- [ ] Understand entire codebase architecture
- [ ] Set up local debugging workflows
- [ ] Complete all required training

**Process Goals**:
- [ ] Participate in 2-3 sprint cycles
- [ ] Attend all team ceremonies
- [ ] Shadow experienced engineer for 1 week
- [ ] Complete peer review training
- [ ] Learn deployment and rollback procedures

**Relationship Goals**:
- [ ] 1:1 meetings with all team members
- [ ] Identify mentor relationship
- [ ] Join team social channels
- [ ] Attend 1-2 social events
- [ ] Understand team communication norms

**Success Metrics**:
- Merged PRs: 5-10
- Code reviews given: 10+
- Training completion: 100%
- Documentation contributions: 2-3
- Manager satisfaction: "Meeting expectations"

### Days 31-60: Contribution & Ownership

**Technical Goals**:
- [ ] Own 2-3 features from design to deployment
- [ ] Improve test coverage in owned areas
- [ ] Contribute to technical design docs
- [ ] Identify and fix 3-5 bugs proactively
- [ ] Optimize performance in 1-2 areas

**Process Goals**:
- [ ] Lead 1-2 feature development cycles
- [ ] Mentor newer hire (if applicable)
- [ ] Participate in architecture discussions
- [ ] Improve team documentation
- [ ] Suggest process improvements

**Relationship Goals**:
- [ ] Cross-team collaboration (work with 2-3 other teams)
- [ ] Present in team demo
- [ ] Participate in hiring (interview 1-2 candidates)
- [ ] Contribute to team culture initiatives

**Success Metrics**:
- Merged PRs: 15-20
- Features owned: 2-3
- Code quality: High (low bug rate)
- Collaboration: Active PR reviewer
- Manager satisfaction: "Exceeding expectations"

### Days 61-90: Autonomy & Impact

**Technical Goals**:
- [ ] Own complex, high-impact feature
- [ ] Propose and implement architectural improvement
- [ ] Reduce technical debt in owned area
- [ ] Contribute to system design
- [ ] Improve development velocity (tooling, automation)

**Process Goals**:
- [ ] Lead sprint planning discussion
- [ ] Create reusable component/library
- [ ] Write technical blog post or give tech talk
- [ ] Improve CI/CD pipeline
- [ ] Establish best practice documentation

**Relationship Goals**:
- [ ] Mentor 1-2 junior engineers
- [ ] Lead cross-functional project
- [ ] Contribute to team strategy
- [ ] Represent team in company meetings

**Success Metrics**:
- Merged PRs: 20-30
- Major features shipped: 1-2
- Technical leadership: Active
- Team impact: Measurable improvements
- Manager satisfaction: "Strong performer"

---

## Role-Specific Onboarding

### Frontend Engineer

**Week 1 Focus**:
- Component library and design system
- State management patterns (Redux, Context API)
- Styling conventions (CSS-in-JS, Tailwind)
- Build tools (Webpack, Vite)
- Browser DevTools proficiency

**Key Tasks**:
1. Build simple UI component (Button, Card)
2. Implement form with validation
3. Connect component to API
4. Add unit tests with Testing Library
5. Optimize bundle size

**Resources**:
- Component library documentation
- Design system Figma files
- Frontend architecture guide
- Performance optimization guide

### Backend Engineer

**Week 1 Focus**:
- API architecture and conventions
- Database schema and ORM
- Authentication and authorization
- Caching strategies
- Queue and background job systems

**Key Tasks**:
1. Add new API endpoint with tests
2. Write database migration
3. Implement caching for slow query
4. Create background job
5. Add API documentation

**Resources**:
- API documentation (OpenAPI/Swagger)
- Database schema documentation
- Security best practices guide
- Performance optimization guide

### Full-Stack Engineer

**Week 1 Focus**:
- End-to-end feature development
- API design and integration
- Database modeling
- Frontend-backend interaction patterns
- Deployment pipelines

**Key Tasks**:
1. Implement complete feature (frontend + backend)
2. Design and create API endpoint
3. Build UI that consumes API
4. Add E2E tests
5. Deploy to staging

**Resources**:
- Full-stack architecture guide
- API-first development guide
- Testing pyramid documentation
- Deployment guide

### DevOps Engineer

**Week 1 Focus**:
- Infrastructure as Code (Terraform, CloudFormation)
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Monitoring and alerting (Prometheus, Grafana)
- Container orchestration (Kubernetes)
- Security and compliance

**Key Tasks**:
1. Improve existing CI/CD pipeline
2. Add monitoring for new service
3. Automate manual deployment step
4. Create infrastructure module
5. Document runbook procedure

**Resources**:
- Infrastructure documentation
- Runbook library
- On-call guide
- Incident response procedures

---

## Onboarding Checkpoints

### Week 1 Checkpoint (Manager)

**Discussion Topics**:
- How is setup going?
- Any blockers or confusion?
- Is the pace appropriate?
- Questions about the team or product?
- Feedback on onboarding experience

**Expected Status**:
- Development environment: 100% set up
- First PRs: In review or merged
- Team connections: Met everyone
- Documentation: Core docs read
- Comfort level: Still learning but progressing

### Week 4 Checkpoint (Manager)

**Discussion Topics**:
- Confidence level with codebase
- Ready for increased autonomy?
- Areas needing more support
- Feedback on mentorship
- Career goals and development areas

**Expected Status**:
- PRs merged: 5-10
- Feature ownership: 1-2 small features
- Code reviews: Actively participating
- Team integration: Comfortable
- Productivity: 60-70% of full productivity

### Day 90 Review (Manager)

**Discussion Topics**:
- Onboarding effectiveness
- Technical skill development
- Team collaboration quality
- Areas for continued growth
- Career trajectory and goals
- Performance expectations going forward

**Expected Status**:
- PRs merged: 25-35
- Feature ownership: 3-5 significant features
- Technical impact: Measurable
- Team contribution: Active
- Productivity: 90-100% of full productivity

---

## Mentor Responsibilities

### Mentor Selection Criteria
- 2+ years with company
- Strong technical skills
- Good communicator
- Patient and supportive
- Available for regular check-ins

### Mentor Duties

**Week 1**:
- Daily check-ins (30 minutes)
- Pair programming (2-3 hours)
- Code review mentorship
- Answer questions promptly
- Provide encouragement

**Week 2-4**:
- Check-ins 3x per week
- Review all PRs before submission
- Provide feedback on code quality
- Help with debugging
- Introduce to other team members

**Month 2-3**:
- Weekly check-ins
- Review major PRs
- Career development discussions
- Technical growth guidance
- Gradual independence

**Mentor Time Commitment**:
- Week 1: 10-15 hours
- Week 2-4: 5-8 hours/week
- Month 2-3: 2-3 hours/week

---

## Remote Onboarding Best Practices

### Challenges
- Harder to ask "quick questions"
- Difficulty sensing team culture
- Technical setup issues without IT support
- Feeling of isolation
- Time zone differences

### Solutions

**Over-communicate**:
- Daily video check-ins with manager
- Dedicated Slack channel for new hire
- Regular "office hours" for questions
- Recorded video tutorials
- Written summaries of meetings

**Virtual Social Connection**:
- Virtual coffee chats (15 min, 1:1)
- Team lunch via video (order delivery)
- Virtual happy hours
- Slack channels for hobbies/interests
- Online team games or activities

**Clear Documentation**:
- Video walkthroughs of setup process
- Troubleshooting FAQ
- Contact list with time zones
- Links to all resources in one place
- Visual diagrams of architecture

**Structured Schedule**:
- Clear daily agenda for first week
- Scheduled pair programming sessions
- Regular 1:1s with multiple team members
- Explicit "focus time" blocks
- End-of-day async updates

---

## Measuring Onboarding Success

### Quantitative Metrics
- Time to first PR merged: Target <3 days
- Time to first feature shipped: Target <30 days
- PR merge rate: Target 80%+ approved first try by Day 90
- Code review participation: Target 15+ reviews by Day 90
- Training completion rate: Target 100% by Day 30

### Qualitative Metrics
- New hire confidence survey (Week 1, 4, 12)
- Manager assessment (Week 4, 12)
- Peer feedback (Day 90)
- Onboarding NPS score
- Retention at 6 months and 1 year

### Survey Questions

**Week 1**:
- How clear are expectations? (1-5)
- How supported do you feel? (1-5)
- How effective is documentation? (1-5)
- What's been most helpful?
- What's been most challenging?

**Week 4**:
- Confidence in contributing independently? (1-5)
- Quality of mentorship? (1-5)
- Understanding of codebase? (1-5)
- Pace of onboarding? (Too fast/Just right/Too slow)
- Suggestions for improvement?

**Day 90**:
- Overall onboarding satisfaction? (1-10)
- Would you recommend company to others? (NPS)
- What went well?
- What could be improved?
- Are you set up for success in your role?

---

## Continuous Improvement

### Onboarding Retrospective
- Conduct after each new hire completes 90 days
- Gather feedback from new hire, mentor, and manager
- Identify pain points and successes
- Update documentation and processes
- Track improvements over time

### Documentation Maintenance
- Quarterly review of all onboarding docs
- Update for codebase changes
- Add learnings from recent onboarding
- Remove outdated information
- Get feedback from recent hires

### Onboarding Playbook Versioning
- Version control onboarding docs
- Track changes over time
- A/B test different approaches
- Measure impact of changes
- Share learnings across teams
