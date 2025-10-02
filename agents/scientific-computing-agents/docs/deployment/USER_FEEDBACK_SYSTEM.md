# User Feedback Collection System

**Scientific Computing Agents**
**Version**: 1.0
**Date**: 2025-10-01

---

## Overview

This document describes the user feedback collection system for gathering insights during Phase 5A Weeks 3-4 (User Validation) and ongoing operation.

---

## Feedback Channels

### 1. Automated Usage Analytics

**Implementation**: Lightweight event tracking

```python
# agents/core/analytics.py
import json
import os
from datetime import datetime
from typing import Dict, Any

class UsageAnalytics:
    """Lightweight usage analytics (opt-in)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled and os.getenv('SCI_AGENTS_ANALYTICS', 'false').lower() == 'true'
        self.log_file = os.getenv('SCI_AGENTS_ANALYTICS_FILE', 'usage_analytics.jsonl')

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a usage event."""
        if not self.enabled:
            return

        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception:
            pass  # Silent failure - don't break user code

# Usage in agents
analytics = UsageAnalytics()

class SomeAgent:
    def process(self, data):
        analytics.log_event('agent_execution', {
            'agent': self.__class__.__name__,
            'task': data.get('task'),
            'success': True
        })
```

**Events to Track**:
- Agent invocations (type, task, duration)
- Workflow executions (steps, parallel/sequential, duration)
- Errors (type, agent, context)
- Performance metrics (execution time, memory usage)

**Privacy**:
- **Opt-in only**: Disabled by default
- **No sensitive data**: Only operation types and performance
- **Local storage**: Data stays on user's machine
- **User control**: Can delete/disable anytime

### 2. Survey System

**Implementation**: Simple web form or Google Forms

**Initial Survey** (Week 3, Day 1):
```markdown
# Welcome Survey

Thank you for joining the beta program!

## Background
1. What is your field? (Physics/Biology/Engineering/CS/Other)
2. What problems do you solve? (Open text)
3. What tools do you currently use? (Open text)

## Expectations
4. What do you hope to achieve with this system? (Open text)
5. What features are most important to you? (Ranking)
   - [ ] ODE/PDE solving
   - [ ] Optimization
   - [ ] Linear algebra
   - [ ] Machine learning integration
   - [ ] Workflow orchestration
   - [ ] Performance profiling

## Technical Setup
6. Installation method? (PyPI/Source/Docker)
7. Any installation issues? (Yes/No + details)
8. Development environment? (Jupyter/IDE/Terminal)

Estimated time: 5 minutes
```

**Mid-Point Survey** (Week 3, Day 5):
```markdown
# Week 1 Check-in

## Usage
1. How many times have you used the system? (0/1-5/6-10/10+)
2. Which agents have you used? (Checkboxes)
3. Have you created workflows? (Yes/No)

## Experience
4. What worked well? (Open text)
5. What was confusing? (Open text)
6. What's missing? (Open text)

## Performance
7. Is performance acceptable? (Yes/No + details)
8. Any errors encountered? (Yes/No + details)

## Next Steps
9. Will you continue using the system? (Yes/Maybe/No)
10. What would make it more useful? (Open text)

Estimated time: 3 minutes
```

**Final Survey** (Week 4, Day 5):
```markdown
# Final Feedback

## Overall Experience
1. Overall satisfaction: ⭐⭐⭐⭐⭐ (1-5 stars)
2. Likelihood to recommend: 0-10 (NPS score)
3. Most valuable feature? (Open text)
4. Biggest pain point? (Open text)

## Specific Areas

### Documentation
5. Documentation quality: (Excellent/Good/Fair/Poor)
6. Examples helpfulness: (Excellent/Good/Fair/Poor)
7. What documentation is missing? (Open text)

### Performance
8. Speed: (Excellent/Good/Fair/Poor)
9. Reliability: (Excellent/Good/Fair/Poor)
10. Resource usage: (Excellent/Good/Fair/Poor)

### Features
11. Feature completeness: (Excellent/Good/Fair/Poor)
12. Most used feature: (Open text)
13. Missing features: (Open text)

## Future
14. Would you use in production? (Yes/Maybe/No + why)
15. What improvements would make this production-ready? (Open text)
16. Interest in contributing? (Yes/Maybe/No)

## Case Study
17. May we feature your use case? (Yes/No)
18. Brief description of your use: (Open text, optional)

Estimated time: 5 minutes
```

### 3. Direct Communication

**Channels**:
- **Email**: feedback@scientific-agents.example.com
- **Slack**: #sci-agents-feedback
- **GitHub Issues**: Bug reports and feature requests
- **1-on-1 Calls**: Schedule with power users (optional)

**Response SLA**:
- **Email**: 24 hours
- **Slack**: 4 hours during business hours
- **GitHub**: 48 hours

### 4. Error Reporting

**Automated Error Collection** (opt-in):

```python
# agents/core/error_reporting.py
import sys
import traceback
from typing import Optional

class ErrorReporter:
    """Collect error reports for debugging."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def report_error(self, error: Exception, context: dict):
        """Report an error with context."""
        if not self.enabled:
            return

        error_data = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }

        # Log to file (user's machine)
        self._log_error(error_data)

    def _log_error(self, error_data):
        """Log error to local file."""
        import json
        from datetime import datetime

        error_data['timestamp'] = datetime.utcnow().isoformat()

        try:
            with open('error_log.jsonl', 'a') as f:
                f.write(json.dumps(error_data) + '\n')
        except Exception:
            pass
```

### 5. Performance Metrics

**Automated Collection**:
```python
# Collect performance metrics during normal operation
metrics = {
    'agent_execution_times': [],
    'workflow_durations': [],
    'error_rates': {},
    'resource_usage': []
}
```

**Analysis Script**:
```bash
# scripts/analyze_feedback.py
python scripts/analyze_feedback.py --analytics-file usage_analytics.jsonl
```

---

## Feedback Analysis Process

### Weekly Analysis (During Weeks 3-4)

**Every Monday**:
1. Collect all feedback from previous week
2. Categorize by type (bug/feature/performance/documentation)
3. Prioritize by frequency and severity
4. Update tracking spreadsheet

**Template**:
```markdown
# Week 3 Feedback Summary

## Metrics
- Active users: X
- Survey responses: Y
- Issues reported: Z
- Feature requests: W

## Top Issues
1. [Issue description] - Reported by X users
2. [Issue description] - Reported by Y users
3. [Issue description] - Reported by Z users

## Top Feature Requests
1. [Feature] - Requested by X users
2. [Feature] - Requested by Y users

## Positive Feedback
- [Quote from user]
- [Quote from user]

## Action Items
- [ ] Fix critical bug in [agent]
- [ ] Improve documentation for [topic]
- [ ] Consider feature: [description]
```

### End-of-Phase Analysis (Week 4, Days 6-7)

**Comprehensive Report**:

```markdown
# Phase 5A User Validation Report

## Executive Summary
- Total users: X
- Survey completion rate: Y%
- NPS Score: Z
- Overall satisfaction: W/5 stars

## Usage Patterns
- Most used agents: [list]
- Most used features: [list]
- Common workflows: [descriptions]

## Pain Points
1. [Pain point] - X% of users
2. [Pain point] - Y% of users
3. [Pain point] - Z% of users

## Feature Requests
1. [Feature] - Priority: High/Medium/Low
2. [Feature] - Priority: High/Medium/Low

## Success Stories
- User A: [achievement]
- User B: [achievement]

## Recommendations for Phase 5B
1. [Recommendation based on feedback]
2. [Recommendation based on feedback]
3. [Recommendation based on feedback]
```

---

## Data Storage and Privacy

### Local Storage

All feedback data stored locally on respective systems:
- **Usage analytics**: `usage_analytics.jsonl`
- **Error logs**: `error_log.jsonl`
- **Survey responses**: Google Forms / local CSV

### Privacy Policy

**User Data**:
- ✅ Stored locally by default
- ✅ No automatic cloud upload
- ✅ User controls all data
- ✅ Can opt-out anytime

**Shared Data** (opt-in):
- User explicitly shares via survey
- Anonymized before aggregation
- No personally identifiable information
- Used only for product improvement

### GDPR Compliance

- **Right to access**: Users can view all collected data
- **Right to deletion**: Simple delete commands
- **Right to opt-out**: Analytics disabled by default
- **Transparency**: Clear documentation of what's collected

---

## Feedback Collection Tools

### Survey Platform

**Option 1: Google Forms** (Recommended for MVP)
- Free
- Easy to set up
- Good analytics
- Export to CSV

**Option 2: TypeForm**
- Better UX
- More customization
- Free tier available

**Option 3: Custom Form**
- Full control
- Requires hosting
- More development effort

### Analytics Dashboard

**Simple Python Dashboard**:
```python
# scripts/feedback_dashboard.py
import pandas as pd
import json

def load_analytics(file_path='usage_analytics.jsonl'):
    """Load and analyze usage data."""
    events = []
    with open(file_path) as f:
        for line in f:
            events.append(json.loads(line))

    df = pd.DataFrame(events)

    # Summary statistics
    print("=" * 60)
    print("Usage Analytics Summary")
    print("=" * 60)
    print(f"\nTotal events: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")  # if tracked
    print(f"\nEvent types:")
    print(df['event_type'].value_counts())

    # Agent usage
    if 'data' in df.columns:
        agent_events = df[df['event_type'] == 'agent_execution']
        print(f"\nMost used agents:")
        # Extract agent names from data field
        # ...

    # Error analysis
    error_events = df[df['event_type'] == 'error']
    if len(error_events) > 0:
        print(f"\nErrors: {len(error_events)}")
        # Categorize errors
        # ...

if __name__ == "__main__":
    load_analytics()
```

---

## User Communication Templates

### Welcome Email

```
Subject: Welcome to Scientific Computing Agents Beta!

Hi [Name],

Thank you for joining the beta program! We're excited to have you.

Quick Start:
1. Installation: pip install scientific-computing-agents
2. First tutorial: examples/tutorial_01_quick_start.py
3. Documentation: docs/USER_ONBOARDING.md

Help Resources:
- Slack: #sci-agents-help
- Email: support@example.com
- Office hours: Thursdays 2-4 PM UTC

We'd love your feedback! Please complete this short survey:
[Survey Link]

Happy computing!

The Sci-Agents Team
```

### Weekly Check-in

```
Subject: Week 1 Check-in - How's it going?

Hi [Name],

You've been using Scientific Computing Agents for a week now. We'd love to hear how it's going!

Quick check-in survey (3 minutes):
[Survey Link]

This week's tips:
- Try the workflow orchestration for multi-step analyses
- Use the profiler to optimize performance
- Join office hours if you have questions

Questions? Reply to this email or ping us on Slack.

Thanks for your feedback!

The Sci-Agents Team
```

### Thank You

```
Subject: Thank you for your feedback!

Hi [Name],

Thank you for completing the feedback survey. Your insights are invaluable!

Based on your feedback:
- [Specific action we're taking]
- [Feature you requested - timeline]

Keep the feedback coming! We read every response.

As a token of appreciation, you're eligible for:
- Early access to new features
- Co-authorship on future publications
- Credit in release notes

Thanks for being part of the community!

The Sci-Agents Team
```

---

## Success Metrics

### Quantitative Metrics

- **User Engagement**
  - Target: 10+ active users in Weeks 3-4
  - Definition: Used system at least 3 times

- **Survey Response Rate**
  - Target: >60%
  - Initial survey: >80%

- **Satisfaction**
  - Target NPS: >40 (Excellent: >50)
  - Target satisfaction: >3.5/5 stars

- **Retention**
  - Target: >70% plan to continue using

### Qualitative Metrics

- **Use Case Diversity**: 3+ different application domains
- **Success Stories**: 3+ documented use cases
- **Community Engagement**: Active discussions in Slack
- **Feature Clarity**: <20% confusion about features

---

## Timeline

### Week 3
- **Day 1**: Send welcome email + initial survey
- **Day 3**: First check-in with users
- **Day 5**: Mid-point survey
- **Day 7**: Analyze Week 1 feedback

### Week 4
- **Day 1**: Share Week 1 improvements
- **Day 3**: Second check-in
- **Day 5**: Final survey
- **Day 7**: Comprehensive analysis + Phase 5B planning

---

**Document Version**: 1.0
**Owner**: Product Team
**Next Review**: After Phase 5A completion
