# SLO Best Practices & Strategies
## Progressive SLO Implementation

Implement SLOs progressively, starting with achievable targets and gradually increasing reliability requirements.

### Progressive Implementation Strategy

```python
#!/usr/bin/env python3
"""
Progressive SLO Implementation

Gradually increase SLO targets over time as reliability improves.
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum


class ImplementationPhase(Enum):
    """Progressive implementation phases"""
    BASELINE = "baseline"           # Phase 1: 99.0%
    IMPROVEMENT = "improvement"     # Phase 2: 99.5%
    PRODUCTION_READY = "production_ready"  # Phase 3: 99.9%
    EXCELLENCE = "excellence"       # Phase 4: 99.95%


class ProgressiveSLOManager:
    """Manage progressive SLO implementation"""

    def __init__(self):
        self.phases = self._define_phases()

    def _define_phases(self) -> Dict[ImplementationPhase, Dict]:
        """Define progressive implementation phases"""
        return {
            ImplementationPhase.BASELINE: {
                'duration_days': 30,
                'availability_target': 0.990,   # 99.0%
                'latency_p95_ms': 2000,
                'latency_p99_ms': 5000,
                'description': 'Baseline establishment - measure current performance',
                'objectives': [
                    'Establish baseline metrics',
                    'Implement basic monitoring',
                    'Create initial dashboards',
                    'Set up alerting infrastructure'
                ],
                'success_criteria': [
                    'Metrics collection stable for 30 days',
                    'Basic alerts configured',
                    'Team trained on SLO concepts'
                ]
            },

            ImplementationPhase.IMPROVEMENT: {
                'duration_days': 60,
                'availability_target': 0.995,   # 99.5%
                'latency_p95_ms': 1000,
                'latency_p99_ms': 2000,
                'description': 'Initial improvement - address low-hanging fruit',
                'objectives': [
                    'Fix obvious reliability issues',
                    'Implement retry logic',
                    'Add circuit breakers',
                    'Improve error handling',
                    'Optimize slow queries'
                ],
                'success_criteria': [
                    'Meet 99.5% availability for 30 consecutive days',
                    'Error budget policy established',
                    'Regular SLO reviews scheduled'
                ]
            },

            ImplementationPhase.PRODUCTION_READY: {
                'duration_days': 90,
                'availability_target': 0.999,   # 99.9%
                'latency_p95_ms': 500,
                'latency_p99_ms': 1000,
                'description': 'Production readiness - robust reliability',
                'objectives': [
                    'Implement comprehensive monitoring',
                    'Add automated remediation',
                    'Deploy to multiple availability zones',
                    'Implement load balancing',
                    'Add auto-scaling',
                    'Create runbooks'
                ],
                'success_criteria': [
                    'Meet 99.9% availability for 60 consecutive days',
                    'Incidents resolved within SLA',
                    'Automated testing in place',
                    'Disaster recovery tested'
                ]
            },

            ImplementationPhase.EXCELLENCE: {
                'duration_days': None,  # Ongoing
                'availability_target': 0.9995,  # 99.95%
                'latency_p95_ms': 200,
                'latency_p99_ms': 500,
                'description': 'Excellence - industry-leading reliability',
                'objectives': [
                    'Implement chaos engineering',
                    'Deploy multi-region',
                    'Add advanced observability',
                    'Continuous optimization',
                    'Predictive alerting'
                ],
                'success_criteria': [
                    'Sustained 99.95% availability',
                    'Zero-downtime deployments',
                    'Proactive issue detection',
                    'Industry recognition'
                ]
            }
        }

    def implement_progressive_slos(self, service: str) -> Dict:
        """
        Generate progressive SLO implementation plan

        Args:
            service: Service name

        Returns:
            Implementation plan with phases and timeline
        """
        start_date = datetime.now()
        phases = []

        current_date = start_date
        for phase_enum in ImplementationPhase:
            phase = self.phases[phase_enum]

            if phase['duration_days']:
                end_date = current_date + timedelta(days=phase['duration_days'])
            else:
                end_date = None  # Ongoing

            phases.append({
                'phase': phase_enum.value,
                'start_date': current_date.isoformat(),
                'end_date': end_date.isoformat() if end_date else 'ongoing',
                'duration_days': phase['duration_days'],
                'slo_config': self._generate_phase_slo(service, phase_enum),
                'objectives': phase['objectives'],
                'success_criteria': phase['success_criteria'],
                'description': phase['description']
            })

            if end_date:
                current_date = end_date

        return {
            'service': service,
            'implementation_start': start_date.isoformat(),
            'phases': phases,
            'total_duration_days': sum(
                p['duration_days'] for p in self.phases.values()
                if p['duration_days']
            )
        }

    def _generate_phase_slo(self, service: str, phase: ImplementationPhase) -> Dict:
        """Generate SLO configuration for a specific phase"""
        phase_config = self.phases[phase]

        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': f"{service}-availability-{phase.value}",
                'namespace': 'production',
                'labels': {
                    'service': service,
                    'phase': phase.value,
                    'progressive': 'true'
                },
                'annotations': {
                    'description': phase_config['description']
                }
            },
            'spec': {
                'service': service,
                'description': f"{phase.value} phase SLO for {service}",
                'indicator': {
                    'type': 'ratio',
                    'ratio': {
                        'good': {
                            'metric': 'http_requests_total',
                            'filters': ['status_code !~ "5.."']
                        },
                        'total': {
                            'metric': 'http_requests_total'
                        }
                    }
                },
                'objectives': [
                    {
                        'displayName': f'{phase.value} availability target',
                        'window': '30d',
                        'target': phase_config['availability_target']
                    }
                ],
                'alerting': self._generate_phase_alerting(phase)
            }
        }

    def _generate_phase_alerting(self, phase: ImplementationPhase) -> Dict:
        """Generate alerting configuration appropriate for phase"""

        if phase == ImplementationPhase.BASELINE:
            # Minimal alerting during baseline
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'info',
                        'shortWindow': '24h',
                        'longWindow': '1h',
                        'burnRate': 5
                    }
                ]
            }

        elif phase == ImplementationPhase.IMPROVEMENT:
            # Moderate alerting
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'warning',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 8
                    }
                ]
            }

        else:
            # Full production alerting
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'critical',
                        'shortWindow': '1h',
                        'longWindow': '5m',
                        'burnRate': 14.4
                    },
                    {
                        'severity': 'warning',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 3
                    }
                ]
            }

    def check_phase_readiness(
        self,
        current_phase: ImplementationPhase,
        metrics: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Check if service is ready to advance to next phase

        Args:
            current_phase: Current implementation phase
            metrics: Service metrics

        Returns:
            (ready, issues) tuple
        """
        phase_config = self.phases[current_phase]
        issues = []

        # Check availability target
        if metrics['availability'] < phase_config['availability_target']:
            issues.append(
                f"Availability {metrics['availability']*100:.2f}% "
                f"below target {phase_config['availability_target']*100:.2f}%"
            )

        # Check latency targets
        if metrics['latency_p95'] > phase_config['latency_p95_ms']:
            issues.append(
                f"P95 latency {metrics['latency_p95']}ms "
                f"exceeds target {phase_config['latency_p95_ms']}ms"
            )

        # Check stability (coefficient of variation)
        if metrics.get('availability_stddev', 0) > 0.01:  # 1% variation
            issues.append("Availability too variable - not stable")

        # Check incident rate
        if metrics.get('incidents_last_30d', 0) > 5:
            issues.append("Too many incidents - improve stability first")

        ready = len(issues) == 0
        return ready, issues

    def generate_implementation_report(self, plan: Dict) -> str:
        """Generate human-readable implementation report"""

        report = f"""
# Progressive SLO Implementation Plan

**Service**: {plan['service']}
**Start Date**: {plan['implementation_start']}
**Total Duration**: {plan['total_duration_days']} days


## Implementation Phases

"""

        for i, phase in enumerate(plan['phases'], 1):
            report += f"""
### Phase {i}: {phase['phase'].title()}

**Timeline**: {phase['start_date']} → {phase['end_date']}
**Duration**: {phase['duration_days']} days
**Description**: {phase['description']}

**SLO Target**: {phase['slo_config']['spec']['objectives'][0]['target']*100:.2f}%

**Objectives**:
{self._format_list(phase['objectives'])}

**Success Criteria**:
{self._format_list(phase['success_criteria'])}

---
"""

        return report

    def _format_list(self, items: List[str]) -> str:
        """Format list items as markdown"""
        return '\n'.join(f"- {item}" for item in items)


# Example usage
if __name__ == "__main__":
    manager = ProgressiveSLOManager()

    # Generate implementation plan
    plan = manager.implement_progressive_slos('api-service')

    # Generate report
    report = manager.generate_implementation_report(plan)
    print(report)

    # Check readiness to advance
    current_metrics = {
        'availability': 0.9965,
        'latency_p95': 450,
        'availability_stddev': 0.005,
        'incidents_last_30d': 2
    }

    ready, issues = manager.check_phase_readiness(
        ImplementationPhase.IMPROVEMENT,
        current_metrics
    )

    if ready:
        print("\nReady to advance to next phase!")
    else:
        print("\nNot ready to advance. Issues:")
        for issue in issues:
            print(f"  - {issue}")
```

---


## Migration Strategies

Strategies for migrating from manual to automated SLO management.

### Migration Guide

```markdown
# SLO Automation Migration Guide

## Overview

This guide provides a phased approach to migrating from manual SLO management to fully automated SLO-as-code.

## Migration Phases

### Phase 1: Assessment (Week 1-2)

**Objectives:**
- Inventory existing SLOs and monitoring
- Identify gaps in current implementation
- Define target state

**Activities:**
1. Document existing SLOs
   - What services have SLOs?
   - How are they measured?
   - Where are they defined?

2. Assess current tooling
   - Prometheus/monitoring setup
   - Dashboard tools
   - Alerting infrastructure

3. Identify stakeholders
   - Service owners
   - SRE team
   - Product management

**Deliverables:**
- Current state documentation
- Gap analysis
- Migration roadmap

### Phase 2: Pilot (Week 3-6)

**Objectives:**
- Implement SLO-as-code for 2-3 pilot services
- Validate approach
- Refine templates and tooling

**Activities:**
1. Select pilot services
   - Choose services with different characteristics
   - Ensure team buy-in

2. Create SLO definitions
   - Use templates from library
   - Customize for each service

3. Deploy automation
   - Set up GitOps pipeline
   - Configure validation
   - Deploy monitoring

4. Run in parallel
   - Keep existing SLOs
   - Compare automated vs manual

**Success Criteria:**
- Automated SLOs match manual SLOs
- Deployments succeed
- Alerts fire correctly

### Phase 3: Expansion (Week 7-12)

**Objectives:**
- Migrate all production services
- Establish processes and training
- Build confidence in automation

**Activities:**
1. Migrate services by tier
   - Start with best-effort
   - Move to critical last

2. Train teams
   - SLO-as-code concepts
   - GitOps workflow
   - Troubleshooting

3. Establish review process
   - SLO change reviews
   - Regular SLO meetings

**Success Criteria:**
- 80%+ services migrated
- Teams comfortable with process
- Error budget policies in use

### Phase 4: Optimization (Week 13+)

**Objectives:**
- Refine SLO targets
- Improve automation
- Drive continuous improvement

**Activities:**
1. Analyze SLO performance
   - Review adherence
   - Adjust targets
   - Optimize alerts

2. Enhance automation
   - Add more templates
   - Improve validation
   - Automate remediation

3. Expand scope
   - Add new SLI types
   - Multi-region SLOs
   - Composite SLOs

**Success Criteria:**
- All services on automated SLOs
- Regular refinement process
- SLOs driving decisions

## Migration Checklist

### Prerequisites
- [ ] Prometheus deployed and stable
- [ ] Grafana for dashboards
- [ ] Git repository for SLO definitions
- [ ] CI/CD pipeline available
- [ ] Kubernetes cluster (if using CRDs)

### Setup
- [ ] Install SLO automation tools
- [ ] Configure Prometheus integration
- [ ] Set up GitOps workflow
- [ ] Deploy Kubernetes CRDs (optional)
- [ ] Create initial templates

### Pilot Services
- [ ] Select 2-3 pilot services
- [ ] Define SLOs in YAML
- [ ] Validate definitions
- [ ] Deploy to monitoring
- [ ] Verify metrics and alerts
- [ ] Run parallel for 2 weeks
- [ ] Get team feedback

### Production Rollout
- [ ] Create SLO definitions for all services
- [ ] Validate all definitions
- [ ] Deploy in phases (by tier)
- [ ] Train service owners
- [ ] Document processes
- [ ] Establish review cadence
- [ ] Decommission manual SLOs

### Continuous Improvement
- [ ] Monthly SLO reviews
- [ ] Quarterly target adjustments
- [ ] Regular template updates
- [ ] Automation enhancements
- [ ] Team training refreshers

## Common Challenges

### Challenge: Resistance to Change
**Solution:**
- Start with volunteers
- Show value quickly
- Make it easy to adopt
- Provide good documentation

### Challenge: Incomplete Metrics
**Solution:**
- Add instrumentation first
- Use progressive implementation
- Start with basic SLOs
- Improve over time

### Challenge: Alert Fatigue
**Solution:**
- Start with generous thresholds
- Use multi-window burn rates
- Adjust based on feedback
- Focus on actionable alerts

### Challenge: Complex Services
**Solution:**
- Break down into components
- Use multiple SLOs
- Start simple, add complexity
- Get architecture input

## Support Resources

- **Documentation**: /docs/slo-automation
- **Templates**: /slo-definitions/templates
- **Examples**: /slo-definitions/examples
- **Slack**: #slo-automation
- **Office Hours**: Tuesdays 2-3pm
```

---

