# Error Budget Concepts

## Error Budget Fundamentals

### What is an Error Budget?

An error budget is the **maximum amount of unreliability** that a service can tolerate while still meeting its SLO. It represents the difference between 100% reliability and your SLO target.

**Key Principles:**
- Error budgets balance reliability with feature velocity
- They provide a shared incentive between product and engineering teams
- They enable data-driven decisions about risk and reliability
- They make reliability a feature, not an afterthought

### Error Budget Philosophy

```
SLO = 99.9% availability
Error Budget = 100% - 99.9% = 0.1% allowed unreliability

In a 30-day month:
Total time = 30 days × 24 hours × 60 minutes = 43,200 minutes
Error budget = 43,200 × 0.001 = 43.2 minutes of downtime allowed
```

**Strategic Uses:**
1. **Feature Velocity**: When budget is healthy, ship faster
2. **Risk Taking**: Budget enables innovation and experimentation
3. **Maintenance Windows**: Planned downtime consumes error budget
4. **Incident Response**: Prioritize based on budget consumption
5. **Release Decisions**: Gate risky releases when budget is low

## Burn Rate Concepts

### What is Burn Rate?

**Burn rate** is the ratio of the rate at which you're consuming your error budget compared to the expected consumption rate for meeting your SLO exactly.

```
Burn Rate = (Actual Error Rate) / (SLO Error Rate)

Where:
- Burn rate = 1: Consuming budget at expected rate (on track)
- Burn rate > 1: Consuming budget faster than expected (problem)
- Burn rate < 1: Consuming budget slower than expected (healthy)
```
