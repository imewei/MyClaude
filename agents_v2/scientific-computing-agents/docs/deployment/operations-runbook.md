# Operations Runbook

**Scientific Computing Agents System**
**Version**: 1.0
**Last Updated**: 2025-10-01

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Quick Reference](#quick-reference)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring and Alerts](#monitoring-and-alerts)
5. [Incident Response](#incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Rollback Procedures](#rollback-procedures)
9. [Emergency Contacts](#emergency-contacts)

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────┐
│           Load Balancer (if applicable)          │
└──────────────────┬──────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐    ┌───▼────┐    ┌───▼────┐
│ API-1  │    │ API-2  │    │ API-3  │
└───┬────┘    └───┬────┘    └───┬────┘
    │             │              │
    └─────────────┼──────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───▼────────┐         ┌───────▼─────┐
│   Redis    │         │  Postgres   │
│   Cache    │         │  Database   │
└────────────┘         └─────────────┘
```

### Key Components

- **API Servers**: Scientific computing agent system instances
- **Redis**: Caching layer (optional)
- **PostgreSQL**: State persistence (optional)
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards

---

## Quick Reference

### Critical Commands

```bash
# Health check
python scripts/health_check.py

# Run benchmarks
python scripts/benchmark.py

# Security audit
python scripts/security_audit.py

# View logs
docker-compose logs -f sci-agents

# Restart service
docker-compose restart sci-agents

# View metrics
curl http://localhost:9090/metrics
```

### Key Files

- **Configuration**: `config/agents.yaml`
- **Environment**: `.env`
- **Logs**: `/var/log/sci-agents.log`
- **Docker Compose**: `docker-compose.yml`

### Service Ports

- **API**: 8000 (if exposed)
- **Jupyter**: 8888 (dev only)
- **Prometheus**: 9090
- **Grafana**: 3000
- **Redis**: 6379
- **PostgreSQL**: 5432

---

## Deployment Procedures

### Standard Deployment

#### 1. Pre-Deployment Checks

```bash
# Run tests
pytest tests/ -v --tb=short

# Run security audit
python scripts/security_audit.py

# Run benchmarks (optional)
python scripts/benchmark.py
```

#### 2. Staging Deployment

```bash
# Pull latest code
git pull origin main

# Build Docker images
docker-compose build sci-agents

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run health check
python scripts/health_check.py

# Monitor logs
docker-compose logs -f sci-agents
```

#### 3. Production Deployment

```bash
# Tag release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Build production images
docker-compose build sci-agents

# Stop old containers (zero-downtime with load balancer)
docker-compose stop sci-agents

# Start new containers
docker-compose up -d sci-agents

# Health check
python scripts/health_check.py

# Monitor for 10 minutes
watch -n 10 python scripts/health_check.py
```

### Blue-Green Deployment

```bash
# Start green environment
docker-compose -f docker-compose.green.yml up -d

# Health check green
curl http://green-host:8000/health

# Switch load balancer to green
# (implementation depends on your load balancer)

# Monitor for issues
watch -n 30 python scripts/health_check.py

# If successful, stop blue
docker-compose -f docker-compose.blue.yml down

# If issues, rollback (see Rollback Procedures)
```

### Canary Deployment

```bash
# Deploy to 10% of instances
docker-compose scale sci-agents=1

# Monitor metrics for 1 hour
# Check error rates, latency, success rates

# If metrics good, scale to 50%
docker-compose scale sci-agents=5

# Monitor for 30 minutes

# If metrics good, scale to 100%
docker-compose scale sci-agents=10
```

---

## Monitoring and Alerts

### Health Checks

#### Manual Health Check

```bash
python scripts/health_check.py
```

**Expected Output**:
```
Status: HEALTHY ✓
Total checks: 5
Passed: 5
Failed: 0
```

#### Automated Health Checks

**Cron Job** (every 5 minutes):
```bash
*/5 * * * * /path/to/health_check.py >> /var/log/health.log 2>&1
```

### Key Metrics to Monitor

#### System Metrics

- **CPU Usage**: <80% normal, >80% warning, >95% critical
- **Memory Usage**: <85% normal, >85% warning, >95% critical
- **Disk Usage**: <80% normal, >80% warning, >90% critical

#### Application Metrics

- **Test Pass Rate**: >97% normal, 95-97% warning, <95% critical
- **Workflow Success Rate**: >98% normal, 95-98% warning, <95% critical
- **Average Execution Time**: <60s normal, 60-300s warning, >300s critical
- **Error Rate**: <1% normal, 1-5% warning, >5% critical

### Grafana Dashboards

Access at `http://localhost:3000` (default credentials: admin/admin)

**Key Dashboards**:
1. **System Overview**: CPU, memory, disk, network
2. **Application Performance**: Execution times, success rates
3. **Error Tracking**: Error rates by type and agent
4. **Resource Utilization**: Agent-specific resource usage

### Alert Channels

- **Email**: ops-team@example.com
- **Slack**: #sci-agents-alerts
- **PagerDuty**: Service key in secrets

---

## Incident Response

### Severity Levels

#### P0 - Critical
- **Definition**: Complete system outage, data loss
- **Response Time**: Immediate (24/7)
- **Examples**: All instances down, database corruption

#### P1 - High
- **Definition**: Major functionality broken, performance degraded >50%
- **Response Time**: <15 minutes
- **Examples**: One agent completely failing, 50% instances down

#### P2 - Medium
- **Definition**: Minor functionality impaired, performance degraded <50%
- **Response Time**: <1 hour
- **Examples**: High error rate in one agent, flaky tests

#### P3 - Low
- **Definition**: Minimal impact, cosmetic issues
- **Response Time**: <1 day
- **Examples**: Minor logging issues, documentation errors

### Incident Response Workflow

#### 1. Detection

**Automated Alerts**:
- Prometheus alerts to Slack/Email
- Health check failures

**Manual Detection**:
- User reports
- Monitoring dashboard review

#### 2. Triage

```bash
# Quick assessment
python scripts/health_check.py

# Check logs
docker-compose logs --tail=100 sci-agents

# Check resource usage
docker stats

# Check recent deployments
git log --oneline -5
```

#### 3. Communication

```
Subject: [P1] Scientific Computing Agents - High Error Rate

Status: INVESTIGATING
Started: 2025-10-01 14:30 UTC
Affected: Production environment
Impact: 15% increase in error rate

Actions Taken:
- Reviewed logs: [finding]
- Checked resource usage: [finding]
- Reviewing recent changes

Next Steps:
- [action 1]
- [action 2]

Updates: Every 15 minutes or when status changes
```

#### 4. Resolution

**Common Resolutions**:
- Restart service: `docker-compose restart sci-agents`
- Rollback: See Rollback Procedures
- Scale up: `docker-compose scale sci-agents=10`
- Clear cache: `docker-compose restart redis`

#### 5. Post-Mortem

**Template**: `docs/templates/post_mortem.md`

**Required Sections**:
- Timeline
- Root cause
- Impact assessment
- Resolution steps
- Prevention measures
- Action items

---

## Maintenance Procedures

### Routine Maintenance

#### Daily

```bash
# Check health
python scripts/health_check.py

# Review error logs
grep ERROR /var/log/sci-agents.log | tail -50

# Check disk space
df -h
```

#### Weekly

```bash
# Security audit
python scripts/security_audit.py

# Run benchmarks
python scripts/benchmark.py

# Review metrics
# (Check Grafana dashboards)

# Update dependencies (dev environment)
pip install --upgrade -r requirements.txt
```

#### Monthly

```bash
# Full system audit
python scripts/security_audit.py > audit_$(date +%Y%m).txt

# Performance review
python scripts/benchmark.py

# Log rotation
logrotate /etc/logrotate.d/sci-agents

# Backup review
# (Verify backups are working)
```

### Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update in development first
git checkout -b update-dependencies
pip install --upgrade -r requirements.txt

# Run tests
pytest tests/ -v

# If tests pass, create PR
git add requirements.txt
git commit -m "Update dependencies"
git push origin update-dependencies

# After review, deploy to staging, then production
```

### Database Maintenance

#### PostgreSQL (if used)

```bash
# Vacuum database
docker-compose exec postgres psql -U agent_user -d sci_agents -c "VACUUM ANALYZE;"

# Backup database
docker-compose exec postgres pg_dump -U agent_user sci_agents > backup_$(date +%Y%m%d).sql

# Check database size
docker-compose exec postgres psql -U agent_user -d sci_agents -c "SELECT pg_size_pretty(pg_database_size('sci_agents'));"
```

#### Redis (if used)

```bash
# Check memory usage
docker-compose exec redis redis-cli info memory

# Save snapshot
docker-compose exec redis redis-cli BGSAVE

# Clear cache (if needed)
docker-compose exec redis redis-cli FLUSHALL
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: High Error Rate

**Symptoms**: Error rate >5%

**Diagnosis**:
```bash
# Check logs
docker-compose logs --tail=1000 sci-agents | grep ERROR

# Check resource usage
docker stats sci-agents

# Run health check
python scripts/health_check.py
```

**Resolution**:
1. Identify error pattern in logs
2. If resource issue, scale up: `docker-compose scale sci-agents=5`
3. If code issue, rollback: See Rollback Procedures
4. If transient, monitor for 10 minutes

#### Issue: Slow Performance

**Symptoms**: Average execution time >300s

**Diagnosis**:
```bash
# Run benchmarks
python scripts/benchmark.py

# Check profiler
# (Review profiler outputs)

# Check resource usage
docker stats
```

**Resolution**:
1. Identify bottleneck from benchmarks
2. Scale horizontally: `docker-compose scale sci-agents=10`
3. Optimize configuration: Adjust `max_workers`
4. Consider caching: Enable Redis if not already

#### Issue: Container Won't Start

**Symptoms**: Container exits immediately

**Diagnosis**:
```bash
# Check logs
docker-compose logs sci-agents

# Check configuration
docker-compose config

# Check resource availability
docker system df
```

**Resolution**:
1. Fix configuration errors
2. Clear old containers: `docker-compose down -v`
3. Rebuild: `docker-compose build --no-cache sci-agents`
4. Restart: `docker-compose up -d sci-agents`

#### Issue: Tests Failing

**Symptoms**: CI/CD tests failing

**Diagnosis**:
```bash
# Run tests locally
pytest tests/ -v --tb=short

# Check for flaky tests
pytest tests/ --count=3

# Check recent changes
git log --oneline -10
```

**Resolution**:
1. Identify failing test
2. Run test in isolation: `pytest tests/test_specific.py -v`
3. If flaky, mark as such: `@pytest.mark.flaky`
4. If regression, fix or revert changes

---

## Rollback Procedures

### Quick Rollback (Docker)

```bash
# Identify last working version
git tag -l | tail -5

# Pull previous version
git checkout v0.0.9

# Rebuild and deploy
docker-compose build sci-agents
docker-compose up -d sci-agents

# Health check
python scripts/health_check.py

# Monitor
watch -n 30 python scripts/health_check.py
```

### Database Rollback (if needed)

```bash
# Stop application
docker-compose stop sci-agents

# Restore database backup
docker-compose exec postgres psql -U agent_user -d sci_agents < backup_YYYYMMDD.sql

# Start application
docker-compose up -d sci-agents

# Verify
python scripts/health_check.py
```

### Configuration Rollback

```bash
# Restore from git
git checkout HEAD~1 -- config/agents.yaml

# Restart services
docker-compose restart sci-agents

# Verify
python scripts/health_check.py
```

---

## Emergency Contacts

### On-Call Rotation

- **Primary**: [Name] - [Phone] - [Email]
- **Secondary**: [Name] - [Phone] - [Email]
- **Manager**: [Name] - [Phone] - [Email]

### Escalation Path

1. **P0/P1**: Contact primary on-call immediately
2. **No response in 15 min**: Contact secondary
3. **No response in 30 min**: Contact manager
4. **Critical data loss**: Contact CTO immediately

### External Contacts

- **Cloud Provider Support**: [Support number]
- **Database Support**: [Support number]
- **Security Team**: security@example.com

---

## Appendix

### Useful Commands

```bash
# View all containers
docker ps -a

# View resource usage
docker stats

# Exec into container
docker-compose exec sci-agents bash

# View network
docker network ls

# Clean up
docker system prune -a
```

### Configuration Reference

See `docs/DEPLOYMENT.md` for detailed configuration options.

### Change Log

- **2025-10-01**: Initial runbook creation
- **v1.0**: First production release

---

**Document Version**: 1.0
**Maintained By**: DevOps Team
**Review Frequency**: Monthly
