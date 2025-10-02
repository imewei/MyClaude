# Production Deployment Checklist

**Scientific Computing Agents System**
**Version**: 1.0
**Date**: 2025-10-01

---

## Pre-Deployment Checklist

### Code Quality ✅

- [ ] **All tests passing**
  ```bash
  pytest tests/ -v
  # Expected: >97% pass rate
  ```

- [ ] **Code coverage acceptable**
  ```bash
  pytest tests/ --cov=agents --cov=core
  # Expected: >75% coverage
  ```

- [ ] **Linting clean**
  ```bash
  flake8 agents/ core/
  black --check agents/ core/
  isort --check agents/ core/
  ```

- [ ] **Type checking passes**
  ```bash
  mypy agents/ core/ --ignore-missing-imports
  ```

- [ ] **Security audit passed**
  ```bash
  python scripts/security_audit.py
  # Expected: 0 critical issues
  ```

### Documentation ✅

- [ ] **README.md up to date**
  - Version number current
  - Installation instructions tested
  - Examples work

- [ ] **CHANGELOG.md updated**
  - New features documented
  - Bug fixes listed
  - Breaking changes highlighted

- [ ] **API documentation complete**
  - All public methods documented
  - Examples provided
  - Parameters described

- [ ] **Deployment guide reviewed**
  - Commands tested
  - Screenshots current
  - Links working

### Infrastructure ✅

- [ ] **CI/CD pipeline operational**
  ```bash
  # GitHub Actions should show all green
  # Check: .github/workflows/ci.yml
  ```

- [ ] **Docker images built and tested**
  ```bash
  docker-compose build
  docker-compose up -d
  docker-compose exec sci-agents python scripts/health_check.py
  ```

- [ ] **Monitoring configured**
  - Prometheus targets defined
  - Alert rules tested
  - Grafana dashboards created

- [ ] **Backup strategy defined**
  - Database backup automated
  - Configuration backed up
  - Rollback procedure tested

### Configuration ✅

- [ ] **Environment variables set**
  ```bash
  # Production .env file
  SCI_AGENTS_ENV=production
  SCI_AGENTS_LOG_LEVEL=WARNING
  SCI_AGENTS_MAX_WORKERS=16
  SCI_AGENTS_ENABLE_PROFILING=false
  ```

- [ ] **Secrets secured**
  - API keys in secrets manager
  - Database credentials encrypted
  - SSL certificates obtained

- [ ] **Resource limits configured**
  - CPU limits set
  - Memory limits set
  - Disk quotas defined

---

## Deployment Day Checklist

### T-4 Hours: Final Preparation

- [ ] **Announce deployment window**
  ```
  Subject: Scheduled Deployment - [Date] [Time]

  Deployment window: [Start] - [End]
  Expected downtime: [Duration]
  Impact: [Description]

  Contact: [On-call engineer]
  ```

- [ ] **Create deployment branch**
  ```bash
  git checkout main
  git pull origin main
  git tag -a v0.1.0 -m "Production release v0.1.0"
  git push origin v0.1.0
  ```

- [ ] **Run final tests**
  ```bash
  pytest tests/ -v --tb=short
  python scripts/health_check.py
  python scripts/benchmark.py
  ```

- [ ] **Backup current state**
  ```bash
  # Backup database
  pg_dump production_db > backup_$(date +%Y%m%d_%H%M%S).sql

  # Backup configuration
  tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
  ```

### T-2 Hours: Staging Validation

- [ ] **Deploy to staging**
  ```bash
  # Use staging environment
  export SCI_AGENTS_ENV=staging

  # Deploy
  docker-compose -f docker-compose.staging.yml up -d

  # Wait for startup
  sleep 30

  # Health check
  python scripts/health_check.py
  ```

- [ ] **Run smoke tests on staging**
  ```bash
  pytest tests/ -m "not slow" -v
  ```

- [ ] **Verify monitoring on staging**
  - Check Prometheus targets
  - Verify metrics flowing
  - Test alert rules

- [ ] **Performance test on staging**
  ```bash
  python scripts/benchmark.py
  # Compare with baseline
  ```

### T-0: Production Deployment

#### Step 1: Pre-Deployment Verification (T+0min)

- [ ] **Verify current production status**
  ```bash
  curl https://api.production.example.com/health
  python scripts/health_check.py --env=production
  ```

- [ ] **Record current metrics**
  ```bash
  # Capture baseline
  curl http://prometheus:9090/api/v1/query?query=up > baseline_metrics.json
  ```

- [ ] **Notify team**
  ```
  Slack: #sci-agents-ops
  "🚀 Starting production deployment v0.1.0"
  ```

#### Step 2: Deployment (T+10min)

- [ ] **Build production images**
  ```bash
  docker-compose -f docker-compose.prod.yml build
  ```

- [ ] **Deploy new version**

  **Option A: Rolling deployment (zero downtime)**
  ```bash
  # Deploy one instance at a time
  for instance in app-1 app-2 app-3; do
    docker-compose stop $instance
    docker-compose up -d $instance
    sleep 30
    python scripts/health_check.py --instance=$instance
  done
  ```

  **Option B: Blue-green deployment**
  ```bash
  # Start green environment
  docker-compose -f docker-compose.green.yml up -d

  # Health check
  python scripts/health_check.py --env=green

  # Switch load balancer
  ./scripts/switch_lb.sh green

  # Monitor for 10 minutes
  watch -n 30 python scripts/health_check.py
  ```

  **Option C: Simple deployment (with downtime)**
  ```bash
  # Stop current
  docker-compose -f docker-compose.prod.yml down

  # Start new
  docker-compose -f docker-compose.prod.yml up -d

  # Health check
  python scripts/health_check.py
  ```

#### Step 3: Verification (T+20min)

- [ ] **Health check passed**
  ```bash
  python scripts/health_check.py
  # Expected: Status HEALTHY, exit code 0
  ```

- [ ] **Smoke tests passed**
  ```bash
  pytest tests/ -m "smoke" -v
  ```

- [ ] **Key workflows tested**
  ```bash
  # Test critical paths
  python examples/tutorial_01_quick_start.py
  ```

- [ ] **Monitoring operational**
  - Prometheus scraping targets
  - Grafana dashboards loading
  - Alerts configured

- [ ] **Performance acceptable**
  ```bash
  python scripts/benchmark.py
  # Compare with baseline: <20% regression OK
  ```

#### Step 4: Monitoring Period (T+30min to T+2hr)

- [ ] **Monitor error rates** (every 15 min)
  ```bash
  # Check logs
  docker-compose logs --tail=100 sci-agents | grep ERROR

  # Check metrics
  curl http://prometheus:9090/api/v1/query?query=error_rate
  ```

- [ ] **Monitor resource usage** (every 15 min)
  ```bash
  docker stats
  ```

- [ ] **Monitor user reports**
  - Check Slack #sci-agents-users
  - Check support email
  - Monitor GitHub issues

- [ ] **Verify alerts working**
  ```bash
  # Trigger test alert
  curl -X POST http://alertmanager:9093/api/v1/alerts
  ```

#### Step 5: Post-Deployment (T+2hr)

- [ ] **Announce completion**
  ```
  Slack: #sci-agents-ops
  "✅ Deployment v0.1.0 complete. All systems nominal."

  Email: users@example.com
  Subject: "Scientific Computing Agents v0.1.0 Released"
  ```

- [ ] **Update status page**
  ```
  Status: Operational
  Version: v0.1.0
  Deployed: [Timestamp]
  ```

- [ ] **Tag successful deployment**
  ```bash
  git tag -a v0.1.0-deployed -m "Production deployment $(date)"
  git push origin v0.1.0-deployed
  ```

- [ ] **Document any issues**
  ```markdown
  # Deployment Log v0.1.0

  ## Timeline
  - T+0: Started deployment
  - T+10: Images built
  - T+20: Deployed to production
  - T+30: Health checks passed
  - T+120: Monitoring period complete

  ## Issues
  - [Issue 1]: [Resolution]
  - [Issue 2]: [Resolution]

  ## Metrics
  - Downtime: [Duration]
  - Error rate: [Rate]
  - Performance: [Comparison]
  ```

---

## Post-Deployment Checklist

### Day 1 (First 24 hours)

- [ ] **Monitor continuously**
  - Check dashboards every 2 hours
  - Review error logs
  - Track user feedback

- [ ] **Collect metrics**
  ```bash
  python scripts/feedback_dashboard.py
  ```

- [ ] **Address critical issues**
  - P0: Immediate fix
  - P1: Fix within 24h
  - P2+: Schedule for next sprint

- [ ] **Update documentation**
  - Document any config changes
  - Update runbook with lessons learned

### Week 1

- [ ] **Daily health checks**
  ```bash
  # Automated via cron
  0 9 * * * python scripts/health_check.py
  ```

- [ ] **Review analytics**
  - User adoption rate
  - Feature usage
  - Error patterns

- [ ] **User check-in**
  - Send Week 1 survey
  - Schedule office hours
  - Respond to feedback

- [ ] **Performance review**
  ```bash
  python scripts/benchmark.py
  # Compare with baseline
  ```

### Month 1

- [ ] **Stability assessment**
  - Uptime: Target >99.5%
  - Error rate: Target <1%
  - User satisfaction: Target >3.5/5

- [ ] **Capacity planning**
  - Review resource usage
  - Plan scaling if needed
  - Optimize bottlenecks

- [ ] **Security review**
  ```bash
  python scripts/security_audit.py
  ```

- [ ] **Dependency updates**
  ```bash
  pip list --outdated
  # Update and test in dev first
  ```

---

## Rollback Procedure

### When to Rollback

**Critical issues (rollback immediately)**:
- System completely down
- Data corruption detected
- Security vulnerability exploited
- >50% error rate

**Major issues (consider rollback)**:
- >10% error rate
- Critical feature broken
- Significant performance regression (>50%)

### Rollback Steps

#### Quick Rollback (10 minutes)

```bash
# 1. Stop current version
docker-compose down

# 2. Checkout previous version
git checkout v0.0.9

# 3. Rebuild and deploy
docker-compose build
docker-compose up -d

# 4. Verify
python scripts/health_check.py

# 5. Announce
# Slack: "⚠️ Rolled back to v0.0.9 due to [issue]"
```

#### Database Rollback (if needed)

```bash
# 1. Stop application
docker-compose stop sci-agents

# 2. Restore database
psql -U user -d db < backup_YYYYMMDD.sql

# 3. Restart application
docker-compose start sci-agents

# 4. Verify
python scripts/health_check.py
```

#### Post-Rollback

- [ ] **Analyze failure**
  - Root cause analysis
  - Document findings
  - Plan fix

- [ ] **Communicate**
  - Notify users
  - Update status page
  - Post-mortem (within 48h)

---

## Success Criteria

### Deployment Success

- [ ] Health check: PASSED (exit code 0)
- [ ] Smoke tests: >95% pass
- [ ] Error rate: <2% (first hour), <1% (first day)
- [ ] Performance: Within 20% of baseline
- [ ] Uptime: >99% (first week)

### User Validation Success

- [ ] Active users: 10+
- [ ] User satisfaction: >3.5/5
- [ ] Critical bugs: 0
- [ ] Support requests: Manageable (<5/day)

---

## Emergency Contacts

**On-Call Engineer**: [Phone]
**Backup**: [Phone]
**Manager**: [Phone]

**Escalation**:
1. Primary on-call (immediate)
2. Backup on-call (+15min)
3. Manager (+30min)
4. CTO (critical data loss only)

---

## Appendix

### Useful Commands

```bash
# Health check
python scripts/health_check.py

# View logs
docker-compose logs -f --tail=100 sci-agents

# Restart service
docker-compose restart sci-agents

# Check resource usage
docker stats

# Run benchmarks
python scripts/benchmark.py

# Security audit
python scripts/security_audit.py
```

### Configuration Files

- **Production config**: `config/production.yaml`
- **Environment**: `.env.production`
- **Docker compose**: `docker-compose.prod.yml`
- **Secrets**: AWS Secrets Manager / Vault

### Documentation Links

- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Operations Runbook**: `docs/OPERATIONS_RUNBOOK.md`
- **User Onboarding**: `docs/USER_ONBOARDING.md`

---

**Document Version**: 1.0
**Owner**: DevOps Team
**Last Tested**: 2025-10-01
**Next Review**: After first production deployment
