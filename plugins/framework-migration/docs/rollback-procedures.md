# Rollback & Recovery Procedures

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Safety Guide

Emergency rollback procedures and recovery strategies for failed migrations.

---

## Quick Rollback Decision Tree

```
Migration deployed to production?
├─ Yes → Is there an incident?
│   ├─ Yes → IMMEDIATE ROLLBACK (see below)
│   └─ No → Monitor closely, prepare rollback plan
└─ No → Safe to experiment, use git rollback
```

---

## Immediate Rollback Procedures

### Git-Based Rollback (< 1 hour since deploy)

```bash
# 1. Revert the merge commit
git revert -m 1 HEAD
git push origin main

# 2. Trigger re-deployment
./deploy.sh production

# 3. Verify rollback
curl https://api.example.com/health
```

### Feature Flag Rollback (< 1 minute)

```javascript
// Emergency disable
featureFlags.disable('new-payment-flow');
// Takes effect immediately for new requests
```

### Blue-Green Rollback (< 5 minutes)

```bash
# Switch load balancer back to blue (old) environment
aws elbv2 modify-listener --listener-arn $LISTENER_ARN \
  --default-actions Type=forward,TargetGroupArn=$BLUE_TG

# Verify traffic switched
watch 'aws elbv2 describe-target-health --target-group-arn $BLUE_TG'
```

---

## Rollback Triggers

**Immediate Rollback If**:
- Error rate > 5% (vs baseline < 1%)
- p95 latency > 2x baseline
- Any data corruption detected
- Critical functionality broken
- Security vulnerability introduced

---

## Post-Rollback Actions

1. ✅ Verify system stable on old version
2. ✅ Communicate to stakeholders
3. ✅ Begin root cause analysis
4. ✅ Document what went wrong
5. ✅ Plan remediation strategy
6. ✅ Add tests to prevent recurrence

---

**For migration safety strategies**, see `/code-migrate` and `/legacy-modernize` commands.
