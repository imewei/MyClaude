# Testing Strategies for Migrations

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Reference

Comprehensive testing approaches for safe code migrations, dependency upgrades, and legacy modernization.

---

## Characterization Tests

**Purpose**: Capture current behavior before refactoring to ensure no regressions.

**Golden Master Pattern**:
```python
def test_legacy_payment_processor():
    """Characterization test - captures current behavior."""
    # Setup
    order = create_test_order(amount=100.00, items=3)

    # Execute legacy implementation
    result = legacy_payment_processor.process(order)

    # Assert current behavior (even if not ideal)
    assert result.status == "pending"  # Current behavior
    assert result.transaction_id.startswith("TXN_")
    assert len(result.receipt_items) == 3
```

---

## Contract Tests

**Purpose**: Validate integration points remain compatible.

**Consumer-Driven Contracts**:
```javascript
// Consumer test
describe('Payment API Contract', () => {
  it('should return valid payment response', async () => {
    const response = await paymentAPI.process({
      amount: 100,
      currency: 'USD'
    });

    expect(response).toMatchSchema({
      status: expect.stringMatching(/^(success|pending|failed)$/),
      transactionId: expect.any(String),
      timestamp: expect.any(Number)
    });
  });
});
```

---

## Performance Baseline Tests

**Purpose**: Ensure new implementation maintains SLAs.

```python
@pytest.mark.benchmark
def test_performance_baseline(benchmark):
    """Baseline: p95 latency < 500ms"""
    result = benchmark(payment_processor.process, test_order)
    assert benchmark.stats['max'] < 0.5  # 500ms
```

---

## Parallel Run Testing

**Purpose**: Validate new implementation against legacy in production.

```python
def process_payment_with_validation(order):
    # Primary: legacy (production)
    legacy_result = legacy_processor.process(order)

    # Shadow: modern (validation only)
    try:
        modern_result = modern_processor.process(order)
        compare_results(legacy_result, modern_result)
    except Exception as e:
        logger.error(f"Modern implementation error: {e}")

    return legacy_result  # Always return legacy
```

---

**For full testing workflows**, see `/code-migrate` and `/legacy-modernize` commands.
