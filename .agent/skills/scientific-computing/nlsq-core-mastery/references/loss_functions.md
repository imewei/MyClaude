# NLSQ Loss Functions Reference

## Loss Function Theory

Loss functions determine how residuals are weighted in the cost function:

```
C(p) = Σᵢ ρ(rᵢ(p))
```

where `rᵢ = yᵢ - f(xᵢ, p)` is the residual.

## Available Loss Functions

### 1. Linear (Standard Least Squares)

**Formula:**
```
ρ(r) = r²
```

**Characteristics:**
- Outliers heavily penalized (quadratic)
- Optimal for Gaussian noise
- Fastest computation
- No robustness

**Use when:**
- Data is clean
- Noise is normally distributed
- No outliers expected
- Speed is critical

**Avoid when:**
- Outliers present
- Heavy-tailed noise
- Measurement errors

### 2. Soft L1

**Formula:**
```
ρ(r) = 2((1 + r²)^0.5 - 1)
```

**Characteristics:**
- Smooth transition at r=0
- Linear for large residuals
- Good balance of speed and robustness
- Mild outlier rejection

**Use when:**
- Few outliers (<5%)
- Smooth robustness desired
- Good first choice for real data

**Gradient:**
```
ρ'(r) = r / (1 + r²)^0.5
```

### 3. Huber

**Formula:**
```
ρ(r) = r² / 2           if |r| ≤ δ
       δ|r| - δ²/2      if |r| > δ
```

where δ is the transition point (typically δ=1).

**Characteristics:**
- Quadratic for small residuals
- Linear for large residuals
- Well-studied properties
- Moderate outlier rejection

**Use when:**
- Moderate outliers (5-10%)
- Well-understood robustness needed
- Standard robust method desired

**Tuning:**
```python
# NLSQ uses fixed δ=1
# To adjust sensitivity, scale data
y_scaled = y / scale_factor
# Fit with huber loss
# Unscale results
```

### 4. Cauchy (Lorentzian)

**Formula:**
```
ρ(r) = log(1 + r²)
```

**Characteristics:**
- Logarithmic growth
- Strong outlier rejection
- Slower convergence
- May need more iterations

**Use when:**
- Many outliers (10-20%)
- Extreme robustness needed
- Can afford extra iterations

**Warning:**
- May downweight too many points
- Check that fit uses enough data

### 5. Arctan

**Formula:**
```
ρ(r) = arctan(r²)
```

**Characteristics:**
- Strongest outlier rejection
- Bounded growth
- Slowest convergence
- Most robust

**Use when:**
- Extreme outliers (>20%)
- Adversarial data
- Last resort for robustness

**Warning:**
- May lose too much information
- Consider data cleaning instead
- Convergence may be difficult

## Decision Tree

```
Start
│
├─ Is data clean with Gaussian noise?
│  └─ YES → Use 'linear'
│      - Fastest
│      - Optimal for normal errors
│      - Standard least squares
│
├─ Are there a few outliers (<5%)?
│  └─ YES → Use 'soft_l1' or 'huber'
│      - Good balance
│      - Mild robustness
│      - Fast convergence
│
├─ Are there many outliers (5-20%)?
│  └─ YES → Use 'cauchy'
│      - Strong robustness
│      - Moderate speed
│      - May need more iterations
│
└─ Are outliers extreme (>20%)?
   └─ YES → Consider:
       1. Data cleaning first
       2. If cleaning not possible, use 'arctan'
       3. Expect slow convergence
       4. Verify fit quality
```

## Comparison Examples

### Example 1: Clean Data

```python
# Generate clean data
x = jnp.linspace(0, 10, 100)
y_clean = 2 * x + 1
y = y_clean + jax.random.normal(key, y_clean.shape) * 0.1

# Linear is optimal
for loss in ['linear', 'soft_l1', 'huber']:
    result = CurveFit(model, x, y, p0, loss=loss).fit()
    print(f"{loss}: params={result.x}, cost={result.cost:.2e}")

# Output:
# linear:  params=[2.00, 1.00], cost=1.05e-01
# soft_l1: params=[2.00, 1.00], cost=1.06e-01  (similar)
# huber:   params=[2.00, 1.00], cost=1.06e-01  (similar)
```

### Example 2: Data with Outliers

```python
# Add 10% outliers
outlier_mask = jax.random.bernoulli(key, 0.1, y.shape)
y_outliers = jnp.where(outlier_mask,
                       y_clean + 10 * jax.random.normal(key, y.shape),
                       y)

# Compare loss functions
for loss in ['linear', 'soft_l1', 'huber', 'cauchy']:
    result = CurveFit(model, x, y_outliers, p0, loss=loss).fit()
    param_error = jnp.linalg.norm(result.x - true_params)
    print(f"{loss:8s}: error={param_error:.3f}, cost={result.cost:.2e}")

# Output:
# linear:  error=0.523, cost=9.87e+01  (Biased by outliers!)
# soft_l1: error=0.112, cost=1.45e+01  (Better)
# huber:   error=0.098, cost=1.32e+01  (Better)
# cauchy:  error=0.045, cost=4.21e+00  (Best for outliers)
```

## Impact on Convergence

| Loss | Iterations (typical) | Speed | Outlier Handling |
|------|---------------------|-------|------------------|
| linear | 10-20 | Very Fast | None |
| soft_l1 | 15-30 | Fast | Mild |
| huber | 15-35 | Fast | Moderate |
| cauchy | 30-60 | Moderate | Strong |
| arctan | 50-100 | Slow | Very Strong |

## Weighting Comparison

How much weight each loss gives to different residual magnitudes:

```
Residual (r) | linear | soft_l1 | huber | cauchy | arctan
-------------|---------|---------|-------|--------|--------
0.1          | 100%    | 99%     | 100%  | 98%    | 95%
0.5          | 100%    | 89%     | 100%  | 80%    | 60%
1.0          | 100%    | 71%     | 100%  | 50%    | 35%
2.0          | 100%    | 45%     | 50%   | 20%    | 15%
5.0          | 100%    | 20%     | 20%   | 4%     | 3%
10.0         | 100%    | 10%     | 10%   | 1%     | 1%
```

**Interpretation:**
- Linear: All residuals weighted fully (100%)
- Soft L1: Starts downweighting at r≈1
- Huber: Downweights beyond δ=1
- Cauchy: Strong downweighting for r>1
- Arctan: Extreme downweighting for all r>1

## When to Switch Loss Functions

### Signs you need more robustness:
- Fitted parameters change drastically with small data changes
- Residual plots show clear outliers
- Cost dominated by a few large residuals
- Physical parameters unreasonable

### Signs you have too much robustness:
- Fit doesn't follow majority of data
- Cost is suspiciously low
- Many "good" points ignored
- Underfitting apparent

### Switching strategy:
1. Start with `linear` for clean data
2. If outliers suspected, try `soft_l1`
3. If still poor, try `huber`
4. If extreme outliers, try `cauchy`
5. Only use `arctan` as last resort

## Custom Loss Functions

NLSQ supports only the built-in loss functions. For custom loss:

**Option 1: Transform data**
```python
# Apply transformation to approximate desired loss
y_transformed = custom_transform(y)
result = CurveFit(model, x, y_transformed, p0, loss='linear').fit()
```

**Option 2: Reweighting (if convergence allows)**
```python
# Iteratively reweight based on residuals
weights = jnp.ones_like(y)
for iteration in range(10):
    # Fit with weights
    result = CurveFit(weighted_model, x, y, p0).fit()
    # Update weights based on residuals
    residuals = y - model(x, result.x)
    weights = custom_weight_function(residuals)
```

**Option 3: Manual implementation**
```python
# Implement full optimizer with custom loss
# (beyond scope of NLSQ, use optax or scipy)
```

## Loss Function Visualization

To understand loss behavior:

```python
import matplotlib.pyplot as plt

r = jnp.linspace(-5, 5, 200)

# Compute loss values
linear = r**2
soft_l1 = 2 * (jnp.sqrt(1 + r**2) - 1)
huber = jnp.where(jnp.abs(r) <= 1, r**2 / 2, jnp.abs(r) - 0.5)
cauchy = jnp.log(1 + r**2)
arctan = jnp.arctan(r**2)

plt.figure(figsize=(10, 6))
plt.plot(r, linear, label='linear', linewidth=2)
plt.plot(r, soft_l1, label='soft_l1', linewidth=2)
plt.plot(r, huber, label='huber', linewidth=2)
plt.plot(r, cauchy, label='cauchy', linewidth=2)
plt.plot(r, arctan, label='arctan', linewidth=2)
plt.xlabel('Residual (r)')
plt.ylabel('Loss ρ(r)')
plt.title('Loss Function Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-5, 5])
plt.show()
```
