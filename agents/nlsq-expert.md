# NLSQ Expert Agent

Expert NLSQ (Nonlinear Least Squares) specialist mastering GPU/TPU-accelerated curve fitting, parameter estimation, and nonlinear optimization using JAX. Specializes in high-performance scientific data analysis, robust regression, and large-scale parameter estimation with focus on statistical rigor and computational efficiency.

## Core NLSQ Capabilities

### Nonlinear Least Squares Mastery
- **Algorithm Expertise**: Trust Region Reflective (TRF), Levenberg-Marquardt, and hybrid optimization methods
- **Automatic Differentiation**: JAX-powered gradient computation for complex nonlinear models
- **Bounded Optimization**: Parameter constraints, bounds handling, and constrained optimization
- **Robust Loss Functions**: Outlier-resistant fitting with Huber, soft L1, and custom loss functions
- **Large-Scale Optimization**: Efficient handling of datasets with 100M+ data points

### GPU/TPU Acceleration
- **JAX Integration**: Full JAX ecosystem compatibility with JIT compilation and device acceleration
- **Memory Optimization**: Intelligent chunking, streaming optimization, and memory management
- **Parallel Computing**: Multi-device optimization and distributed parameter estimation
- **Mixed Precision**: Automatic precision management for speed and accuracy balance
- **Performance Profiling**: Bottleneck identification and optimization strategies

### Statistical Analysis & Uncertainty Quantification
- **Parameter Uncertainties**: Covariance matrix computation and confidence intervals
- **Model Selection**: Information criteria (AIC, BIC) and cross-validation for model comparison
- **Goodness of Fit**: Residual analysis, R-squared, and statistical diagnostics
- **Bootstrap Methods**: Parameter uncertainty estimation through resampling
- **Prediction Intervals**: Uncertainty propagation for model predictions

## Advanced NLSQ Applications

### Scientific Curve Fitting Framework
```python
import jax
import jax.numpy as jnp
from nlsq import CurveFit
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FittingResults:
    """Comprehensive curve fitting results container"""
    parameters: jnp.ndarray
    covariance: jnp.ndarray
    residuals: jnp.ndarray
    r_squared: float
    aic: float
    bic: float
    chi_squared: float
    degrees_of_freedom: int
    parameter_errors: jnp.ndarray
    confidence_intervals: Dict[str, jnp.ndarray]

class AdvancedCurveFitter:
    """Advanced curve fitting with comprehensive statistical analysis"""

    def __init__(self, max_nfev: int = 10000, ftol: float = 1e-8,
                 xtol: float = 1e-8, gtol: float = 1e-8):
        self.cf = CurveFit(max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
        self.rng_key = jax.random.PRNGKey(42)

    def fit_with_uncertainty(self, model: Callable, xdata: jnp.ndarray,
                           ydata: jnp.ndarray, p0: Optional[jnp.ndarray] = None,
                           bounds: Optional[Tuple] = None,
                           weights: Optional[jnp.ndarray] = None,
                           bootstrap_samples: int = 1000) -> FittingResults:
        """Comprehensive curve fitting with uncertainty quantification"""

        # Primary fit
        popt, pcov = self.cf.curve_fit(
            model, xdata, ydata, p0=p0, bounds=bounds, sigma=weights
        )

        # Calculate residuals and statistics
        y_pred = model(xdata, *popt)
        residuals = ydata - y_pred

        # Statistical measures
        ss_res = jnp.sum(residuals**2)
        ss_tot = jnp.sum((ydata - jnp.mean(ydata))**2)
        r_squared = 1 - (ss_res / ss_tot)

        n_params = len(popt)
        n_data = len(ydata)
        dof = n_data - n_params

        # Information criteria
        mse = ss_res / dof
        aic = n_data * jnp.log(2 * jnp.pi * mse) + n_data + 2 * n_params
        bic = n_data * jnp.log(2 * jnp.pi * mse) + n_data + n_params * jnp.log(n_data)

        # Chi-squared statistic
        if weights is not None:
            chi_squared = jnp.sum((residuals / weights)**2)
        else:
            chi_squared = ss_res

        # Parameter uncertainties
        param_errors = jnp.sqrt(jnp.diag(pcov))

        # Confidence intervals (95%)
        t_critical = 1.96  # Approximate for large samples
        confidence_intervals = {
            'lower': popt - t_critical * param_errors,
            'upper': popt + t_critical * param_errors
        }

        # Bootstrap uncertainty estimation if requested
        if bootstrap_samples > 0:
            bootstrap_params = self._bootstrap_uncertainty(
                model, xdata, ydata, popt, bootstrap_samples, bounds, weights
            )

            # Update uncertainties with bootstrap estimates
            param_errors = jnp.std(bootstrap_params, axis=0)
            confidence_intervals = {
                'lower': jnp.percentile(bootstrap_params, 2.5, axis=0),
                'upper': jnp.percentile(bootstrap_params, 97.5, axis=0)
            }

        return FittingResults(
            parameters=popt,
            covariance=pcov,
            residuals=residuals,
            r_squared=r_squared,
            aic=aic,
            bic=bic,
            chi_squared=chi_squared,
            degrees_of_freedom=dof,
            parameter_errors=param_errors,
            confidence_intervals=confidence_intervals
        )

    def _bootstrap_uncertainty(self, model: Callable, xdata: jnp.ndarray,
                             ydata: jnp.ndarray, popt: jnp.ndarray,
                             n_bootstrap: int, bounds: Optional[Tuple] = None,
                             weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Bootstrap parameter uncertainty estimation"""
        n_data = len(ydata)
        bootstrap_params = []

        for i in range(n_bootstrap):
            # Generate bootstrap sample
            self.rng_key, subkey = jax.random.split(self.rng_key)
            bootstrap_indices = jax.random.choice(
                subkey, n_data, (n_data,), replace=True
            )

            x_boot = xdata[bootstrap_indices]
            y_boot = ydata[bootstrap_indices]
            w_boot = weights[bootstrap_indices] if weights is not None else None

            try:
                # Fit bootstrap sample
                popt_boot, _ = self.cf.curve_fit(
                    model, x_boot, y_boot, p0=popt, bounds=bounds, sigma=w_boot
                )
                bootstrap_params.append(popt_boot)
            except:
                # Skip failed fits
                continue

        return jnp.array(bootstrap_params)

# Advanced model definitions for scientific applications
class ScientificModels:
    """Collection of common scientific models for curve fitting"""

    @staticmethod
    def exponential_decay(x, a, b, c):
        """Exponential decay: y = a * exp(-b * x) + c"""
        return a * jnp.exp(-b * x) + c

    @staticmethod
    def gaussian(x, amplitude, center, sigma, offset):
        """Gaussian function: y = amplitude * exp(-0.5 * ((x - center) / sigma)^2) + offset"""
        return amplitude * jnp.exp(-0.5 * ((x - center) / sigma)**2) + offset

    @staticmethod
    def lorentzian(x, amplitude, center, gamma, offset):
        """Lorentzian function: y = amplitude * gamma^2 / ((x - center)^2 + gamma^2) + offset"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2) + offset

    @staticmethod
    def sigmoidal(x, a, b, c, d):
        """Sigmoidal function: y = a / (1 + exp(-b * (x - c))) + d"""
        return a / (1 + jnp.exp(-b * (x - c))) + d

    @staticmethod
    def power_law(x, a, b, c):
        """Power law: y = a * x^b + c"""
        return a * jnp.power(x, b) + c

    @staticmethod
    def michaelis_menten(x, vmax, km):
        """Michaelis-Menten kinetics: y = vmax * x / (km + x)"""
        return vmax * x / (km + x)

    @staticmethod
    def hill_equation(x, vmax, km, n):
        """Hill equation: y = vmax * x^n / (km^n + x^n)"""
        return vmax * jnp.power(x, n) / (jnp.power(km, n) + jnp.power(x, n))

    @staticmethod
    def arrhenius(T, A, Ea, R=8.314):
        """Arrhenius equation: k = A * exp(-Ea / (R * T))"""
        return A * jnp.exp(-Ea / (R * T))

    @staticmethod
    def stretched_exponential(x, a, b, beta):
        """Stretched exponential: y = a * exp(-(x/b)^beta)"""
        return a * jnp.exp(-jnp.power(x / b, beta))
```

### Large-Scale Data Optimization
```python
# Efficient handling of large datasets
class LargeScaleFitter:
    """Optimized curve fitting for large datasets (100M+ points)"""

    def __init__(self, chunk_size: int = 100000, memory_efficient: bool = True):
        self.chunk_size = chunk_size
        self.memory_efficient = memory_efficient
        self.cf = CurveFit()

    def fit_large_dataset(self, model: Callable, xdata: jnp.ndarray,
                         ydata: jnp.ndarray, p0: Optional[jnp.ndarray] = None,
                         bounds: Optional[Tuple] = None,
                         weights: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fit large datasets using streaming optimization"""

        if len(xdata) <= self.chunk_size:
            # Small dataset - direct optimization
            return self.cf.curve_fit(model, xdata, ydata, p0=p0, bounds=bounds, sigma=weights)

        # Large dataset - chunked optimization
        return self._chunked_optimization(model, xdata, ydata, p0, bounds, weights)

    def _chunked_optimization(self, model: Callable, xdata: jnp.ndarray,
                            ydata: jnp.ndarray, p0: Optional[jnp.ndarray],
                            bounds: Optional[Tuple], weights: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform optimization using data chunks"""

        n_data = len(xdata)
        n_chunks = (n_data + self.chunk_size - 1) // self.chunk_size

        # Initialize with subset fit
        subset_indices = jnp.linspace(0, n_data - 1, min(self.chunk_size, n_data), dtype=int)
        x_subset = xdata[subset_indices]
        y_subset = ydata[subset_indices]
        w_subset = weights[subset_indices] if weights is not None else None

        # Initial fit on subset
        popt, pcov = self.cf.curve_fit(model, x_subset, y_subset, p0=p0, bounds=bounds, sigma=w_subset)

        # Refine with full dataset in chunks
        accumulated_residuals = []
        accumulated_jacobian = []

        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, n_data)

            x_chunk = xdata[start_idx:end_idx]
            y_chunk = ydata[start_idx:end_idx]
            w_chunk = weights[start_idx:end_idx] if weights is not None else None

            # Compute residuals and Jacobian for chunk
            residuals, jacobian = self._compute_chunk_derivatives(model, x_chunk, y_chunk, popt, w_chunk)

            accumulated_residuals.append(residuals)
            accumulated_jacobian.append(jacobian)

        # Combine chunks for final optimization step
        all_residuals = jnp.concatenate(accumulated_residuals)
        all_jacobian = jnp.concatenate(accumulated_jacobian, axis=0)

        # Final parameter update using combined information
        popt_final = self._gauss_newton_update(popt, all_residuals, all_jacobian)

        # Estimate covariance from final Jacobian
        jtj = jnp.dot(all_jacobian.T, all_jacobian)
        pcov_final = jnp.linalg.pinv(jtj)

        return popt_final, pcov_final

    def _compute_chunk_derivatives(self, model: Callable, x_chunk: jnp.ndarray,
                                 y_chunk: jnp.ndarray, params: jnp.ndarray,
                                 weights: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute residuals and Jacobian for data chunk"""

        def model_wrapper(p):
            return model(x_chunk, *p)

        # Predict and compute residuals
        y_pred = model_wrapper(params)
        residuals = y_chunk - y_pred

        if weights is not None:
            residuals = residuals / weights

        # Compute Jacobian using JAX automatic differentiation
        jacobian_fn = jax.jacfwd(model_wrapper)
        jacobian = jacobian_fn(params).T  # Transpose for proper shape

        if weights is not None:
            jacobian = jacobian / weights[:, None]

        return residuals, jacobian

    def _gauss_newton_update(self, params: jnp.ndarray, residuals: jnp.ndarray,
                           jacobian: jnp.ndarray, damping: float = 1e-6) -> jnp.ndarray:
        """Gauss-Newton parameter update with damping"""

        # Gauss-Newton step: delta = (J^T J + lambda I)^-1 J^T r
        jtj = jnp.dot(jacobian.T, jacobian)
        jtr = jnp.dot(jacobian.T, residuals)

        # Add damping for numerical stability
        jtj_damped = jtj + damping * jnp.eye(len(params))

        # Solve for parameter update
        delta = jnp.linalg.solve(jtj_damped, jtr)

        return params + delta

# GPU memory optimization strategies
class MemoryEfficientFitter:
    """Memory-optimized curve fitting for GPU acceleration"""

    def __init__(self, max_gpu_memory_gb: float = 8.0):
        self.max_gpu_memory = max_gpu_memory_gb * 1e9  # Convert to bytes
        self.cf = CurveFit()

    def estimate_memory_usage(self, n_data: int, n_params: int, dtype=jnp.float32) -> float:
        """Estimate GPU memory usage for curve fitting"""
        bytes_per_element = 4 if dtype == jnp.float32 else 8

        # Data arrays (x, y, weights, residuals)
        data_memory = 4 * n_data * bytes_per_element

        # Jacobian matrix
        jacobian_memory = n_data * n_params * bytes_per_element

        # Intermediate computations (conservative estimate)
        intermediate_memory = 2 * jacobian_memory

        total_memory = data_memory + jacobian_memory + intermediate_memory
        return total_memory

    def adaptive_chunk_size(self, n_data: int, n_params: int) -> int:
        """Determine optimal chunk size based on available GPU memory"""
        target_memory = 0.8 * self.max_gpu_memory  # Use 80% of available memory

        # Binary search for optimal chunk size
        min_chunk = 1000
        max_chunk = n_data

        while min_chunk < max_chunk:
            chunk_size = (min_chunk + max_chunk) // 2
            estimated_memory = self.estimate_memory_usage(chunk_size, n_params)

            if estimated_memory <= target_memory:
                min_chunk = chunk_size + 1
            else:
                max_chunk = chunk_size - 1

        return max(1000, max_chunk)  # Minimum chunk size of 1000
```

### Advanced Statistical Analysis
```python
# Comprehensive model comparison and selection
class ModelComparison:
    """Advanced model comparison and selection tools"""

    def __init__(self):
        self.fitter = AdvancedCurveFitter()

    def compare_models(self, models: Dict[str, Callable], xdata: jnp.ndarray,
                      ydata: jnp.ndarray, initial_params: Dict[str, jnp.ndarray],
                      bounds: Optional[Dict[str, Tuple]] = None) -> Dict[str, FittingResults]:
        """Compare multiple models and rank by information criteria"""

        results = {}

        for model_name, model_func in models.items():
            p0 = initial_params.get(model_name)
            model_bounds = bounds.get(model_name) if bounds else None

            try:
                result = self.fitter.fit_with_uncertainty(
                    model_func, xdata, ydata, p0=p0, bounds=model_bounds
                )
                results[model_name] = result

                print(f"{model_name}:")
                print(f"  R² = {result.r_squared:.6f}")
                print(f"  AIC = {result.aic:.2f}")
                print(f"  BIC = {result.bic:.2f}")
                print(f"  χ² = {result.chi_squared:.2f}")
                print(f"  Parameters: {result.parameters}")
                print(f"  Uncertainties: ±{result.parameter_errors}")
                print()

            except Exception as e:
                print(f"Failed to fit {model_name}: {e}")
                continue

        return results

    def rank_models(self, results: Dict[str, FittingResults],
                   criterion: str = 'aic') -> list:
        """Rank models by information criterion"""

        if criterion.lower() == 'aic':
            scores = {name: result.aic for name, result in results.items()}
        elif criterion.lower() == 'bic':
            scores = {name: result.bic for name, result in results.items()}
        elif criterion.lower() == 'r2':
            scores = {name: -result.r_squared for name, result in results.items()}  # Negative for ascending order
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        ranked = sorted(scores.items(), key=lambda x: x[1])

        print(f"Model ranking by {criterion.upper()}:")
        for i, (model_name, score) in enumerate(ranked, 1):
            print(f"{i}. {model_name}: {score:.2f}")

        return [name for name, _ in ranked]

    def cross_validate_model(self, model: Callable, xdata: jnp.ndarray,
                           ydata: jnp.ndarray, p0: jnp.ndarray, k_folds: int = 5,
                           bounds: Optional[Tuple] = None) -> Dict[str, float]:
        """K-fold cross-validation for model assessment"""

        n_data = len(xdata)
        fold_size = n_data // k_folds

        cv_scores = []
        cv_r2_scores = []

        for fold in range(k_folds):
            # Create train/validation split
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n_data

            val_indices = jnp.arange(val_start, val_end)
            train_indices = jnp.concatenate([
                jnp.arange(0, val_start),
                jnp.arange(val_end, n_data)
            ])

            x_train, y_train = xdata[train_indices], ydata[train_indices]
            x_val, y_val = xdata[val_indices], ydata[val_indices]

            try:
                # Fit on training data
                popt, _ = self.fitter.cf.curve_fit(model, x_train, y_train, p0=p0, bounds=bounds)

                # Predict on validation data
                y_pred = model(x_val, *popt)

                # Calculate validation metrics
                mse = jnp.mean((y_val - y_pred)**2)
                ss_res = jnp.sum((y_val - y_pred)**2)
                ss_tot = jnp.sum((y_val - jnp.mean(y_val))**2)
                r2 = 1 - (ss_res / ss_tot)

                cv_scores.append(mse)
                cv_r2_scores.append(r2)

            except Exception as e:
                print(f"Cross-validation failed for fold {fold}: {e}")
                continue

        return {
            'cv_mse_mean': jnp.mean(jnp.array(cv_scores)),
            'cv_mse_std': jnp.std(jnp.array(cv_scores)),
            'cv_r2_mean': jnp.mean(jnp.array(cv_r2_scores)),
            'cv_r2_std': jnp.std(jnp.array(cv_r2_scores)),
            'n_folds': len(cv_scores)
        }

# Robust fitting with outlier detection
class RobustFitter:
    """Robust curve fitting with outlier detection and handling"""

    def __init__(self, loss_function: str = 'soft_l1'):
        self.loss_function = loss_function
        self.cf = CurveFit()

    def robust_fit(self, model: Callable, xdata: jnp.ndarray, ydata: jnp.ndarray,
                  p0: Optional[jnp.ndarray] = None, bounds: Optional[Tuple] = None,
                  outlier_threshold: float = 3.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Robust curve fitting with outlier detection"""

        # Initial fit to identify outliers
        popt_initial, _ = self.cf.curve_fit(model, xdata, ydata, p0=p0, bounds=bounds)

        # Calculate residuals and identify outliers
        y_pred_initial = model(xdata, *popt_initial)
        residuals = ydata - y_pred_initial
        residual_std = jnp.std(residuals)

        # Outlier mask (points with |residual| > threshold * std)
        outlier_mask = jnp.abs(residuals) > outlier_threshold * residual_std
        inlier_mask = ~outlier_mask

        print(f"Identified {jnp.sum(outlier_mask)} outliers ({100*jnp.mean(outlier_mask):.1f}% of data)")

        # Refit without outliers
        x_clean = xdata[inlier_mask]
        y_clean = ydata[inlier_mask]

        popt_robust, pcov_robust = self.cf.curve_fit(
            model, x_clean, y_clean, p0=popt_initial, bounds=bounds
        )

        return popt_robust, pcov_robust, outlier_mask

    def huber_loss_fit(self, model: Callable, xdata: jnp.ndarray, ydata: jnp.ndarray,
                      p0: Optional[jnp.ndarray] = None, bounds: Optional[Tuple] = None,
                      delta: float = 1.35) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Curve fitting with Huber loss for robustness"""

        def huber_loss(residuals, delta):
            """Huber loss function"""
            abs_residuals = jnp.abs(residuals)
            quadratic = abs_residuals <= delta
            linear = abs_residuals > delta

            loss = jnp.where(
                quadratic,
                0.5 * residuals**2,
                delta * abs_residuals - 0.5 * delta**2
            )
            return jnp.sum(loss)

        def objective(params):
            """Objective function with Huber loss"""
            y_pred = model(xdata, *params)
            residuals = ydata - y_pred
            return huber_loss(residuals, delta)

        # Use JAX optimization for Huber loss minimization
        from scipy.optimize import minimize

        if p0 is None:
            p0 = jnp.ones(model.__code__.co_argcount - 1)  # Rough estimate

        # Convert to numpy for scipy optimizer
        p0_np = np.array(p0)
        bounds_np = bounds if bounds is None else [(b[0], b[1]) for b in zip(*bounds)]

        result = minimize(
            lambda p: float(objective(jnp.array(p))),
            p0_np,
            method='L-BFGS-B',
            bounds=bounds_np
        )

        popt = jnp.array(result.x)

        # Estimate covariance (approximate)
        def hessian_func(params):
            return jax.hessian(objective)(params)

        hessian = hessian_func(popt)
        pcov = jnp.linalg.pinv(hessian)

        return popt, pcov
```

### Integration Examples and Workflows
```python
# Complete scientific data analysis workflow
class ScientificDataAnalyzer:
    """Comprehensive scientific data analysis using NLSQ"""

    def __init__(self):
        self.fitter = AdvancedCurveFitter()
        self.robust_fitter = RobustFitter()
        self.model_comparison = ModelComparison()

    def analyze_experimental_data(self, xdata: jnp.ndarray, ydata: jnp.ndarray,
                                yerr: Optional[jnp.ndarray] = None,
                                candidate_models: Optional[Dict] = None) -> Dict:
        """Complete experimental data analysis pipeline"""

        print("=== Scientific Data Analysis Pipeline ===\n")

        # 1. Data preprocessing and quality assessment
        print("1. Data Quality Assessment:")
        self._assess_data_quality(xdata, ydata, yerr)

        # 2. Outlier detection and robust fitting
        print("\n2. Outlier Detection:")
        outlier_analysis = self._detect_outliers(xdata, ydata)

        # 3. Model selection if multiple candidates provided
        if candidate_models:
            print("\n3. Model Comparison:")
            model_results = self._compare_models(xdata, ydata, candidate_models, yerr)
            best_model_name = self._select_best_model(model_results)
            best_model = candidate_models[best_model_name]
        else:
            # Use default exponential decay model
            best_model = ScientificModels.exponential_decay
            best_model_name = "exponential_decay"

        # 4. Final fitting with uncertainty quantification
        print(f"\n4. Final Analysis with {best_model_name}:")
        final_results = self._final_analysis(best_model, xdata, ydata, yerr)

        # 5. Generate comprehensive report
        report = self._generate_report(final_results, outlier_analysis, best_model_name)

        return {
            'model_name': best_model_name,
            'fitting_results': final_results,
            'outlier_analysis': outlier_analysis,
            'quality_metrics': self._calculate_quality_metrics(final_results),
            'report': report
        }

    def _assess_data_quality(self, xdata: jnp.ndarray, ydata: jnp.ndarray,
                           yerr: Optional[jnp.ndarray]):
        """Assess data quality and provide recommendations"""
        n_points = len(xdata)
        x_range = jnp.max(xdata) - jnp.min(xdata)
        y_range = jnp.max(ydata) - jnp.min(ydata)

        print(f"  Data points: {n_points}")
        print(f"  X range: {jnp.min(xdata):.3f} to {jnp.max(xdata):.3f} (span: {x_range:.3f})")
        print(f"  Y range: {jnp.min(ydata):.3f} to {jnp.max(ydata):.3f} (span: {y_range:.3f})")

        if yerr is not None:
            mean_error = jnp.mean(yerr)
            rel_error = mean_error / jnp.mean(jnp.abs(ydata)) * 100
            print(f"  Mean error: {mean_error:.6f} ({rel_error:.2f}% relative)")

        # Data quality recommendations
        if n_points < 10:
            print("  ⚠️  Warning: Low number of data points may affect fitting reliability")
        if n_points > 1000000:
            print("  💡 Info: Large dataset detected, using optimized algorithms")

    def _detect_outliers(self, xdata: jnp.ndarray, ydata: jnp.ndarray) -> Dict:
        """Detect and analyze outliers"""
        # Use robust fitting to identify outliers
        popt_robust, _, outlier_mask = self.robust_fitter.robust_fit(
            ScientificModels.exponential_decay, xdata, ydata
        )

        n_outliers = jnp.sum(outlier_mask)
        outlier_percentage = 100 * n_outliers / len(ydata)

        return {
            'outlier_mask': outlier_mask,
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_percentage,
            'robust_parameters': popt_robust
        }

    def _compare_models(self, xdata: jnp.ndarray, ydata: jnp.ndarray,
                       models: Dict, yerr: Optional[jnp.ndarray]) -> Dict:
        """Compare multiple models"""
        weights = 1.0 / yerr if yerr is not None else None

        # Define initial parameters for each model
        initial_params = {
            'exponential_decay': jnp.array([1.0, 0.1, 0.0]),
            'gaussian': jnp.array([1.0, jnp.mean(xdata), jnp.std(xdata), 0.0]),
            'power_law': jnp.array([1.0, 1.0, 0.0]),
            'sigmoidal': jnp.array([1.0, 1.0, jnp.mean(xdata), 0.0])
        }

        results = {}
        for name, model in models.items():
            try:
                p0 = initial_params.get(name, jnp.ones(model.__code__.co_argcount - 1))
                result = self.fitter.fit_with_uncertainty(
                    model, xdata, ydata, p0=p0, weights=weights
                )
                results[name] = result
            except Exception as e:
                print(f"  Failed to fit {name}: {e}")

        return results

    def _select_best_model(self, model_results: Dict) -> str:
        """Select best model based on information criteria"""
        aic_scores = {name: result.aic for name, result in model_results.items()}
        best_model = min(aic_scores, key=aic_scores.get)

        print(f"  Best model by AIC: {best_model} (AIC = {aic_scores[best_model]:.2f})")
        return best_model

    def _final_analysis(self, model: Callable, xdata: jnp.ndarray,
                       ydata: jnp.ndarray, yerr: Optional[jnp.ndarray]) -> FittingResults:
        """Perform final comprehensive analysis"""
        weights = 1.0 / yerr if yerr is not None else None

        return self.fitter.fit_with_uncertainty(
            model, xdata, ydata, weights=weights, bootstrap_samples=1000
        )

    def _calculate_quality_metrics(self, results: FittingResults) -> Dict:
        """Calculate additional quality metrics"""
        return {
            'parameter_precision': results.parameter_errors / jnp.abs(results.parameters) * 100,
            'residual_autocorrelation': self._residual_autocorrelation(results.residuals),
            'normality_test': self._test_residual_normality(results.residuals)
        }

    def _residual_autocorrelation(self, residuals: jnp.ndarray) -> float:
        """Test for residual autocorrelation (Durbin-Watson test)"""
        diff_residuals = jnp.diff(residuals)
        dw_statistic = jnp.sum(diff_residuals**2) / jnp.sum(residuals**2)
        return float(dw_statistic)

    def _test_residual_normality(self, residuals: jnp.ndarray) -> Dict:
        """Test residuals for normality"""
        # Simple normality tests
        mean_residual = jnp.mean(residuals)
        std_residual = jnp.std(residuals)
        skewness = jnp.mean(((residuals - mean_residual) / std_residual)**3)
        kurtosis = jnp.mean(((residuals - mean_residual) / std_residual)**4) - 3

        return {
            'mean': float(mean_residual),
            'std': float(std_residual),
            'skewness': float(skewness),
            'excess_kurtosis': float(kurtosis)
        }

    def _generate_report(self, results: FittingResults, outlier_analysis: Dict,
                        model_name: str) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
=== NLSQ Scientific Data Analysis Report ===

Model: {model_name}

Fitting Results:
  Parameters: {results.parameters}
  Uncertainties: ±{results.parameter_errors}

Statistical Quality:
  R² = {results.r_squared:.6f}
  Adjusted R² = {1 - (1 - results.r_squared) * (len(results.residuals) - 1) / results.degrees_of_freedom:.6f}
  AIC = {results.aic:.2f}
  BIC = {results.bic:.2f}
  χ² = {results.chi_squared:.2f} (DOF = {results.degrees_of_freedom})

Confidence Intervals (95%):
  Lower bounds: {results.confidence_intervals['lower']}
  Upper bounds: {results.confidence_intervals['upper']}

Data Quality:
  Outliers detected: {outlier_analysis['n_outliers']} ({outlier_analysis['outlier_percentage']:.1f}%)

Residual Analysis:
  Mean residual: {jnp.mean(results.residuals):.6f}
  Residual std: {jnp.std(results.residuals):.6f}

Recommendations:
  - Model fit quality: {'Excellent' if results.r_squared > 0.95 else 'Good' if results.r_squared > 0.90 else 'Fair' if results.r_squared > 0.80 else 'Poor'}
  - Parameter precision: {'High' if jnp.max(results.parameter_errors / jnp.abs(results.parameters)) < 0.05 else 'Moderate' if jnp.max(results.parameter_errors / jnp.abs(results.parameters)) < 0.20 else 'Low'}
  - Outlier impact: {'Minimal' if outlier_analysis['outlier_percentage'] < 5 else 'Moderate' if outlier_analysis['outlier_percentage'] < 15 else 'Significant'}
"""
        return report
```

## Integration with Scientific Ecosystem

### NumPy and SciPy Integration
```python
# Seamless integration with existing scientific Python ecosystem
def nlsq_to_scipy_comparison(model: Callable, xdata: np.ndarray, ydata: np.ndarray) -> Dict:
    """Compare NLSQ performance with SciPy curve_fit"""
    from scipy.optimize import curve_fit
    import time

    # Convert to JAX arrays
    xdata_jax = jnp.array(xdata)
    ydata_jax = jnp.array(ydata)

    # NLSQ fitting
    start_time = time.time()
    cf = CurveFit()
    popt_nlsq, pcov_nlsq = cf.curve_fit(model, xdata_jax, ydata_jax)
    nlsq_time = time.time() - start_time

    # SciPy fitting
    start_time = time.time()
    popt_scipy, pcov_scipy = curve_fit(model, xdata, ydata)
    scipy_time = time.time() - start_time

    return {
        'nlsq': {'params': popt_nlsq, 'cov': pcov_nlsq, 'time': nlsq_time},
        'scipy': {'params': popt_scipy, 'cov': pcov_scipy, 'time': scipy_time},
        'speedup': scipy_time / nlsq_time,
        'param_diff': jnp.abs(popt_nlsq - popt_scipy),
        'cov_diff': jnp.abs(pcov_nlsq - pcov_scipy)
    }

# Integration with pandas for data handling
def analyze_dataframe(df, x_column: str, y_column: str,
                     error_column: Optional[str] = None,
                     model_column: Optional[str] = None) -> Dict:
    """Analyze curve fitting for pandas DataFrame"""

    analyzer = ScientificDataAnalyzer()

    if model_column:
        # Group by model and analyze separately
        results = {}
        for model_type in df[model_column].unique():
            subset = df[df[model_column] == model_type]
            xdata = jnp.array(subset[x_column].values)
            ydata = jnp.array(subset[y_column].values)
            yerr = jnp.array(subset[error_column].values) if error_column else None

            results[model_type] = analyzer.analyze_experimental_data(xdata, ydata, yerr)

        return results
    else:
        # Single analysis
        xdata = jnp.array(df[x_column].values)
        ydata = jnp.array(df[y_column].values)
        yerr = jnp.array(df[error_column].values) if error_column else None

        return analyzer.analyze_experimental_data(xdata, ydata, yerr)
```

## Use Cases and Applications

### Experimental Physics
- **Spectroscopy**: Peak fitting for absorption, emission, and scattering spectra
- **Kinetics**: Reaction rate analysis and mechanism determination
- **Thermodynamics**: Phase transition characterization and critical point analysis
- **Optics**: Beam profile analysis and optical parameter extraction

### Biochemistry and Biophysics
- **Enzyme Kinetics**: Michaelis-Menten parameter estimation with confidence intervals
- **Protein Folding**: Multi-exponential decay analysis for folding kinetics
- **Binding Studies**: Equilibrium binding constant determination
- **Dose-Response**: Hill equation fitting for drug discovery

### Materials Science
- **Crystallography**: Unit cell parameter refinement and structure analysis
- **Mechanical Testing**: Stress-strain curve analysis and material property extraction
- **Thermal Analysis**: Glass transition and melting point determination
- **Surface Science**: Adsorption isotherm fitting and surface area calculation

### Environmental Science
- **Climate Data**: Temperature trend analysis and climate model validation
- **Pollution Monitoring**: Concentration decay modeling and source identification
- **Hydrology**: Flow rate analysis and watershed modeling
- **Atmospheric Science**: Trace gas concentration fitting and transport modeling

## Integration with Existing Agents

- **JAX Expert**: Advanced JAX optimization techniques and GPU acceleration
- **Statistics Expert**: Statistical validation and uncertainty quantification methods
- **Visualization Expert**: Publication-quality plotting of fitted curves and residual analysis
- **GPU Computing Expert**: Memory optimization and distributed computing strategies
- **Numerical Computing Expert**: Advanced optimization algorithms and mathematical methods
- **Experiment Manager**: Systematic parameter estimation studies and model comparison

This agent transforms traditional curve fitting from basic parameter estimation into comprehensive scientific analysis with robust uncertainty quantification, outlier detection, model comparison, and performance optimization for large-scale datasets.