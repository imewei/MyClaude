import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import importlib.util

# Helper to import source files from arbitrary paths
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Define paths to scripts
PLUGIN_ROOT = Path(__file__).parent.parent.parent / "plugins" / "science-suite"
DIAGNOSE_SCRIPT = PLUGIN_ROOT / "skills/nlsq-core-mastery/scripts/diagnose_optimization.py"
COMPARE_SCRIPT = PLUGIN_ROOT / "skills/training-diagnostics/scripts/compare_training_runs.py"

class TestDiagnoseOptimization(unittest.TestCase):
    def setUp(self):
        if not DIAGNOSE_SCRIPT.exists():
            self.skipTest(f"Script not found: {DIAGNOSE_SCRIPT}")
            
        # Mock jax and jax.numpy before importing
        self.mock_jax = MagicMock()
        self.mock_jnp = MagicMock()
        
        # Setup fake JNP behavior for norms and finite checks
        # We must set return_value on the specific nested attributes
        self.mock_jnp.linalg.norm.return_value = 1.0
        self.mock_jnp.linalg.cond.return_value = 10.0
        self.mock_jnp.all.return_value = True
        self.mock_jnp.max.return_value = 1.0
        self.mock_jnp.min.return_value = 0.0
        self.mock_jnp.abs.return_value = MagicMock() # abs returns a mock (array), max of that is float
        self.mock_jnp.isnan.return_value = False
        self.mock_jnp.isinf.return_value = False
        
        with patch.dict(sys.modules, {'jax': self.mock_jax, 'jax.numpy': self.mock_jnp}):
            self.module = import_from_path("diagnose_optimization", DIAGNOSE_SCRIPT)
            
        # FORCE override jnp in the module to ensure it uses our configured mock
        self.module.jnp = self.mock_jnp

        # Conversion to float is explicit in the script, so returning a float from norm is critical
        self.mock_jnp.linalg.norm.return_value = 1.0
        self.mock_jnp.linalg.cond.return_value = 10.0
        self.mock_jnp.all.return_value = True
        self.mock_jnp.max.return_value = 1.0
        self.mock_jnp.min.return_value = 0.0
        self.mock_jnp.mean.return_value = 0.0
        self.mock_jnp.std.return_value = 1.0
        self.mock_jnp.abs.return_value = MagicMock()
        self.mock_jnp.isnan.return_value = False
        self.mock_jnp.isnan.return_value = False
      
    def test_analyze_convergence_converged(self):
        # Setup specific mock for this test
        result = MagicMock()
        result.success = True
        result.message = "Optimization terminated successfully."
        result.nfev = 10
        result.cost_history = [10.0, 5.0, 1.0]
        result.initial_cost = 10.0
        result.cost = 1.0
        result.grad_norm = 0.1
        result.condition_number = 10.0
        # result.fun is residuals array
        result.fun = self.mock_jnp.array([0.0, 0.0]) # mock_jnp.array returns a mock but we need it to work with mean/std
        # Actually residuals can be a list or array. mean/std take it.
        # Since usage is jnp.mean(residuals), and we mocked jnp.mean to return 0.0, the value of residuals doesn't strictly matter for the mean call return, 
        # BUT result.fun is accessed.
        result.fun = MagicMock() 
        result.grad = MagicMock()
        result.jac = MagicMock()
        
        # Test private helper if accessible, or public diagnose_result
        # Since we are testing logic, calling diagnose_result is the entry point
        with patch('builtins.print') as mock_print:
            self.module.diagnose_result(result, verbose=True)
            # Verify "Success" message
            printed_text = "".join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn("Success: True", printed_text)

class TestCompareTrainingRuns(unittest.TestCase):
    def setUp(self):
        if not COMPARE_SCRIPT.exists():
            self.skipTest(f"Script not found: {COMPARE_SCRIPT}")
            
        self.module = import_from_path("compare_training_runs", COMPARE_SCRIPT)

    def test_print_comparison_report(self):
        configs = {
            "run1": {"lr": 0.001, "batch_size": 32},
            "run2": {"lr": 0.002, "batch_size": 32}
        }
        metrics = {
            "run1": {"val_loss": [0.5, 0.4, 0.3]},
            "run2": {"val_loss": [0.6, 0.5, 0.2]}
        }
        
        with patch('builtins.print') as mock_print:
            self.module.print_comparison_report(configs, metrics, primary_metric="val_loss")
            
            # Check for header
            self.assertTrue(any("COMPARISON REPORT" in str(c) for c in mock_print.call_args_list))
            # Check for config diff
            printed_text = " ".join([str(call.args[0]) for call in mock_print.call_args_list])
            self.assertIn("lr", printed_text)
            self.assertIn("0.001", printed_text)
            self.assertIn("0.002", printed_text)
