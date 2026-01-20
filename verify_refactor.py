
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

# Mocking JAX and other dependencies to avoid needing full environment for structural verification
class FakeJNP:
    @staticmethod
    def max(*args, **kwargs): return np.max(*args, **kwargs)
    @staticmethod
    def min(*args, **kwargs): return np.min(*args, **kwargs)
    @staticmethod
    def mean(*args, **kwargs): return np.mean(*args, **kwargs)
    @staticmethod
    def std(*args, **kwargs): return np.std(*args, **kwargs)
    @staticmethod
    def abs(*args, **kwargs): return np.abs(*args, **kwargs)
    @staticmethod
    def any(*args, **kwargs): return np.any(*args, **kwargs)
    @staticmethod
    def where(*args, **kwargs): return np.where(*args, **kwargs)
    
    class linalg:
        @staticmethod
        def norm(*args, **kwargs): return np.linalg.norm(*args, **kwargs)
        @staticmethod
        def cond(*args, **kwargs): return np.linalg.cond(*args, **kwargs)

sys.modules['jax'] = MagicMock()
sys.modules['jax.numpy'] = FakeJNP()
sys.modules['jax.tree_util'] = MagicMock()

# Add plugins path to sys.path
sys.path.append('plugins/science-suite/skills/jax-mastery/nlsq-core-mastery/scripts')
sys.path.append('plugins/science-suite/skills/deep-learning/training-diagnostics/scripts')

from diagnose_optimization import diagnose_result
from compare_training_runs import print_comparison_report


class TestRefactoring(unittest.TestCase):
    def test_diagnose_result_structure(self):
        """Verify diagnose_result runs without error on mock object"""
        mock_result = MagicMock()
        mock_result.params = {'x': np.array([1.0, 2.0])}
        # Configure mock to behave like a result object with float values
        mock_result.success = True
        mock_result.message = "Converged"
        mock_result.nfev = 10
        mock_result.initial_cost = 1.0
        mock_result.cost = 0.1
        mock_result.grad = np.array([1e-5, 1e-5])
        mock_result.jac = np.eye(2)
        mock_result.x = np.array([1.0, 2.0])
        mock_result.fun = np.array([0.1, -0.1])
        
        # Mock active_mask for bound check coverage
        mock_result.active_mask = np.array([0, 0])

        print("\n--- Testing diagnose_result ---")
        try:
            diagnose_result(mock_result, verbose=True)
        except Exception as e:
             self.fail(f"diagnose_result raised exception: {e}")

    def test_print_comparison_report_structure(self):
        """Verify print_comparison_report runs without error"""
        run1 = {
            'config': {'lr': 0.01, 'batch_size': 32},
            'metrics': {'loss': [1.0, 0.5, 0.2], 'accuracy': [0.5, 0.8, 0.9]},
            'final_metrics': {'loss': 0.2, 'accuracy': 0.9}
        }
        run2 = {
            'config': {'lr': 0.001, 'batch_size': 32},
            'metrics': {'loss': [1.0, 0.8, 0.6], 'accuracy': [0.5, 0.6, 0.7]},
            'final_metrics': {'loss': 0.6, 'accuracy': 0.7}
        }
        
        configs = {
            "Run A": run1['config'],
            "Run B": run2['config']
        }
        metrics = {
            "Run A": run1['metrics'],
            "Run B": run2['metrics']
        }
        
        print("\n--- Testing print_comparison_report ---")
        try:
            print_comparison_report(configs, metrics, "loss")
        except Exception as e:
            self.fail(f"print_comparison_report raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
