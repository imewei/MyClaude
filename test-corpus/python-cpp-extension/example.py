
import numpy as np
import fast_math

# Create large array
arr = np.random.randn(1000000)

# Use C++ extension for fast computation
result = fast_math.compute_sum(arr)
print(f"Sum: {result}")
