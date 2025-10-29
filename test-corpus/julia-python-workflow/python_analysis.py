
import pandas as pd
import matplotlib.pyplot as plt
from julia import Main

# Call Julia functions from Python
Main.include("workflow.jl")

# Load results from Julia
results = Main.eval("julia_predictions")

# Visualize in Python
plt.figure(figsize=(10, 6))
plt.hist(results, bins=50)
plt.title("Prediction Distribution")
plt.savefig("results.png")
