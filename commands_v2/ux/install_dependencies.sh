#!/bin/bash

# Installation script for UX Enhancement System dependencies

echo "========================================"
echo "UX Enhancement System - Dependency Setup"
echo "========================================"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

echo "Installing required dependencies..."
echo ""

# Install required packages
echo "1. Installing rich (beautiful terminal output)..."
pip install -q "rich>=13.0.0"

echo "2. Installing click (enhanced CLI)..."
pip install -q "click>=8.0.0"

echo "3. Installing psutil (system monitoring)..."
pip install -q "psutil>=5.9.0"

echo ""
echo "Required dependencies installed successfully!"
echo ""

# Optional dependencies
read -p "Install optional dependencies? (prompt_toolkit, scikit-learn) [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing optional dependencies..."
    echo ""

    echo "1. Installing prompt_toolkit (interactive features)..."
    pip install -q "prompt-toolkit>=3.0.0"

    echo "2. Installing scikit-learn (ML recommendations)..."
    pip install -q "scikit-learn>=1.0.0"

    echo ""
    echo "Optional dependencies installed successfully!"
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "You can now run the examples:"
echo "  python ux/examples/progress_example.py"
echo "  python ux/examples/error_example.py"
echo "  python ux/examples/recommendation_example.py"
echo "  python ux/examples/integration_example.py"
echo ""
echo "See README.md for usage documentation."
echo ""