#!/usr/bin/env bash
# Build documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Building documentation..."

# Install documentation dependencies
pip install -q mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions

# Clean previous build
rm -rf site/

# Build documentation
mkdocs build --strict --verbose

echo "Documentation built successfully!"
echo "Output: $PROJECT_ROOT/site/"