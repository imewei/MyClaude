#!/usr/bin/env bash
# Run performance benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Running Performance Benchmarks"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install benchmark dependencies
pip install pytest-benchmark memory-profiler py-spy > /dev/null 2>&1

# Run benchmarks
pytest tests/benchmarks/ \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=3 \
    --benchmark-min-rounds=5 \
    --benchmark-json=benchmark-results.json \
    --benchmark-compare=baseline.json \
    --benchmark-autosave \
    "$@"

echo "Performance benchmarks completed!"
echo "Results saved to: benchmark-results.json"