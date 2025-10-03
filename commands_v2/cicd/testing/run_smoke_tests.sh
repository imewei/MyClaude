#!/usr/bin/env bash
# Run quick smoke tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Running Smoke Tests"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run smoke tests (fast subset)
pytest tests/ \
    -v \
    -m "smoke" \
    --tb=short \
    --maxfail=3 \
    "$@"

echo "Smoke tests passed!"