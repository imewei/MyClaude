#!/usr/bin/env bash
# Run integration tests only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Running Integration Tests"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run integration tests with various markers
pytest tests/integration/ \
    -v \
    --cov=claude_commands \
    --cov-report=term \
    --junit-xml=junit-integration.xml \
    -m "not slow" \
    "$@"

echo "Integration tests completed successfully!"