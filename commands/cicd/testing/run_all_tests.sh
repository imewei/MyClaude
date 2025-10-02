#!/usr/bin/env bash
# Run complete test suite with coverage and reporting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running Complete Test Suite"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -e .[dev] > /dev/null 2>&1

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
rm -f .coverage coverage.xml
rm -rf htmlcov/

# Run unit tests
echo -e "\n${YELLOW}Running unit tests...${NC}"
pytest tests/unit/ \
    -v \
    --cov=claude_commands \
    --cov-report=term \
    --cov-report=html \
    --cov-report=xml \
    --junit-xml=junit-unit.xml \
    --tb=short \
    || { echo -e "${RED}Unit tests failed!${NC}"; exit 1; }

# Run integration tests
echo -e "\n${YELLOW}Running integration tests...${NC}"
pytest tests/integration/ \
    -v \
    --cov=claude_commands \
    --cov-append \
    --cov-report=term \
    --junit-xml=junit-integration.xml \
    --tb=short \
    || { echo -e "${RED}Integration tests failed!${NC}"; exit 1; }

# Run performance benchmarks
echo -e "\n${YELLOW}Running performance benchmarks...${NC}"
pytest tests/benchmarks/ \
    -v \
    --benchmark-only \
    --benchmark-json=benchmark-results.json \
    --tb=short \
    || { echo -e "${RED}Performance tests failed!${NC}"; exit 1; }

# Generate coverage report
echo -e "\n${YELLOW}Generating coverage report...${NC}"
coverage report --fail-under=90 || {
    echo -e "${RED}Coverage is below 90%!${NC}"
    exit 1
}

# Check coverage for critical modules
echo -e "\n${YELLOW}Checking critical module coverage...${NC}"
python "$SCRIPT_DIR/check_module_coverage.py" \
    --threshold 95 \
    --critical claude_commands/core \
    --critical claude_commands/executor

# Run doctests
echo -e "\n${YELLOW}Running doctests...${NC}"
pytest --doctest-modules claude_commands/ -v || {
    echo -e "${YELLOW}Warning: Some doctests failed${NC}"
}

# Display summary
echo -e "\n=========================================="
echo -e "${GREEN}All Tests Passed!${NC}"
echo -e "=========================================="
echo "Coverage report: file://$PROJECT_ROOT/htmlcov/index.html"
echo "Benchmark results: $PROJECT_ROOT/benchmark-results.json"
echo "JUnit reports:"
echo "  - Unit tests: junit-unit.xml"
echo "  - Integration tests: junit-integration.xml"
echo "=========================================="