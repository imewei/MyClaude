#!/usr/bin/env bash
# Deploy package to PyPI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Deploying to PyPI"
echo "=========================================="

# Check for required environment variables
if [ -z "$PYPI_API_TOKEN" ]; then
    echo "ERROR: PYPI_API_TOKEN environment variable not set"
    exit 1
fi

# Ensure we have a clean build
echo "Building package..."
python cicd/build/build.py --clean --checksums

# Validate package
echo "Validating package..."
twine check dist/*

# Upload to PyPI
echo "Uploading to PyPI..."
twine upload dist/* \
    --username __token__ \
    --password "$PYPI_API_TOKEN" \
    --verbose

echo "=========================================="
echo "Deployment to PyPI completed successfully!"
echo "=========================================="