#!/bin/bash
# Fix Commit Errors Command Wrapper
# Executes the fix-commit-errors GitHub Actions error fixing tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/fix-commit-errors.py" "$@"