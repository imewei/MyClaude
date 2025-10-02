#!/bin/bash
# Double-Check Command Wrapper
# Executes the double-check verification engine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/double-check.py" "$@"