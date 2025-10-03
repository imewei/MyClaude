#!/bin/bash
# Think-Ultra Command Wrapper
# Executes the think-ultra analytical engine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/think-ultra.py" "$@"