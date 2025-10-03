#!/usr/bin/env bash
# Run production smoke tests

set -e

echo "Running Production Smoke Tests"

# Test basic import
python -c "import claude_commands; print(f'Version: {claude_commands.__version__}')"

# Test command execution
python -c "
from claude_commands.executor import CommandExecutor
executor = CommandExecutor()
result = executor.execute('echo \"Smoke test\"')
assert result.success, 'Command execution failed'
print('✓ Command execution works')
"

# Test basic commands
python -c "
from claude_commands import execute_command
result = execute_command('python --version')
assert result.returncode == 0, 'Python version check failed'
print('✓ Python version check works')
"

echo "All smoke tests passed!"