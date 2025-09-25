# Claude Command: Commit

This command helps you create well-formatted commits with conventional commit messages and emoji, ensuring ALL changes are properly committed.

## Usage

To create a commit, just type:
```
/commit
```

Or with options:
```
/commit --all        # Stage and commit all changes (default behavior)
/commit --staged     # Only commit currently staged files
/commit --amend      # Amend the previous commit
/commit --no-verify  # Skip pre-commit hooks
```

## What This Command Does

### Step 1: Check Status
- Runs `git status --porcelain` to get ALL changed files (modified, new, deleted)
- Counts total changes to determine commit strategy

### Step 2: Stage Files
- **Default behavior**: If no files staged, runs `git add -A` to stage EVERYTHING
- **With --staged**: Only commits already staged files
- **Smart grouping**: For 10+ files, suggests logical groupings

### Step 3: Create Commit
- Analyzes changes with `git diff --cached --stat`
- Generates appropriate emoji and message
- Runs `git commit` with the message

### Step 4: Verify Completeness
- **CRITICAL**: Runs `git status --porcelain` again after commit
- If files remain unstaged:
  - Shows list of remaining files
  - Prompts to create additional commit(s)
  - Repeats until ALL changes are committed
- Confirms with `git log -1 --oneline`

### Performance Optimizations
- Uses `--porcelain` for faster parsing
- Batches file operations with `git add -A`
- Caches diff analysis to avoid redundant calls
- Uses `--stat` for quick change summaries

## Best Practices for Commits

- **Verify before committing**: Ensure code is linted, builds correctly, and documentation is updated
- **Atomic commits**: Each commit should contain related changes that serve a single purpose
- **Split large changes**: If changes touch multiple concerns, split them into separate commits
- **Conventional commit format**: Use the format `<type>: <description>` where type is one of:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation changes
  - `style`: Code style changes (formatting, etc)
  - `refactor`: Code changes that neither fix bugs nor add features
  - `perf`: Performance improvements
  - `test`: Adding or fixing tests
  - `chore`: Changes to the build process, tools, etc.
- **Present tense, imperative mood**: Write commit messages as commands (e.g., "add feature" not "added feature")
- **Concise first line**: Keep the first line under 72 characters
- **Emoji**: Each commit type is paired with an appropriate emoji:
  - ✨ `feat`: New feature
  - 🐛 `fix`: Bug fix
  - 📝 `docs`: Documentation
  - 💄 `style`: Formatting/style
  - ♻️ `refactor`: Code refactoring
  - ⚡️ `perf`: Performance improvements
  - ✅ `test`: Tests
  - 🔧 `chore`: Tooling, configuration
  - 🚀 `ci`: CI/CD improvements
  - 🗑️ `revert`: Reverting changes
  - 🧪 `test`: Add a failing test
  - 🚨 `fix`: Fix compiler/linter warnings
  - 🔒️ `fix`: Fix security issues
  - 👥 `chore`: Add or update contributors
  - 🚚 `refactor`: Move or rename resources
  - 🏗️ `refactor`: Make architectural changes
  - 🔀 `chore`: Merge branches
  - 📦️ `chore`: Add or update compiled files or packages
  - ➕ `chore`: Add a dependency
  - ➖ `chore`: Remove a dependency
  - 🌱 `chore`: Add or update seed files
  - 🧑‍💻 `chore`: Improve developer experience
  - 🧵 `feat`: Add or update code related to multithreading or concurrency
  - 🔍️ `feat`: Improve SEO
  - 🏷️ `feat`: Add or update types
  - 💬 `feat`: Add or update text and literals
  - 🌐 `feat`: Internationalization and localization
  - 👔 `feat`: Add or update business logic
  - 📱 `feat`: Work on responsive design
  - 🚸 `feat`: Improve user experience / usability
  - 🩹 `fix`: Simple fix for a non-critical issue
  - 🥅 `fix`: Catch errors
  - 👽️ `fix`: Update code due to external API changes
  - 🔥 `fix`: Remove code or files
  - 🎨 `style`: Improve structure/format of the code
  - 🚑️ `fix`: Critical hotfix
  - 🎉 `chore`: Begin a project
  - 🔖 `chore`: Release/Version tags
  - 🚧 `wip`: Work in progress
  - 💚 `fix`: Fix CI build
  - 📌 `chore`: Pin dependencies to specific versions
  - 👷 `ci`: Add or update CI build system
  - 📈 `feat`: Add or update analytics or tracking code
  - ✏️ `fix`: Fix typos
  - ⏪️ `revert`: Revert changes
  - 📄 `chore`: Add or update license
  - 💥 `feat`: Introduce breaking changes
  - 🍱 `assets`: Add or update assets
  - ♿️ `feat`: Improve accessibility
  - 💡 `docs`: Add or update comments in source code
  - 🗃️ `db`: Perform database related changes
  - 🔊 `feat`: Add or update logs
  - 🔇 `fix`: Remove logs
  - 🤡 `test`: Mock things
  - 🥚 `feat`: Add or update an easter egg
  - 🙈 `chore`: Add or update .gitignore file
  - 📸 `test`: Add or update snapshots
  - ⚗️ `experiment`: Perform experiments
  - 🚩 `feat`: Add, update, or remove feature flags
  - 💫 `ui`: Add or update animations and transitions
  - ⚰️ `refactor`: Remove dead code
  - 🦺 `feat`: Add or update code related to validation
  - ✈️ `feat`: Improve offline support

## Guidelines for Splitting Commits

When analyzing the diff, consider splitting commits based on these criteria:

1. **Different concerns**: Changes to unrelated parts of the codebase
2. **Different types of changes**: Mixing features, fixes, refactoring, etc.
3. **File patterns**: Changes to different types of files (e.g., source code vs documentation)
4. **Logical grouping**: Changes that would be easier to understand or review separately
5. **Size**: Very large changes that would be clearer if broken down

## Examples

Good commit messages:
- ✨ feat: add user authentication system
- 🐛 fix: resolve memory leak in rendering process
- 📝 docs: update API documentation with new endpoints
- ♻️ refactor: simplify error handling logic in parser
- 🚨 fix: resolve linter warnings in component files
- 🧑‍💻 chore: improve developer tooling setup process
- 👔 feat: implement business logic for transaction validation
- 🩹 fix: address minor styling inconsistency in header
- 🚑️ fix: patch critical security vulnerability in auth flow
- 🎨 style: reorganize component structure for better readability
- 🔥 fix: remove deprecated legacy code
- 🦺 feat: add input validation for user registration form
- 💚 fix: resolve failing CI pipeline tests
- 📈 feat: implement analytics tracking for user engagement
- 🔒️ fix: strengthen authentication password requirements
- ♿️ feat: improve form accessibility for screen readers

Example of splitting commits:
- First commit: ✨ feat: add new solc version type definitions
- Second commit: 📝 docs: update documentation for new solc versions
- Third commit: 🔧 chore: update package.json dependencies
- Fourth commit: 🏷️ feat: add type definitions for new API endpoints
- Fifth commit: 🧵 feat: improve concurrency handling in worker threads
- Sixth commit: 🚨 fix: resolve linting issues in new code
- Seventh commit: ✅ test: add unit tests for new solc version features
- Eighth commit: 🔒️ fix: update dependencies with security vulnerabilities

## Command Options

- `--all`: Stage and commit all changes (default behavior)
- `--staged`: Only commit currently staged files
- `--amend`: Amend the previous commit
- `--no-verify`: Skip pre-commit hooks
- `--push`: Automatically push after successful commit

## Important Notes

- **Complete commits**: The command ensures ALL changes are committed, not just staged files
- **Automatic staging**: If no files are staged, uses `git add -A` to stage everything (modified, new, and deleted files)
- **Post-commit verification**: After each commit, checks for remaining unstaged changes
- **Multiple commits**: For complex changes, helps create multiple logical commits
- **File grouping**: Intelligently groups related files together for atomic commits
- **Deletion handling**: Properly stages deleted files with `git add -A` or `git rm`
- **Status verification**: Always runs `git status` after commits to ensure nothing is left behind
- **Pre-commit hooks**: If the repository has pre-commit hooks, they will run automatically
- **Amended commits**: If pre-commit hooks modify files, the commit will be amended to include those changes

## Common Issues and Solutions

- **Files left uncommitted**: The command now checks for this and prompts for additional commits
- **Deleted files not staged**: Uses `git add -A` which properly stages deletions
- **Mixed changes**: Suggests splitting into multiple commits for better history
- **Large commits**: Automatically suggests breaking down into smaller, logical commits
- **Pre-commit hook changes**: Automatically detects and includes hook-modified files

## Workflow Example

```bash
# Typical workflow that ensures everything is committed:
/commit
# 1. Detects 15 changed files
# 2. Stages all with `git add -A`
# 3. Creates commit with appropriate message
# 4. Checks for remaining files
# 5. If found, prompts: "5 files remain uncommitted. Create another commit?"
# 6. Repeats until all changes are committed
```