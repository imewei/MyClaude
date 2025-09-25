---
description: Comprehensive GitHub issue analysis and resolution for Python and Julia projects
category: github-workflow
argument-hint: [issue-number-or-url] [--draft] [--branch=<name>]
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, TodoWrite
---

# Advanced GitHub Issue Resolver

Automatically analyze, fix, and resolve GitHub issues with comprehensive workflow management for Python and Julia scientific computing projects.

## Usage

```bash
# Fix specific issue by number
/fix-github-issue 42

# Fix issue by URL
/fix-github-issue https://github.com/user/repo/issues/42

# Create draft PR for complex issues
/fix-github-issue 42 --draft

# Use custom branch name
/fix-github-issue 42 --branch=fix/memory-leak-analysis
```

## Comprehensive Issue Resolution Workflow

### 1. **Issue Analysis & Understanding**
- Fetch issue details using `gh issue view`
- Parse issue type: bug report, feature request, enhancement, documentation
- Extract key information:
  - Problem description and context
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details (Python/Julia versions, dependencies)
  - Labels, assignees, and project context
- Analyze linked discussions, related issues, and PR references
- Identify affected components from issue description

### 2. **Codebase Investigation & Root Cause Analysis**
- **Smart Search Strategy**:
  - Use keywords from issue to locate relevant files
  - Search for error messages, function names, class names
  - Identify related modules and dependencies
  - Check recent commits that might have introduced the issue
- **Python-Specific Analysis**:
  - Trace through import chains and module dependencies
  - Check for test failures related to the issue
  - Analyze stack traces and error contexts
  - Review package configuration (pyproject.toml, setup.py)
- **Julia-Specific Analysis**:
  - Check Project.toml and Manifest.toml for dependency issues
  - Analyze method dispatch and type stability problems
  - Review package documentation and exports
  - Examine performance profiling if relevant

### 3. **Solution Design & Planning**
- Create structured plan using TodoWrite for complex issues
- Break down solution into logical steps
- Identify potential risks and breaking changes
- Plan for backwards compatibility when needed
- Design test strategy to verify fix and prevent regression
- Consider performance implications and optimizations

### 4. **Implementation Strategy**

#### **Bug Fixes**
- Implement minimal, targeted fixes
- Add comprehensive error handling
- Include input validation where appropriate
- Ensure numerical stability for scientific computing
- Add logging/debugging information if helpful

#### **Feature Implementations**
- Follow existing code patterns and architecture
- Implement core functionality with proper abstraction
- Add configuration options for flexibility
- Include examples and usage documentation
- Design for extensibility and maintainability

#### **Performance Improvements**
- Benchmark current performance to establish baseline
- Implement optimizations (vectorization, JIT compilation, caching)
- Verify improvements with performance tests
- Document performance characteristics

### 5. **Comprehensive Testing Strategy**

#### **Python Testing**
- Write unit tests with pytest
- Add integration tests for complex workflows  
- Include edge cases and boundary conditions
- Test error handling and exception scenarios
- Add performance benchmarks when relevant
- Test across different Python versions if needed

#### **Julia Testing**
- Write comprehensive test suite with Pkg.test
- Include type stability tests with @code_warntype
- Add performance benchmarks and allocation tests
- Test method dispatch for different input types
- Verify numerical accuracy and stability

#### **Cross-Platform Testing**
- Test on different operating systems when relevant
- Verify compatibility with different dependency versions
- Test with different hardware configurations (CPU/GPU)

### 6. **Documentation & Communication**

#### **Code Documentation**
- Add comprehensive docstrings (NumPy/Google style for Python)
- Include usage examples and parameter descriptions
- Document any breaking changes or migration notes
- Update API documentation automatically

#### **User Documentation**  
- Update README if user-facing changes
- Add or update examples and tutorials
- Update changelog with clear descriptions
- Create migration guides for breaking changes

### 7. **Quality Assurance & Code Review**

#### **Code Quality Checks**
- **Python**: Run ruff, black, mypy, bandit
- **Julia**: Run formatter, check package quality
- **Cross-language**: Run pre-commit hooks, documentation builds
- Ensure all tests pass and coverage is maintained
- Verify no new security vulnerabilities introduced

#### **Review Preparation**
- Create detailed PR description linking to original issue
- Include before/after comparisons for bug fixes
- Add screenshots/examples for UI changes
- Document testing performed and results
- Highlight any potential risks or considerations

### 8. **Pull Request Creation & Management**

#### **Automated PR Creation**
- Create feature branch with descriptive name
- Generate comprehensive PR title and description
- Link to original issue with "Fixes #X" or "Closes #X"
- Add appropriate labels based on change type
- Request review from relevant maintainers

#### **PR Description Template**
```markdown
## Summary
Brief description of changes made

## Fixes
Closes #[issue-number]

## Changes Made
- [ ] Core implementation details
- [ ] Testing additions
- [ ] Documentation updates
- [ ] Configuration changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks (if applicable)
- [ ] Manual testing performed

## Breaking Changes
- None / List any breaking changes

## Migration Guide
- Not applicable / Include migration steps if needed
```

## Advanced Features

### **Multi-Issue Resolution**
Handle complex issues that span multiple components:
- Create separate branches for each component
- Coordinate fixes across related issues  
- Manage dependencies between changes
- Create linked PRs with proper sequencing

### **Regression Prevention**
- Add regression tests for fixed bugs
- Update CI/CD pipelines with new test scenarios
- Create monitoring for performance regressions
- Document known failure modes

### **Security Issue Handling**
- Handle security issues with appropriate discretion  
- Follow responsible disclosure practices
- Coordinate with maintainers for security patches
- Ensure security fixes are properly tested

### **Performance Issue Resolution**
- Profile current performance to identify bottlenecks
- Implement targeted optimizations
- Benchmark improvements with statistical significance
- Document performance characteristics
- Add performance monitoring for future regression detection

## Python-Specific Issue Types

### **Common Bug Patterns**
- Import errors and module path issues
- Dependency version conflicts  
- Type errors and annotation issues
- Numerical precision and stability problems
- Memory leaks and resource management
- Async/await and concurrency issues

### **Scientific Computing Issues**
- Algorithm correctness and numerical accuracy
- Performance bottlenecks in computational loops
- Memory usage with large datasets
- Visualization and plotting problems
- Statistical analysis edge cases
- GPU/CUDA compatibility issues

### **Package Issues**
- Setup and installation problems
- Configuration and environment issues
- Documentation build failures
- CI/CD pipeline problems
- Release and versioning issues

## Julia-Specific Issue Types

### **Performance Issues**
- Type instability and inference problems
- Memory allocation optimization
- Broadcasting and vectorization improvements
- Multiple dispatch optimization
- Package loading and compilation times

### **Package Ecosystem Issues**
- Dependency compatibility problems
- Project.toml and Manifest.toml issues
- Package registration and versioning
- Documentation generation with Documenter.jl
- Test suite organization and execution

### **Language Features**
- Method dispatch ambiguities
- Type system and abstract type hierarchies
- Macro expansion and metaprogramming
- Interoperability with other languages
- Package development best practices

## Command Options

- `--draft`: Create draft PR for review before final submission
- `--branch=<name>`: Use custom branch name instead of auto-generated
- `--no-tests`: Skip running full test suite (development mode)
- `--security`: Handle as security issue with appropriate protocols
- `--breaking`: Flag as potentially breaking change
- `--performance`: Focus on performance optimization approach

## Workflow Integration

### **Project Detection**
- Automatically detects Python vs Julia projects
- Identifies testing frameworks and CI systems
- Recognizes documentation systems (Sphinx, Documenter.jl)
- Adapts workflow to project conventions and standards

### **Team Collaboration**
- Respects project contribution guidelines
- Follows existing code style and patterns
- Integrates with project management tools
- Communicates effectively with maintainers

### **Continuous Integration**
- Ensures all CI checks pass before creating PR
- Handles flaky tests and environment issues
- Coordinates with existing automation
- Provides clear feedback on test results

## Success Metrics

### **Quality Indicators**
- ✅ Issue completely resolved
- ✅ All tests pass (existing + new)
- ✅ Code quality checks pass
- ✅ Documentation updated
- ✅ No performance regressions
- ✅ Backwards compatibility maintained
- ✅ Security considerations addressed

### **Process Efficiency**
- ⏱️ Time from issue assignment to PR creation
- 🎯 First-time fix success rate
- 📋 Completeness of issue resolution
- 🔄 Number of review cycles needed
- ✨ Code quality improvement

Target: Resolve GitHub issue $ARGUMENTS with comprehensive analysis and professional implementation