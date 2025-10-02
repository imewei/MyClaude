# Adding Validation Projects

Guide for adding new projects to the validation suite.

## Quick Start

1. Edit `validation/suite/validation_projects.yaml`
2. Add your project under appropriate size category
3. Test the addition
4. Commit changes

## Project Configuration

### Basic Structure

```yaml
validation_projects:
  small:  # or medium, large, enterprise
    - name: my_project            # Required: unique identifier
      repo: https://github.com/...  # Required: git repository URL
      language: python             # Required: primary language
      loc: ~3000                   # Required: approximate lines of code
      domain: web_framework        # Required: project domain
      validation_priority: high    # Required: critical/high/medium/low
      key_features:                # Optional: notable features
        - Feature 1
        - Feature 2
```

### Required Fields

#### name
- Unique identifier for the project
- Use lowercase with underscores
- Example: `my_awesome_project`

#### repo
- Full HTTPS URL to git repository
- Must be publicly accessible
- Example: `https://github.com/user/repo`

#### language
- Primary programming language
- Supported: `python`, `javascript`, `typescript`, `java`, `go`, `rust`, `julia`
- Example: `python`

#### loc
- Approximate lines of code
- Use tilde (~) prefix for estimates
- Example: `~3000`

#### domain
- Project domain/category
- Examples: `web_framework`, `cli_tool`, `data_analysis`, `ml`, `testing`

#### validation_priority
- Importance for validation
- Values: `critical`, `high`, `medium`, `low`
- Critical = always run, low = optional

### Optional Fields

#### key_features
- List of notable features
- Helps understand project complexity
- Example:
  ```yaml
  key_features:
    - REST API
    - Async support
    - Plugin system
  ```

#### branch
- Specific branch to clone
- Default: main/master
- Example: `branch: develop`

#### tags
- Custom tags for filtering
- Example:
  ```yaml
  tags:
    - async
    - web
    - api
  ```

## Size Categories

### Small (1K-5K LOC)
- Simple, focused projects
- Quick validation (<10 minutes)
- Good for smoke testing
- Examples: CLI tools, libraries

### Medium (10K-50K LOC)
- Moderate complexity
- Medium validation time (20-30 minutes)
- Representative of typical projects
- Examples: Web frameworks, ORMs

### Large (50K-200K LOC)
- Complex, mature projects
- Long validation time (45-90 minutes)
- Comprehensive testing
- Examples: Full-stack frameworks, data libraries

### Enterprise (200K+ LOC)
- Very large codebases
- Extended validation (2-4 hours)
- Real-world complexity
- Examples: Workflow systems, platforms

## Domain Categories

Choose appropriate domain:

- `web_framework` - Web application frameworks
- `cli_tool` - Command-line tools
- `data_analysis` - Data processing and analysis
- `ml` - Machine learning and AI
- `scientific` - Scientific computing
- `testing` - Testing frameworks
- `database` - Database tools and ORMs
- `networking` - Network libraries
- `security` - Security tools
- `devops` - DevOps and automation

## Validation Priority

### Critical
- Core validation projects
- Always run in CI/CD
- Examples: FastAPI, Django, pytest

### High
- Important but not critical
- Run in nightly builds
- Examples: Flask, requests

### Medium
- Nice to have
- Run weekly
- Examples: Specialized tools

### Low
- Optional validation
- Run on demand
- Examples: Experimental projects

## Adding a Project

### Step 1: Research

Gather information:
- Repository URL
- Approximate LOC (use `cloc` or similar)
- Primary language
- Project domain
- Key features

### Step 2: Edit Configuration

Add to `validation/suite/validation_projects.yaml`:

```yaml
validation_projects:
  medium:
    - name: awesome_framework
      repo: https://github.com/awesome/framework
      language: python
      loc: ~15000
      domain: web_framework
      validation_priority: high
      key_features:
        - Async support
        - Built-in ORM
        - Auto-documentation
```

### Step 3: Test Addition

Run validation on new project:

```bash
python validation/executor.py --projects awesome_framework
```

### Step 4: Verify Results

Check:
- Clone successful
- Scenarios run
- Metrics collected
- Reports generated

### Step 5: Document

Add to project list in README if notable.

## Testing New Projects

### Minimal Test

```bash
# Test clone and basic validation
python validation/executor.py \
    --projects your_project \
    --scenarios code_quality_improvement \
    --parallel 1
```

### Full Test

```bash
# Test all scenarios
python validation/executor.py \
    --projects your_project
```

### Verify Results

```bash
# Check logs
tail -f validation/logs/validation_*.log

# Check reports
ls validation/reports/

# Check metrics
python -c "
from validation.executor import ValidationExecutor
executor = ValidationExecutor()
# ... check results
"
```

## Multi-Language Projects

For projects with multiple languages:

```yaml
- name: jupyterlab
  repo: https://github.com/jupyterlab/jupyterlab
  languages: [python, typescript, javascript]  # Note: plural
  loc: ~80000
  domain: interactive_computing
  validation_priority: high
  primary_language: python  # For language-specific tools
```

## Special Cases

### Private Repositories

Not currently supported. Must be public or accessible without authentication.

### Large Projects with Long Clone

Adjust timeout in config:

```yaml
validation_config:
  clone_timeout_seconds: 600  # 10 minutes
```

### Projects Requiring Build

Add build step to scenario:

```yaml
my_custom_scenario:
  steps:
    - action: build
      command: npm install && npm build
    - action: validate
      command: check-code-quality
```

### Projects with Non-Standard Structure

May need custom scenarios or skip certain validations.

## Best Practices

### DO

✅ Add projects actively maintained
✅ Verify repository is public
✅ Test before adding
✅ Use appropriate size category
✅ Set realistic priority

### DON'T

❌ Add abandoned projects
❌ Add private repositories
❌ Add without testing
❌ Over-estimate priority
❌ Add without LOC estimate

## Common Issues

### Clone Fails

**Problem**: Can't clone repository

**Solutions**:
- Verify URL is correct
- Check repository is public
- Increase timeout
- Check network connection

### Validation Fails

**Problem**: All scenarios fail

**Solutions**:
- Check project structure is standard
- Verify language is supported
- Review logs for specific errors
- May need custom scenarios

### Takes Too Long

**Problem**: Validation exceeds expected time

**Solutions**:
- Reduce parallel jobs
- Move to larger size category
- Adjust timeout settings
- Consider if project is too large

### Metrics Not Collected

**Problem**: Some metrics missing

**Solutions**:
- Verify required tools installed
- Check project has tests
- Review project structure
- May be expected for some projects

## Project Selection Criteria

Good validation projects:

✅ Actively maintained
✅ Good test coverage
✅ Standard structure
✅ Public repository
✅ Permissive license
✅ Representative of real-world code

Avoid:

❌ Abandoned/unmaintained
❌ No tests
❌ Non-standard structure
❌ Private/restricted
❌ Unusual build requirements

## Examples

### Example 1: Adding a CLI Tool

```yaml
cli_tools:
  - name: my_cli_tool
    repo: https://github.com/user/my-cli
    language: python
    loc: ~2500
    domain: cli_tool
    validation_priority: medium
    key_features:
      - Click-based CLI
      - Config file support
      - Plugin system
```

### Example 2: Adding a Web Framework

```yaml
medium:
  - name: awesome_web
    repo: https://github.com/awesome/web
    language: python
    loc: ~25000
    domain: web_framework
    validation_priority: high
    key_features:
      - ASGI support
      - OpenAPI generation
      - Dependency injection
      - Built-in validation
```

### Example 3: Adding a Data Library

```yaml
large:
  - name: data_processor
    repo: https://github.com/data/processor
    language: python
    loc: ~80000
    domain: data_analysis
    validation_priority: critical
    key_features:
      - DataFrame operations
      - SQL integration
      - Parallel processing
      - GPU support
```

## Maintenance

### Regular Review

- Quarterly review of all projects
- Remove if no longer maintained
- Update LOC estimates
- Adjust priorities as needed

### Monitoring

Watch for:
- Clone failures (repo moved/deleted)
- Consistent validation failures
- Excessive time increases
- Deprecated projects

### Deprecating Projects

If removing a project:

1. Mark as `validation_priority: low`
2. Monitor for 1 month
3. Remove from config
4. Document in changelog

## Support

Questions? Check:
- Main README: `validation/README.md`
- Validation Guide: `validation/VALIDATION_GUIDE.md`
- Logs: `validation/logs/`