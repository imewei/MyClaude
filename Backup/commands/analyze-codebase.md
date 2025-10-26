---
description: Generate comprehensive analysis and documentation of entire codebase
allowed-tools: Bash(find:*), Bash(tree:*), Bash(grep:*), Bash(wc:*), Bash(du:*)
argument-hint: [output-file]
color: cyan
agents:
  primary:
    - systems-architect
    - code-quality
  conditional:
    - agent: research-intelligence
      trigger: files "*.tex|*.bib|*.md" OR dir "research/|papers/"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|matplotlib|scientific.*computing" OR files "*.ipynb"
    - agent: jax-scientific-domains
      trigger: pattern "jax_md|jax_cfd|cirq|pennylane|qiskit" OR files "*quantum*|*cfd*|*molecular*"
    - agent: simulation-expert
      trigger: pattern "lammps|gromacs|namd|md.*simulation" OR files "*simulation*|*.lmp|*.gro"
    - agent: correlation-function-expert
      trigger: pattern "correlation|scattering|saxs|sans|structure.*factor" OR files "*correlation*|*scattering*"
    - agent: visualization-interface
      trigger: pattern "plotly|d3|matplotlib|dashboard|viz" OR files "*viz*|*dashboard*|*plot*" OR dir "visualization/|dashboards/"
    - agent: fullstack-developer
      trigger: files "package.json|src/components/"
    - agent: data-engineering-coordinator
      trigger: pattern "spark|dask|ray|airflow|prefect|data.*pipeline" OR dir "data/|pipelines/"
    - agent: devops-security-engineer
      trigger: files "Dockerfile|.gitlab-ci.yml|.github/workflows/"
    - agent: ml-pipeline-coordinator
      trigger: pattern "sklearn|tensorflow|torch|mlflow|wandb" OR files "*.h5|*.pkl|models/"
    - agent: database-workflow-engineer
      trigger: pattern "airflow|prefect|dagster|postgresql|mysql" OR files "*.sql|dags/|workflows/"
  orchestrated: true
---

# Comprehensive Codebase Analysis

## Auto-Discovery

### Project Metrics
- Total files: !`find . -type f ! -path "*/node_modules/*" ! -path "*/.git/*" 2>/dev/null | wc -l`
- Code files: !`find . -name "*.{js,ts,py,java,go,rs,rb,php}" ! -path "*/node_modules/*" 2>/dev/null | wc -l`
- Size: !`du -sh . --exclude=node_modules --exclude=.git 2>/dev/null | cut -f1`
- Languages: !`find . -name "*.py" -o -name "*.js" -o -name "*.go" -o -name "*.rs" ! -path "*/node_modules/*" 2>/dev/null | sed 's/.*\.//' | sort -u | tr '\n' ',' | sed 's/,$//'`

### Tech Stack Detection
- Package manager: !`ls package.json requirements.txt Cargo.toml go.mod Gemfile composer.json 2>/dev/null | head -1`
- Build tools: !`ls webpack.config.js vite.config.js tsconfig.json Makefile 2>/dev/null | tr '\n' ',' | sed 's/,$//'`
- CI/CD: !`find .github .gitlab-ci.yml .circleci -type f 2>/dev/null | wc -l` configs

### Architecture Pattern Detection
- API routes: !`find . -path "*/routes/*" -o -path "*/api/*" ! -path "*/node_modules/*" 2>/dev/null | wc -l` files
- Database: !`find . -path "*/models/*" -o -path "*/migrations/*" ! -path "*/node_modules/*" 2>/dev/null | wc -l` files
- Frontend: !`find . -path "*/components/*" -o -path "*/views/*" ! -path "*/node_modules/*" 2>/dev/null | wc -l` files
- Tests: !`find . -name "*test*" -o -name "*spec*" ! -path "*/node_modules/*" 2>/dev/null | wc -l` files

### Key Configuration Files
@package.json
@requirements.txt
@Cargo.toml
@tsconfig.json
@README.md

### Entry Points (First 50 Lines)
!`find . \( -name "main.*" -o -name "index.*" -o -name "app.*" \) ! -path "*/node_modules/*" 2>/dev/null | head -3 | while read f; do echo "=== $f ==="; head -50 "$f" 2>/dev/null; done`

## Your Task

Create comprehensive analysis covering:

### 1. Project Overview
- Type: web app / API / library / CLI tool
- Primary language & frameworks
- Architecture: MVC / microservices / serverless / monolith
- Deployment target

### 2. Architecture Analysis
```
[Frontend] ──→ [API/Backend] ──→ [Database]
     │              │                  │
[Static Assets]  [Services]      [Cache/Queue]
```
- Component relationships
- Data flow patterns
- External integrations
- Design patterns used

### 3. Technology Stack
**Runtime**: Node.js / Python / Go / Rust / etc.
**Framework**: Express / FastAPI / Gin / Actix / etc.
**Database**: PostgreSQL / MongoDB / Redis / etc.
**Testing**: Jest / Pytest / etc.
**CI/CD**: GitHub Actions / GitLab CI / etc.

### 4. Directory Structure
```
src/
  ├─ api/         → REST endpoints
  ├─ models/      → Data schemas
  ├─ services/    → Business logic
  ├─ utils/       → Helpers
tests/            → Test suites
config/           → Configuration
```

### 5. Entry Points & Flow
- Main entry: `src/index.js:15`
- Request lifecycle: route → controller → service → model → database
- Key functions identified

### 6. Development Setup
```bash
# Install dependencies
npm install / pip install -r requirements.txt

# Run tests
npm test / pytest

# Start development
npm run dev
```

### 7. Code Quality Assessment
- **Strengths**: [identify good patterns]
- **Issues**: [complexity, duplication, security]
- **Recommendations**: [prioritized improvements]

### 8. Security & Performance
- Dependency vulnerabilities check
- Performance bottlenecks
- Scaling considerations
- Security best practices

**Output file**: Write analysis to `codebase_analysis.md` with all sections detailed