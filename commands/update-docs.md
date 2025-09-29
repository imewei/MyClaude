---
title: "Update Docs"
description: "Documentation generation tool with AST-based content extraction and multi-format compilation"
category: documentation
subcategory: documentation-generation
complexity: intermediate
argument-hint: "[--type=readme|api|research|all] [--format=markdown|html|latex] [--interactive] [--collaborative] [--publish] [--optimize] [--agents=auto|documentation|scientific|ai|engineering|research|all] [--orchestrate] [--parallel] [--intelligent]"
allowed-tools: Bash, Read, Write, Glob, MultiEdit, TodoWrite, WebSearch, WebFetch
model: inherit
tags: documentation, generation, ast-extraction, multi-format, technical-writing
dependencies: []
related: [explain-code, check-code-quality, generate-tests, commit, reflection]
workflows: [documentation-generation, technical-writing, publication-workflow]
version: "2.0"
last-updated: "2025-09-28"
---

# Update Docs

Documentation generation tool with AST-based content extraction, multi-format compilation, and version-controlled editing.

```bash
/update-docs [options]

# Basic usage
/update-docs
/update-docs --type=readme --format=markdown
/update-docs --type=research --publish --optimize
```

## Options

- `--type=<type>`: Documentation type (readme, api, research, all)
- `--format=<format>`: Output format (markdown, html, latex)
- `--interactive`: Enable interactive documentation features
- `--collaborative`: Enable collaborative editing features
- `--publish`: Generate publication-ready materials
- `--optimize`: Apply quality optimization and validation
- `--agents=<agents>`: Agent selection (auto, documentation, scientific, ai, engineering, research, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with workflow coordination
- `--parallel`: Run agents in parallel for maximum efficiency
- `--intelligent`: Enable intelligent agent selection based on content analysis

## Documentation Types

### README Documentation
- Project overview and description
- Installation and setup instructions
- Usage examples and quick start guides
- Contributing guidelines and license information
- Badges and project metadata

### API Documentation
- Function and class documentation
- Parameter descriptions and return values
- Code examples and usage patterns
- Type annotations and signatures
- Cross-references and links

### Research Documentation
- Academic paper templates
- Methodology and experimental setup
- Results presentation and analysis
- Bibliography and citation management
- Reproducibility documentation

### All Documentation
- Complete documentation suite
- Consistent formatting and style
- Cross-document linking
- Comprehensive coverage validation
- Integrated publishing workflow

## Output Formats

### Markdown
- Standard GitHub-flavored markdown
- Compatible with static site generators
- Supports code blocks and tables
- Inline math with LaTeX notation
- Interactive elements for web display

### HTML
- Web documentation with HTML output
- Navigation and search functionality
- Syntax highlighting for code blocks
- Mathematical equation rendering
- Mobile-compatible layout

### LaTeX
- Academic publication formatting
- Typesetting with LaTeX
- Mathematical notation support
- Bibliography integration
- Conference and journal templates

## 23-Agent Intelligent Documentation System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes content type, complexity, and requirements to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Content Type Detection → Agent Selection
- Python/Julia/JAX code → scientific-computing-master + jax-pro
- ML/AI projects → ai-systems-architect + neural-networks-master
- Research papers → research-intelligence-master + documentation-architect
- API documentation → systems-architect + fullstack-developer
- Quantum computing → advanced-quantum-computing-expert + scientific-computing-master
```

### Core Documentation Agents

#### **`documentation-architect`** - Master Documentation Expert
- **Technical Writing**: Expert documentation creation with promotional language filtering
- **Educational Content**: Progressive learning pathway design and tutorial engineering
- **Knowledge Management**: Information architecture and cross-document synthesis
- **API Documentation**: OpenAPI/Swagger and developer experience optimization
- **Research Documentation**: Academic paper generation and citation management

#### **`research-intelligence-master`** - Research Documentation Specialist
- **Academic Writing**: Research methodology and publication-ready documentation
- **Knowledge Synthesis**: Cross-domain research integration and analysis
- **Innovation Documentation**: Breakthrough discovery documentation and dissemination
- **Reproducibility**: Research reproducibility frameworks and validation documentation
- **Citation Management**: Advanced bibliography and reference systems

#### **`code-quality-master`** - Documentation Quality & Testing
- **Documentation Quality**: Accuracy verification and readability optimization
- **Accessibility Compliance**: WCAG 2.1 AA validation and universal design
- **Content Testing**: Automated documentation validation and quality gates
- **Performance Optimization**: Loading speed and cross-platform compatibility
- **User Experience Testing**: Navigation design and information discovery optimization

### Specialized Documentation Agents

#### **Scientific Computing Documentation**
- **`scientific-computing-master`**: Numerical computing and scientific workflow documentation
- **`jax-pro`**: JAX ecosystem documentation with GPU acceleration guides
- **`neural-networks-master`**: Deep learning model documentation and architecture guides
- **`advanced-quantum-computing-expert`**: Quantum computing documentation and hybrid systems

#### **Engineering & Architecture Documentation**
- **`systems-architect`**: System design documentation and architecture decision records
- **`fullstack-developer`**: Full-stack application documentation and user guides
- **`devops-security-engineer`**: Infrastructure documentation and security guides
- **`ai-systems-architect`**: AI system documentation and scalability guides

#### **Domain-Specific Documentation Experts**
- **`data-professional`**: Data pipeline documentation and analytics guides
- **`database-workflow-engineer`**: Database schema and query optimization documentation
- **`visualization-interface-master`**: UI/UX documentation and design system guides
- **`command-systems-engineer`**: Command system documentation and workflow guides

#### **Scientific Domain Documentation**
- **`correlation-function-expert`**: Statistical analysis documentation and methodology
- **`neutron-soft-matter-expert`**: Neutron scattering experiment documentation
- **`xray-soft-matter-expert`**: X-ray analysis workflow documentation
- **`nonequilibrium-stochastic-expert`**: Stochastic process documentation and theory
- **`scientific-code-adoptor`**: Legacy code modernization documentation

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection
Automatically analyzes codebase characteristics and selects optimal agent combinations:
- **Content Analysis**: Detects languages, frameworks, domain patterns
- **Complexity Assessment**: Evaluates documentation requirements and scope
- **Agent Matching**: Maps detected patterns to relevant agent expertise
- **Efficiency Optimization**: Balances comprehensive coverage with execution speed

#### **`documentation`** - Core Documentation Team
- `documentation-architect` (lead)
- `code-quality-master` (quality)
- `systems-architect` (technical architecture)
- `research-intelligence-master` (research content)

#### **`scientific`** - Scientific Computing Documentation
- `scientific-computing-master` (lead)
- `research-intelligence-master` (research methodology)
- `jax-pro` (JAX ecosystem)
- `neural-networks-master` (ML documentation)
- Domain-specific experts based on content detection

#### **`ai`** - AI/ML Documentation Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (deep learning)
- `data-professional` (data pipelines)
- `jax-pro` (scientific ML)
- `visualization-interface-master` (ML visualization)

#### **`engineering`** - Software Engineering Documentation
- `systems-architect` (lead)
- `fullstack-developer` (application docs)
- `devops-security-engineer` (infrastructure)
- `code-quality-master` (quality standards)
- `database-workflow-engineer` (data systems)

#### **`research`** - Research-Grade Documentation
- `research-intelligence-master` (lead)
- `documentation-architect` (academic writing)
- `scientific-computing-master` (computational methods)
- Domain-specific scientific experts
- `advanced-quantum-computing-expert` (quantum computing research)

#### **`all`** - Complete 23-Agent Documentation Ecosystem
Activates all relevant agents with intelligent orchestration for comprehensive documentation coverage.

### 23-Agent Orchestration (`--orchestrate`)

#### **Multi-Agent Documentation Pipeline**
1. **Content Analysis Phase**: Multiple agents analyze different aspects simultaneously
2. **Parallel Documentation Generation**: Domain experts work on their specializations
3. **Cross-Agent Synthesis**: `multi-agent-orchestrator` coordinates integration
4. **Quality Assurance**: Multi-agent validation and optimization
5. **Publication Preparation**: Research-grade formatting and citation management

#### **Intelligent Resource Management**
- **Load Balancing**: Optimal distribution of documentation tasks across agents
- **Dependency Coordination**: Sequential agent execution for dependent tasks
- **Conflict Resolution**: Automatic resolution of conflicting documentation approaches
- **Performance Monitoring**: Real-time tracking of documentation generation efficiency

## Features

### Content Generation
- AST parsing for docstring and comment extraction (supports Python, JavaScript, Rust, Go)
- Technical template rendering with promotional language filtering
- Dependency graph analysis for logical content ordering
- Symbol resolution via Language Server Protocol (LSP) integration
- Static analysis for metadata extraction (functions, classes, modules, exports)
- Language neutralization system for technical communication

### Quality Metrics & Validation
- Grammar checking via LanguageTool API integration
- Dead link detection with HTTP status verification (<5s timeout)
- Code coverage analysis: AST node documentation percentage (target: >90%)
- Style consistency via configurable linting rules (markdownlint, textlint)
- Technical accuracy scoring: factual correctness and implementation completeness
- Promotional language detection and removal

### Interactive Elements
- Code execution via Pyodide WebAssembly runtime (Python)
- Plot generation using Observable Plot / D3.js integration
- Jupyter notebook embedding with kernel execution support
- Real-time collaboration via WebSocket connections (Socket.IO)
- Content updates with file system watching

### Collaborative Features
- Multi-user editing support
- Version control integration
- Review and comment systems
- Change tracking and history
- Conflict resolution

### Publishing Support
- Research paper generation
- Citation management
- Bibliography formatting
- Publication templates
- Submission preparation

## Advanced 23-Agent Documentation Examples

```bash
# Intelligent auto-selection with content analysis
/update-docs --agents=auto --intelligent --optimize

# Scientific computing documentation with specialized agents
/update-docs --type=research --agents=scientific --format=latex --publish --orchestrate

# AI/ML project documentation with specialized team
/update-docs --type=api --agents=ai --format=html --optimize --parallel

# Complete 23-agent documentation ecosystem
/update-docs --type=all --agents=all --orchestrate --parallel --intelligent

# Engineering documentation with architecture focus
/update-docs --type=api --agents=engineering --format=html --optimize --collaborative

# Research-grade documentation with academic publishing
/update-docs --type=research --agents=research --format=latex --publish --orchestrate

# Quantum computing documentation with domain experts
/update-docs quantum_algorithm/ --agents=scientific --intelligent --optimize

# Legacy code documentation with modernization experts
/update-docs legacy_codebase/ --agents=auto --intelligent --collaborative

# Cross-domain documentation with full agent orchestra
/update-docs complex_project/ --agents=all --orchestrate --parallel --publish

# JAX/Scientific ML documentation
/update-docs jax_model.py --agents=scientific --intelligent --format=markdown

# Full-stack application documentation
/update-docs webapp/ --agents=engineering --format=html --interactive --optimize

# Research publication with breakthrough discovery documentation
/update-docs research_breakthrough/ --agents=research --format=latex --publish --orchestrate
```

### Agent Selection Examples by Content Type

```bash
# Content Type Detection → Intelligent Agent Selection

# Python scientific computing project
/update-docs simulation.py --agents=auto --intelligent
# → Selects: scientific-computing-master + jax-pro + research-intelligence-master

# Machine learning pipeline
/update-docs ml_pipeline/ --agents=auto --intelligent
# → Selects: ai-systems-architect + neural-networks-master + data-professional

# Quantum computing research
/update-docs quantum_circuit.py --agents=auto --intelligent
# → Selects: advanced-quantum-computing-expert + scientific-computing-master

# Web application with database
/update-docs webapp/ --agents=auto --intelligent
# → Selects: fullstack-developer + database-workflow-engineer + systems-architect

# Research paper with experimental data
/update-docs research_data/ --agents=auto --intelligent
# → Selects: research-intelligence-master + correlation-function-expert + documentation-architect

# Legacy Fortran modernization
/update-docs legacy_fortran/ --agents=auto --intelligent
# → Selects: scientific-code-adoptor + scientific-computing-master + documentation-architect
```

## Content Discovery

### Automatic Detection
- Source code analysis and extraction
- Configuration file processing
- Dependency identification
- Project structure analysis
- Metadata collection

### Template Matching
- Project type identification
- Framework-specific templates
- Language-specific formatting
- Domain-specific structures
- Custom template support

### Content Organization
- Logical section ordering
- Hierarchical structure creation
- Cross-reference mapping
- Navigation generation
- Index and table of contents

## Quality Assurance

### Validation Implementation
- Syntax validation for embedded code blocks (per language parser)
- HTTP link verification with 200/300 status code validation
- Example code execution testing in isolated containers
- Schema validation for document structure (JSON Schema)
- Coverage metrics: documented symbols / total symbols ratio

### Content Processing
- Technical quality scoring via metrics (accuracy, completeness, specification clarity)
- Grammar correction using statistical language models
- Rendering performance: <100ms generation time per 1K source lines
- WCAG 2.1 AA compliance verification (alt text, contrast ratios, semantic HTML)
- Technical search optimization: technical term indexing, API documentation markup
- Promotional language filtering pipeline

### Review Process
- Automated quality scoring
- Issue identification and reporting
- Improvement recommendations
- Progress tracking
- Quality metrics measurement

## Integration

### Development Workflow
- Pre-commit hook integration
- CI/CD pipeline support
- Automated documentation updates
- Version synchronization
- Change notifications

### Publishing Platforms
- GitHub Pages integration
- Static site generator support
- Academic publication systems
- Documentation hosting services
- Content management systems

### Version Control
- Git integration for version control
- Issue tracking system connections
- Team communication platforms
- Review workflow automation
- Project management integration

## Requirements

- Language-specific documentation tools
- Technical template processing engines with language filtering
- Format conversion utilities
- Web browser for interactive features
- Network access for collaborative features
- Promotional language filter system (promotional_language_filter.py)
- Technical optimization system (technical_optimization.py)
- Technical template system (technical_templates.py)

## Language Control System

### Promotional Language Prevention
- Automatic detection and removal of marketing language
- Technical communication enforcement
- Quality scoring based on technical accuracy
- Template filtering for direct technical language

### Technical Standards
- Function and class documentation with precise specifications
- Code examples with proper syntax and error handling
- Implementation details and technical requirements
- Academic and professional language standards for research documentation

## Common Workflows

### Basic Documentation Update
```bash
# 1. Generate README documentation
/update-docs --type=readme --format=markdown

# 2. Optimize content quality
/update-docs --type=readme --optimize

# 3. Commit documentation changes
/commit --template=docs --ai-message
```

### API Documentation Generation
```bash
# 1. Extract API documentation from code
/update-docs --type=api --format=html --optimize

# 2. Add interactive elements
/update-docs --type=api --interactive

# 3. Validate documentation completeness
/double-check "API documentation completeness" --deep-analysis
```

### Research Publication Workflow
```bash
# 1. Generate research documentation
/update-docs --type=research --format=latex --publish

# 2. Optimize for academic standards
/update-docs --type=research --optimize --collaborative

# 3. Prepare for submission
/reflection --type=scientific --export-insights
```

## Related Commands

**Prerequisites**: Commands to run before documentation generation
- `/explain-code` - Understand code structure before documenting
- `/check-code-quality` - Fix quality issues before documentation

**Alternatives**: Different documentation approaches
- `/explain-code --docs` - Code-focused documentation generation
- Manual documentation writing

**Combinations**: Commands that work with update-docs
- `/generate-tests` - Document test coverage and testing strategies
- `/optimize` - Document performance improvements
- `/reflection` - Generate insights for documentation

**Follow-up**: Commands to run after documentation
- `/double-check` - Verify documentation completeness
- `/commit --template=docs` - Commit documentation changes
- `/ci-setup` - Automate documentation builds

## Integration Patterns

### Complete Documentation Pipeline
```bash
# Comprehensive documentation generation
/explain-code project/ --docs --recursive
/update-docs --type=all --format=html --optimize
/generate-tests --interactive  # Document testing approach
/double-check "documentation quality" --deep-analysis --auto-complete
```

### Research Publication Workflow
```bash
# Academic publication preparation
/update-docs --type=research --format=latex --publish
/run-all-tests --scientific --reproducible  # Document reproducibility
/reflection --type=scientific --export-insights
/commit --template=docs --validate
```

### API Documentation Maintenance
```bash
# Automated API documentation updates
/check-code-quality --auto-fix  # Ensure clean code for documentation
/update-docs --type=api --interactive --optimize
/commit --template=docs --ai-message
/ci-setup --type=enterprise  # Automate documentation builds
```