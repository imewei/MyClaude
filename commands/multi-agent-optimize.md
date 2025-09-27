---
model: claude-sonnet-4-0
---

Comprehensive application optimization using coordinated multi-agent strategy:

[Extended thinking: This advanced optimization system coordinates multiple specialized agents in parallel and sequential phases to achieve holistic performance improvements. It includes baseline analysis, coordinated optimization execution, conflict resolution, validation testing, and rollback capabilities. The system ensures optimizations work synergistically while maintaining application stability.]

## Phase 1: Baseline Analysis & Planning

### Performance Assessment
Use Task tool with subagent_type="performance-monitor" to establish baseline:

Prompt: "Establish comprehensive performance baseline for: $ARGUMENTS. Collect and analyze:
1. Current performance metrics (response times, throughput, resource usage)
2. Identify performance bottlenecks across the stack
3. User experience impact assessment
4. Create performance baseline report for optimization tracking
5. Prioritize optimization opportunities by impact/effort ratio"

### Architecture Analysis
Use Task tool with subagent_type="architect-reviewer" to analyze system design:

Prompt: "Analyze system architecture for optimization opportunities in: $ARGUMENTS. Review:
1. Overall system architecture and scalability bottlenecks
2. Component interaction patterns and optimization potential
3. Technology stack assessment for performance improvements
4. Architectural patterns that could improve performance
5. Infrastructure scaling opportunities"

## Phase 2: Parallel Optimization Analysis

Run the following agents IN PARALLEL using multiple Task tool calls in a single message:

### Database Layer Optimization
Use Task tool with subagent_type="database-optimizer":

Prompt: "Optimize database layer for: $ARGUMENTS. Analyze and provide detailed recommendations for:
1. Query performance analysis with execution plans
2. Index optimization strategies (create, modify, remove)
3. Schema design improvements for performance
4. Connection pooling and configuration tuning
5. Database-level caching implementation
6. Provide specific SQL optimizations and configuration changes"

### Application Performance Optimization
Use Task tool with subagent_type="performance-engineer":

Prompt: "Optimize application performance for: $ARGUMENTS. Provide detailed analysis and solutions for:
1. CPU and memory profiling with bottleneck identification
2. Algorithm and data structure optimization opportunities
3. Concurrency and async operation improvements
4. Application-level caching strategies
5. Resource usage optimization
6. Provide specific code optimizations and performance improvements"

### Frontend Performance Optimization
Use Task tool with subagent_type="frontend-developer":

Prompt: "Optimize frontend performance for: $ARGUMENTS. Analyze and improve:
1. Bundle size analysis and reduction strategies
2. Lazy loading and code splitting implementation
3. Rendering optimization and virtual DOM improvements
4. Core Web Vitals optimization (LCP, FID, CLS, INP)
5. Network request optimization and caching
6. Provide specific frontend optimizations and implementation plans"

### Security Performance Optimization
Use Task tool with subagent_type="security-engineer":

Prompt: "Optimize security performance for: $ARGUMENTS. Balance security and performance by:
1. Efficient authentication and authorization mechanisms
2. Optimized encryption/decryption operations
3. Security middleware performance tuning
4. Rate limiting and DDoS protection efficiency
5. Secure caching strategies
6. Provide security optimizations that improve rather than hinder performance"

### Infrastructure Optimization
Use Task tool with subagent_type="devops-engineer":

Prompt: "Optimize infrastructure and deployment for: $ARGUMENTS. Focus on:
1. Container and orchestration optimization
2. CI/CD pipeline performance improvements
3. Infrastructure scaling and auto-scaling configuration
4. Network and CDN optimization
5. Monitoring and alerting performance optimization
6. Provide infrastructure optimizations and deployment improvements"

### Build System Optimization
Use Task tool with subagent_type="build-engineer":

Prompt: "Optimize build and development workflow for: $ARGUMENTS. Improve:
1. Build time reduction and incremental builds
2. Development server performance
3. Hot reload and development experience optimization
4. Bundle optimization and tree shaking
5. Development tooling performance
6. Provide build system optimizations and workflow improvements"

## Phase 3: Optimization Coordination & Validation

### Results Analysis
Use Task tool with subagent_type="data-analyst" to consolidate findings:

Prompt: "Analyze optimization recommendations from all agents for: $ARGUMENTS. Provide:
1. Consolidated optimization priority matrix (impact vs effort)
2. Dependency analysis between optimizations
3. Conflict identification and resolution strategies
4. Resource requirements estimation
5. Risk assessment for each optimization category
6. Create unified optimization roadmap with phases"

### Optimization Implementation Plan

#### Phase A: Foundation & Quick Wins (0-2 days)
- **Database**: Index creation, query optimization, connection tuning
- **Application**: Simple caching, obvious algorithm improvements
- **Frontend**: Bundle splitting, basic lazy loading, image optimization
- **Infrastructure**: CDN setup, basic monitoring improvements
- **Security**: Efficient middleware configuration
- **Build**: Incremental build setup, basic optimizations

#### Phase B: Intermediate Improvements (2-7 days)
- **Database**: Schema optimization, advanced caching implementation
- **Application**: Concurrency improvements, memory optimization
- **Frontend**: Advanced lazy loading, Core Web Vitals optimization
- **Infrastructure**: Auto-scaling configuration, advanced monitoring
- **Security**: Performance-optimized security patterns
- **Build**: Advanced build optimizations, development workflow improvements

#### Phase C: Advanced Optimizations (1-3 weeks)
- **Database**: Major schema changes, distributed caching
- **Application**: Architecture refactoring, advanced algorithms
- **Frontend**: Micro-frontend optimization, advanced rendering patterns
- **Infrastructure**: Multi-region deployment, advanced scaling
- **Security**: Zero-trust architecture optimization
- **Build**: Monorepo optimization, advanced toolchain improvements

## Phase 4: Implementation & Validation

### Pre-Implementation Validation
Use Task tool with subagent_type="qa-expert" for safety checks:

Prompt: "Validate optimization plan safety for: $ARGUMENTS. Ensure:
1. Comprehensive test coverage for optimization areas
2. Rollback procedures for each optimization
3. Performance monitoring setup for tracking improvements
4. Compatibility analysis with existing systems
5. Risk mitigation strategies for each phase
6. Create detailed validation checklist"

### Implementation Execution
For each optimization phase:

1. **Backup Current State**: Create system snapshots and backups
2. **Implement Changes**: Apply optimizations in controlled manner
3. **Monitor Performance**: Track metrics during and after changes
4. **Validate Results**: Confirm improvements meet expectations
5. **Document Changes**: Record what was done and results achieved

### Post-Implementation Analysis
Use Task tool with subagent_type="performance-monitor" for results tracking:

Prompt: "Analyze optimization results for: $ARGUMENTS. Measure:
1. Before/after performance comparison across all metrics
2. User experience impact measurement
3. Resource usage improvements
4. Stability and reliability impact
5. Cost implications of optimizations
6. Generate comprehensive optimization report"

## Phase 5: Continuous Optimization

### Ongoing Monitoring Setup
Use Task tool with subagent_type="sre-engineer" for long-term optimization:

Prompt: "Establish continuous optimization monitoring for: $ARGUMENTS. Setup:
1. Automated performance regression detection
2. Optimization opportunity identification system
3. Alert system for performance degradation
4. Regular optimization review cycles
5. Performance baseline updates
6. Create maintenance and monitoring documentation"

### Error Handling & Rollback Strategies

#### Rollback Procedures
- **Database**: Transaction logs, schema versioning, data backups
- **Application**: Code versioning, feature flags, gradual rollouts
- **Frontend**: Asset versioning, CDN invalidation, progressive deployment
- **Infrastructure**: Infrastructure-as-code versioning, blue-green deployments
- **Security**: Configuration backups, phased security updates
- **Build**: Build artifact versioning, environment restoration

#### Monitoring & Alerts
- Performance threshold alerts
- Error rate monitoring
- Resource usage spike detection
- User experience metric tracking
- Automated rollback triggers

### Success Metrics & KPIs

#### Performance Metrics
- **Database**: Query response time, connection efficiency, cache hit rates
- **Application**: Response times, throughput, resource usage, error rates
- **Frontend**: Core Web Vitals, bundle sizes, load times, user interactions
- **Infrastructure**: Uptime, scaling efficiency, cost optimization
- **Security**: Security scan performance, authentication speed
- **Build**: Build times, deployment frequency, development productivity

#### Business Impact
- User satisfaction improvements
- Conversion rate changes
- Cost savings achieved
- Developer productivity gains
- System reliability improvements

## Command Usage

### Basic Usage
```
/multi-agent-optimize [target-application-or-component]
```

### Advanced Usage
```
/multi-agent-optimize [target] --phase=[all|baseline|analysis|implementation|monitoring] --priority=[performance|cost|scalability|security] --scope=[frontend|backend|database|infrastructure|all]
```

### Arguments
- `target`: Application, service, or component to optimize
- `--phase`: Specific optimization phase to focus on
- `--priority`: Primary optimization goal
- `--scope`: Limit optimization to specific layers

Target for optimization: $ARGUMENTS