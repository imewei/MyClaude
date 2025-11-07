# Agent Orchestration Patterns

Guide to effectively orchestrating specialized agents throughout the feature development lifecycle.

## Table of Contents
- [Agent Overview](#agent-overview)
- [Orchestration Patterns](#orchestration-patterns)
- [Context Passing](#context-passing)
- [Parallel vs Sequential Execution](#parallel-vs-sequential-execution)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Agent Overview

### Available Agents by Phase

#### Phase 1: Discovery & Requirements
| Agent | Subagent Type | Capabilities | When to Use |
|-------|---------------|--------------|-------------|
| **Architect Review** | `comprehensive-review:architect-review` | System design, architecture patterns, scalability | Technical architecture design (Step 2) |
| **Security Auditor** | `comprehensive-review:security-auditor` | Security assessment, compliance, threat modeling | Risk assessment (Step 3), Security validation (Step 8) |

#### Phase 2: Implementation
| Agent | Subagent Type | Capabilities | When to Use |
|-------|---------------|--------------|-------------|
| **Backend Architect** | `backend-development:backend-architect` | API design, microservices, data modeling | Backend services implementation (Step 4) |
| **Frontend Developer** | `frontend-mobile-development:frontend-developer` | React/Vue/Angular, responsive UI, state management | Frontend implementation (Step 5) |
| **GraphQL Architect** | `backend-development:graphql-architect` | GraphQL schema design, resolvers, federation | GraphQL API implementation (alternative to REST) |

#### Phase 3: Testing & Quality
| Agent | Subagent Type | Capabilities | When to Use |
|-------|---------------|--------------|-------------|
| **Test Automator** | `unit-testing:test-automator` | Unit, integration, E2E tests, test strategies | Test suite creation (Step 7) |
| **Security Auditor** | `comprehensive-review:security-auditor` | OWASP, penetration testing, vulnerability scanning | Security validation (Step 8) |
| **Performance Engineer** | `full-stack-orchestration:performance-engineer` | Profiling, optimization, caching, load testing | Performance optimization (Step 9) |
| **Code Reviewer** | `comprehensive-review:code-reviewer` | Code quality, best practices, static analysis | Code review (anytime) |

#### Phase 4: Deployment & Monitoring
| Agent | Subagent Type | Capabilities | When to Use |
|-------|---------------|--------------|-------------|
| **Deployment Engineer** | `cicd-automation:deployment-engineer` | CI/CD pipelines, GitOps, progressive delivery | Deployment pipeline (Step 10) |
| **Observability Engineer** | `observability-monitoring:observability-engineer` | Metrics, logging, tracing, alerting | Observability setup (Step 11) |
| **Docs Architect** | `code-documentation:docs-architect` | API docs, user guides, architecture docs | Documentation generation (Step 12) |

---

## Orchestration Patterns

### Pattern 1: Sequential Dependency Chain

**When to Use**: Each step depends on the output of the previous step.

**Example**: Architecture Design → Backend Implementation → Frontend Implementation

```typescript
// Step 2: Architecture Design
const architectureDesign = await Task({
  subagent_type: "comprehensive-review:architect-review",
  prompt: `Design technical architecture for feature: ${featureName}.

    Requirements: ${businessRequirements}

    Define service boundaries, API contracts, data models, and technology stack.`,
  description: "Design technical architecture"
});

// Step 4: Backend Implementation (depends on Step 2)
const backendImplementation = await Task({
  subagent_type: "backend-development:backend-architect",
  prompt: `Implement backend services for: ${featureName}.

    Technical Design: ${architectureDesign}  // ← Context from previous step

    Build APIs, business logic, database integration, and feature flags.`,
  description: "Implement backend services"
});

// Step 5: Frontend Implementation (depends on Step 4)
const frontendImplementation = await Task({
  subagent_type: "frontend-mobile-development:frontend-developer",
  prompt: `Build frontend components for: ${featureName}.

    Backend APIs: ${backendImplementation.apiEndpoints}  // ← Context from previous step

    Implement responsive UI, state management, and analytics tracking.`,
  description: "Implement frontend components"
});
```

### Pattern 2: Parallel Independent Tasks

**When to Use**: Tasks can be executed simultaneously without dependencies.

**Example**: Security Validation + Performance Optimization (both analyze the same implementation)

```typescript
// Step 8 and Step 9 can run in parallel
const [securityValidation, performanceOptimization] = await Promise.all([
  // Step 8: Security Validation
  Task({
    subagent_type: "comprehensive-review:security-auditor",
    prompt: `Perform security testing for: ${featureName}.

      Implementation: ${implementationCode}

      Run OWASP checks, penetration testing, and compliance validation.`,
    description: "Validate security"
  }),

  // Step 9: Performance Optimization
  Task({
    subagent_type: "full-stack-orchestration:performance-engineer",
    prompt: `Optimize performance for: ${featureName}.

      Backend: ${backendCode}
      Frontend: ${frontendCode}

      Profile code, optimize queries, and improve load times.`,
    description: "Optimize performance"
  })
]);

// Both results available simultaneously
console.log('Security findings:', securityValidation.vulnerabilities);
console.log('Performance improvements:', performanceOptimization.optimizations);
```

### Pattern 3: Multi-Agent Review (Consensus)

**When to Use**: Critical decisions requiring multiple expert perspectives.

**Example**: Architecture review by multiple agents for comprehensive feedback

```typescript
const [architectReview, securityReview, performanceReview] = await Promise.all([
  Task({
    subagent_type: "comprehensive-review:architect-review",
    prompt: `Review proposed architecture for: ${featureName}.

      Assess: Scalability, maintainability, and design patterns.`,
    description: "Architecture review"
  }),

  Task({
    subagent_type: "comprehensive-review:security-auditor",
    prompt: `Review proposed architecture for security concerns.

      Assess: Authentication, authorization, data protection, and compliance.`,
    description: "Security review"
  }),

  Task({
    subagent_type: "full-stack-orchestration:performance-engineer",
    prompt: `Review proposed architecture for performance implications.

      Assess: Query patterns, caching strategy, and scalability bottlenecks.`,
    description: "Performance review"
  })
]);

// Synthesize multi-agent feedback
const consensusDecision = synthesizeReviews([
  architectReview,
  securityReview,
  performanceReview
]);
```

### Pattern 4: Iterative Refinement

**When to Use**: Agent output needs refinement based on validation or feedback.

**Example**: Test generation → Test execution → Fix failures → Repeat

```typescript
let testsPassing = false;
let iteration = 0;
const maxIterations = 3;

while (!testsPassing && iteration < maxIterations) {
  iteration++;

  // Step 7: Generate tests
  const testSuite = await Task({
    subagent_type: "unit-testing:test-automator",
    prompt: `Create comprehensive test suite for: ${featureName}.

      Implementation: ${implementationCode}
      ${iteration > 1 ? `Previous failures: ${previousFailures}` : ''}

      Ensure 80% code coverage.`,
    description: `Generate tests (iteration ${iteration})`
  });

  // Execute tests
  const testResults = await runTests(testSuite);

  if (testResults.allPassed) {
    testsPassing = true;
    console.log('All tests passing!');
  } else {
    console.log(`Iteration ${iteration}: ${testResults.failureCount} failures`);
    previousFailures = testResults.failures;
  }
}
```

### Pattern 5: Agent Specialization by Layer

**When to Use**: Complex features requiring multiple implementation layers.

**Example**: Full-stack feature implementation

```typescript
// Orchestrate multiple agents for different layers
const featureImplementation = {
  // Backend layer
  backend: await Task({
    subagent_type: "backend-development:backend-architect",
    prompt: `Implement backend for: ${featureName}`,
    description: "Backend implementation"
  }),

  // Frontend layer
  frontend: await Task({
    subagent_type: "frontend-mobile-development:frontend-developer",
    prompt: `Implement frontend for: ${featureName}

      Backend APIs: ${backend.endpoints}`,
    description: "Frontend implementation"
  }),

  // Mobile layer (if applicable)
  mobile: await Task({
    subagent_type: "multi-platform-apps:mobile-developer",
    prompt: `Implement mobile app for: ${featureName}

      Backend APIs: ${backend.endpoints}`,
    description: "Mobile implementation"
  }),

  // Infrastructure layer
  infrastructure: await Task({
    subagent_type: "cicd-automation:deployment-engineer",
    prompt: `Set up infrastructure for: ${featureName}`,
    description: "Infrastructure setup"
  })
};
```

---

## Context Passing

### Effective Context Passing Strategies

#### 1. **Full Context Inclusion**
Pass all relevant information from previous steps.

```typescript
const frontendImplementation = await Task({
  subagent_type: "frontend-mobile-development:frontend-developer",
  prompt: `Build frontend for: ${featureName}.

    === BUSINESS REQUIREMENTS ===
    ${businessRequirements}

    === TECHNICAL ARCHITECTURE ===
    ${architectureDesign}

    === BACKEND APIS ===
    ${backendAPIs}

    === USER STORIES ===
    ${userStories}

    Build responsive UI integrating with the backend APIs above.`,
  description: "Frontend implementation"
});
```

#### 2. **Summarized Context**
For long-running workflows, summarize key points to reduce token usage.

```typescript
const summary = {
  feature: featureName,
  apiEndpoints: backendAPIs.endpoints.map(e => `${e.method} ${e.path}`),
  dataModels: architectureDesign.models.map(m => m.name),
  securityRequirements: riskAssessment.criticalRequirements
};

const deploymentPipeline = await Task({
  subagent_type: "cicd-automation:deployment-engineer",
  prompt: `Set up CI/CD pipeline for: ${featureName}.

    Summary: ${JSON.stringify(summary, null, 2)}

    Create automated deployment with testing and rollback capabilities.`,
  description: "Deployment pipeline setup"
});
```

#### 3. **Reference Links**
Point to files or documentation instead of inlining large content.

```typescript
const documentationGeneration = await Task({
  subagent_type: "code-documentation:docs-architect",
  prompt: `Generate documentation for: ${featureName}.

    Source files:
    - Backend: /src/features/${featureName}/backend/
    - Frontend: /src/features/${featureName}/frontend/
    - Tests: /tests/features/${featureName}/

    Architecture doc: /docs/architecture/${featureName}.md

    Generate API docs, user guides, and deployment runbooks.`,
  description: "Documentation generation"
});
```

---

## Parallel vs Sequential Execution

### When to Run in Parallel

✅ **DO run in parallel when**:
- Tasks are independent (no data dependencies)
- Faster completion is critical
- Tasks analyze the same artifact (e.g., security + performance review)

```typescript
// ✅ GOOD: Parallel execution for independent reviews
const [securityResult, performanceResult, codeQualityResult] = await Promise.all([
  Task({ subagent_type: "comprehensive-review:security-auditor", ... }),
  Task({ subagent_type: "full-stack-orchestration:performance-engineer", ... }),
  Task({ subagent_type: "comprehensive-review:code-reviewer", ... })
]);
```

### When to Run Sequentially

✅ **DO run sequentially when**:
- Later tasks depend on earlier outputs
- Resource constraints (API rate limits, compute resources)
- Order matters for correctness

```typescript
// ✅ GOOD: Sequential execution for dependent steps
const architecture = await Task({ subagent_type: "architect-review", ... });
const backend = await Task({
  subagent_type: "backend-architect",
  prompt: `Use architecture: ${architecture}`, // ← Dependency
  ...
});
const frontend = await Task({
  subagent_type: "frontend-developer",
  prompt: `Integrate with APIs: ${backend.endpoints}`, // ← Dependency
  ...
});
```

---

## Error Handling

### Agent Failure Strategies

#### 1. **Retry with Exponential Backoff**

```typescript
async function executeAgentWithRetry(config, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await Task(config);
    } catch (error) {
      if (attempt === maxRetries) throw error;

      const backoffMs = Math.pow(2, attempt) * 1000;
      console.log(`Agent failed (attempt ${attempt}/${maxRetries}), retrying in ${backoffMs}ms...`);
      await sleep(backoffMs);
    }
  }
}

const result = await executeAgentWithRetry({
  subagent_type: "backend-development:backend-architect",
  prompt: "Implement backend...",
  description: "Backend implementation"
});
```

#### 2. **Fallback to Alternative Agent**

```typescript
async function executeWithFallback(primaryConfig, fallbackConfig) {
  try {
    return await Task(primaryConfig);
  } catch (primaryError) {
    console.warn('Primary agent failed, trying fallback...', primaryError);
    try {
      return await Task(fallbackConfig);
    } catch (fallbackError) {
      throw new Error(`Both primary and fallback agents failed: ${fallbackError}`);
    }
  }
}

const result = await executeWithFallback(
  {
    subagent_type: "backend-development:graphql-architect",
    prompt: "Build GraphQL API...",
    description: "GraphQL implementation"
  },
  {
    subagent_type: "backend-development:backend-architect",
    prompt: "Build REST API as fallback...",
    description: "REST implementation (fallback)"
  }
);
```

#### 3. **Partial Success Handling**

```typescript
const parallelTasks = [
  Task({ subagent_type: "security-auditor", ... }),
  Task({ subagent_type: "performance-engineer", ... }),
  Task({ subagent_type: "code-reviewer", ... })
];

const results = await Promise.allSettled(parallelTasks);

const successes = results.filter(r => r.status === 'fulfilled').map(r => r.value);
const failures = results.filter(r => r.status === 'rejected').map(r => r.reason);

if (failures.length > 0) {
  console.warn(`${failures.length} agents failed:`, failures);
}

if (successes.length === 0) {
  throw new Error('All agents failed');
}

// Proceed with partial results
return {
  results: successes,
  partialFailure: failures.length > 0,
  failureReasons: failures
};
```

---

## Best Practices

### 1. **Clear Agent Prompts**

❌ **Bad**: Vague, multi-purpose prompt
```typescript
const result = await Task({
  subagent_type: "backend-development:backend-architect",
  prompt: "Build the backend for the feature",
  description: "Backend work"
});
```

✅ **Good**: Specific, actionable prompt with clear deliverables
```typescript
const result = await Task({
  subagent_type: "backend-development:backend-architect",
  prompt: `Implement backend services for user authentication feature.

    Requirements:
    - OAuth 2.0 authentication with JWT tokens
    - User registration, login, logout endpoints
    - Password reset flow with email verification
    - Role-based access control (admin, user)

    Technical constraints:
    - Node.js + Express
    - PostgreSQL database
    - Redis for session storage

    Deliverables:
    - RESTful API endpoints
    - Database migrations
    - Unit tests (>80% coverage)
    - API documentation (Swagger)`,
  description: "Implement authentication backend"
});
```

### 2. **Appropriate Agent Selection**

Choose agents based on their specialization:

| Task | ❌ Wrong Agent | ✅ Right Agent |
|------|----------------|----------------|
| Design system architecture | backend-architect | **architect-review** |
| Security vulnerability scanning | code-reviewer | **security-auditor** |
| CI/CD pipeline setup | backend-architect | **deployment-engineer** |
| Performance optimization | code-reviewer | **performance-engineer** |
| API documentation | backend-architect | **docs-architect** |

### 3. **Context Hygiene**

Only pass relevant context to reduce token usage and improve focus:

❌ **Bad**: Passing irrelevant context
```typescript
const docs = await Task({
  subagent_type: "code-documentation:docs-architect",
  prompt: `Generate docs.

    Full codebase (500,000 lines): ${entireCodebase}  // ❌ Irrelevant
    Full database dump: ${databaseSchema}  // ❌ Irrelevant
    Generate documentation.`,
});
```

✅ **Good**: Passing only relevant context
```typescript
const docs = await Task({
  subagent_type: "code-documentation:docs-architect",
  prompt: `Generate API documentation for user service.

    API endpoints: ${userServiceAPIs}  // ✅ Relevant
    Data models: ${userModels}  // ✅ Relevant

    Generate OpenAPI specification and usage examples.`,
});
```

### 4. **Progress Tracking**

Track agent execution for visibility and debugging:

```typescript
const workflow = {
  steps: [
    { name: "Architecture Design", status: "pending", agent: "architect-review" },
    { name: "Security Assessment", status: "pending", agent: "security-auditor" },
    { name: "Backend Implementation", status: "pending", agent: "backend-architect" },
    { name: "Frontend Implementation", status: "pending", agent: "frontend-developer" },
  ]
};

for (const step of workflow.steps) {
  step.status = "in_progress";
  console.log(`Starting: ${step.name} (${step.agent})`);

  try {
    step.result = await Task({
      subagent_type: step.agent,
      prompt: getPromptFor(step.name),
      description: step.name
    });
    step.status = "completed";
    console.log(`✓ Completed: ${step.name}`);
  } catch (error) {
    step.status = "failed";
    step.error = error.message;
    console.error(`✗ Failed: ${step.name}`, error);
    throw error; // or continue based on criticality
  }
}
```

### 5. **Agent Composition**

Combine multiple agents for complex tasks:

```typescript
async function implementFullFeature(featureSpec) {
  // Phase 1: Design
  const [architecture, riskAssessment] = await Promise.all([
    Task({ subagent_type: "architect-review", ... }),
    Task({ subagent_type: "security-auditor", ... })
  ]);

  // Phase 2: Implementation (sequential - backend before frontend)
  const backend = await Task({
    subagent_type: "backend-architect",
    prompt: `Implement backend using: ${architecture}`,
    ...
  });

  const frontend = await Task({
    subagent_type: "frontend-developer",
    prompt: `Implement frontend using: ${backend.endpoints}`,
    ...
  });

  // Phase 3: Quality (parallel)
  const [tests, security, performance] = await Promise.all([
    Task({ subagent_type: "test-automator", ... }),
    Task({ subagent_type: "security-auditor", ... }),
    Task({ subagent_type: "performance-engineer", ... })
  ]);

  // Phase 4: Deployment
  const [deployment, monitoring, docs] = await Promise.all([
    Task({ subagent_type: "deployment-engineer", ... }),
    Task({ subagent_type: "observability-engineer", ... }),
    Task({ subagent_type: "docs-architect", ... })
  ]);

  return {
    architecture,
    implementation: { backend, frontend },
    quality: { tests, security, performance },
    operations: { deployment, monitoring, docs }
  };
}
```

---

## Advanced Patterns

### Pattern: Agent Feedback Loop

Use agent output to refine prompts for subsequent agents:

```typescript
// Initial implementation
let implementation = await Task({
  subagent_type: "backend-architect",
  prompt: "Implement user service",
  description: "Initial implementation"
});

// Review by code reviewer
const codeReview = await Task({
  subagent_type: "code-reviewer",
  prompt: `Review this implementation: ${implementation.code}`,
  description: "Code review"
});

// If issues found, refine implementation
if (codeReview.issues.length > 0) {
  implementation = await Task({
    subagent_type: "backend-architect",
    prompt: `Refine implementation based on review:

      Original implementation: ${implementation.code}
      Review feedback: ${codeReview.issues}

      Address all issues raised in the review.`,
    description: "Refined implementation"
  });
}
```

### Pattern: Multi-Stage Agent Pipeline

Build complex workflows with multiple stages:

```typescript
const featurePipeline = {
  planning: async () => ({
    requirements: await Task({ subagent_type: "manual", ... }),
    architecture: await Task({ subagent_type: "architect-review", ... }),
    risks: await Task({ subagent_type: "security-auditor", ... })
  }),

  implementation: async (planning) => ({
    backend: await Task({
      subagent_type: "backend-architect",
      prompt: `Use architecture: ${planning.architecture}`,
      ...
    }),
    frontend: await Task({
      subagent_type: "frontend-developer",
      prompt: `Integrate with: ${planning.backend}`,
      ...
    })
  }),

  quality: async (implementation) => Promise.all([
    Task({ subagent_type: "test-automator", prompt: `Test: ${implementation}`, ... }),
    Task({ subagent_type: "security-auditor", prompt: `Scan: ${implementation}`, ... }),
    Task({ subagent_type: "performance-engineer", prompt: `Optimize: ${implementation}`, ... })
  ]),

  deployment: async (quality) => ({
    pipeline: await Task({ subagent_type: "deployment-engineer", ... }),
    monitoring: await Task({ subagent_type: "observability-engineer", ... }),
    docs: await Task({ subagent_type: "docs-architect", ... })
  })
};

// Execute pipeline
const planning = await featurePipeline.planning();
const implementation = await featurePipeline.implementation(planning);
const quality = await featurePipeline.quality(implementation);
const deployment = await featurePipeline.deployment(quality);
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Agent timeout** | Prompt too complex | Break into smaller tasks |
| **Inconsistent output** | Insufficient context | Provide more specific requirements |
| **Agent conflicts** | Parallel agents modifying same code | Run sequentially or coordinate changes |
| **Token limit exceeded** | Context too large | Summarize or reference files |
| **Agent selecting wrong approach** | Ambiguous prompt | Add explicit constraints and examples |

---

## References

- [Feature Development Command](../../commands/feature-development.md)
- [Phase Templates](./phase-templates.md)
- [Best Practices](./best-practices.md)
