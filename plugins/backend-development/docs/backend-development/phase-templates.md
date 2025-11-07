# Feature Development Phase Templates

Detailed templates and guides for each step of the 4-phase feature development workflow.

## Table of Contents

### Phase 1: Discovery & Requirements Planning
- [Business Analysis](#business-analysis)
- [Architecture Design](#architecture-design)
- [Risk Assessment](#risk-assessment)

### Phase 2: Implementation & Development
- [Backend Implementation](#backend-implementation)
- [Frontend Implementation](#frontend-implementation)
- [Data Pipeline](#data-pipeline)

### Phase 3: Testing & Quality Assurance
- [Automated Testing](#automated-testing)
- [Security Validation](#security-validation)
- [Performance Optimization](#performance-optimization)

### Phase 4: Deployment & Monitoring
- [Deployment Pipeline](#deployment-pipeline)
- [Observability](#observability)
- [Documentation](#documentation)

---

## Phase 1: Discovery & Requirements Planning

### Business Analysis

**Objective**: Transform feature requests into clear, actionable requirements with defined success criteria.

**Template**:

```markdown
# Feature Requirements: [Feature Name]

## Executive Summary
- **Business Value**: [What problem does this solve?]
- **Target Users**: [Who will use this feature?]
- **Success Metrics**: [How will we measure success?]
- **Timeline**: [Expected delivery date]
- **Priority**: [High/Medium/Low]

## User Stories

### Primary User Story
As a [user type]
I want to [action]
So that [benefit]

**Acceptance Criteria**:
- [ ] Given [context], when [action], then [expected result]
- [ ] Given [context], when [action], then [expected result]
- [ ] Given [context], when [action], then [expected result]

### Additional User Stories
[Repeat format for each user story]

## Functional Requirements

### Core Functionality
1. **Requirement 1**: [Description]
   - **Priority**: Must Have / Should Have / Nice to Have
   - **Complexity**: Low / Medium / High
   - **Dependencies**: [Other features, services, or systems]

2. **Requirement 2**: [Description]
   ...

### Edge Cases and Error Handling
- **Scenario 1**: [Edge case description]
  - **Expected Behavior**: [How system should respond]
- **Scenario 2**: [Edge case description]
  ...

## Non-Functional Requirements

### Performance
- Response time: < [X]ms for [Y] operation
- Throughput: Support [N] requests per second
- Database query time: < [X]ms for [Y] queries

### Scalability
- Handle [N] concurrent users
- Support [X] GB of data
- Scale to [N] transactions per day

### Security
- Authentication: [Method]
- Authorization: [RBAC, ABAC, etc.]
- Data encryption: [At rest / in transit]
- Compliance: [GDPR, HIPAA, SOC2, etc.]

### Reliability
- Uptime: [99.9%, 99.99%, etc.]
- RPO (Recovery Point Objective): [X minutes]
- RTO (Recovery Time Objective): [X minutes]

## Success Metrics

### Business Metrics
- **Primary Metric**: [e.g., conversion rate increase by 15%]
- **Secondary Metrics**: [e.g., reduced support tickets, increased engagement]

### Technical Metrics
- **Performance**: [p95 latency < 200ms]
- **Quality**: [< 0.1% error rate]
- **Adoption**: [50% of users using feature within 30 days]

## Stakeholders

| Role | Name | Responsibility | Contact |
|------|------|----------------|---------|
| Product Owner | [Name] | Feature vision | [Email] |
| Engineering Lead | [Name] | Technical delivery | [Email] |
| Design Lead | [Name] | UX/UI | [Email] |
| QA Lead | [Name] | Quality assurance | [Email] |

## Dependencies

### Internal Dependencies
- **Service A**: [Required integration]
- **Feature B**: [Must be completed first]

### External Dependencies
- **Third-party API**: [Vendor, purpose]
- **Infrastructure**: [New servers, database, etc.]

## Risks and Constraints

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| [Risk 1] | High/Medium/Low | High/Medium/Low | [Mitigation strategy] |
| [Risk 2] | ... | ... | ... |

### Constraints
- **Budget**: [$X]
- **Timeline**: [Fixed/Flexible]
- **Technology**: [Must use specific stack]
- **Team Size**: [N developers]

## Out of Scope

Explicitly define what is NOT included:
- [Feature X]
- [Functionality Y]
- [Integration Z]

## Appendix

### Glossary
- **Term 1**: Definition
- **Term 2**: Definition

### References
- [Link to original feature request]
- [Link to user research]
- [Link to competitive analysis]
```

**Validation Checklist**:
- [ ] All user stories have clear acceptance criteria
- [ ] Success metrics are measurable and time-bound
- [ ] Non-functional requirements are quantified
- [ ] Stakeholders have reviewed and approved
- [ ] Dependencies are identified and tracked
- [ ] Risks have mitigation strategies
- [ ] Out-of-scope items are explicitly listed

---

### Architecture Design

**Objective**: Design scalable, maintainable technical architecture aligned with business requirements.

**Agent Prompt Template**:
```
Design technical architecture for feature: [FEATURE_NAME].

Requirements:
[Paste business requirements from Step 1]

Define:
1. Service boundaries and responsibilities
2. API contracts (RESTful/GraphQL schemas)
3. Data models and relationships
4. Integration points with existing systems
5. Technology stack recommendations
6. Scalability and performance strategies
7. Security architecture
8. Deployment architecture

Consider:
- Microservices vs monolithic approach
- Synchronous vs asynchronous communication
- Caching strategies
- Database selection (SQL vs NoSQL)
- Authentication and authorization
- Error handling and resilience patterns
```

**Architecture Document Template**:

```markdown
# Technical Architecture: [Feature Name]

## System Overview

### High-Level Architecture Diagram
```
[Include C4 Context Diagram or similar]
```

### Architecture Style
- **Pattern**: [Microservices / Monolithic / Serverless / Event-Driven]
- **Rationale**: [Why this pattern fits the requirements]

## Service Design

### Service Boundaries

#### Service 1: [Service Name]
- **Responsibility**: [Single responsibility]
- **Owned Data**: [Data models this service owns]
- **Dependencies**: [Other services it calls]
- **Consumers**: [Who calls this service]

#### Service 2: [Service Name]
[Repeat for each service]

### Communication Patterns

| From Service | To Service | Pattern | Protocol | Rationale |
|--------------|------------|---------|----------|-----------|
| Frontend | API Gateway | Synchronous | HTTPS/REST | User-facing, requires immediate response |
| Order Service | Payment Service | Synchronous | gRPC | Strong consistency required |
| Payment Service | Notification Service | Asynchronous | Message Queue | Fire-and-forget |

## API Contracts

### REST API Specification

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: [Feature Name] API
  version: 1.0.0

paths:
  /api/v1/resources:
    get:
      summary: List resources
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Resource'
                  pagination:
                    $ref: '#/components/schemas/Pagination'

    post:
      summary: Create resource
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateResourceRequest'
      responses:
        '201':
          description: Resource created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Resource'
        '400':
          description: Invalid input
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    Resource:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        status:
          type: string
          enum: [active, inactive]
        createdAt:
          type: string
          format: date-time

    CreateResourceRequest:
      type: object
      required:
        - name
      properties:
        name:
          type: string
          minLength: 3
          maxLength: 100
        description:
          type: string

    Error:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object
```

### GraphQL Schema (if applicable)

```graphql
type Query {
  resource(id: ID!): Resource
  resources(
    page: Int = 1
    limit: Int = 20
    filter: ResourceFilter
  ): ResourceConnection!
}

type Mutation {
  createResource(input: CreateResourceInput!): Resource!
  updateResource(id: ID!, input: UpdateResourceInput!): Resource!
  deleteResource(id: ID!): Boolean!
}

type Resource {
  id: ID!
  name: String!
  status: ResourceStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
}

enum ResourceStatus {
  ACTIVE
  INACTIVE
}

input CreateResourceInput {
  name: String! @constraint(minLength: 3, maxLength: 100)
  description: String
}
```

## Data Models

### Entity-Relationship Diagram
```
[Include ERD]
```

### Database Schema

```sql
-- Primary entities
CREATE TABLE resources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(100) NOT NULL,
  description TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_by UUID NOT NULL REFERENCES users(id),

  CONSTRAINT resources_status_check CHECK (status IN ('active', 'inactive')),
  INDEX idx_resources_status ON resources(status),
  INDEX idx_resources_created_at ON resources(created_at DESC)
);

-- Audit trail
CREATE TABLE resource_audit_log (
  id BIGSERIAL PRIMARY KEY,
  resource_id UUID NOT NULL REFERENCES resources(id),
  action VARCHAR(20) NOT NULL,
  changed_by UUID NOT NULL REFERENCES users(id),
  changes JSONB,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  INDEX idx_audit_resource_id ON resource_audit_log(resource_id),
  INDEX idx_audit_timestamp ON resource_audit_log(timestamp DESC)
);
```

### Data Flow Diagram
```
[User Request] → [API Gateway] → [Backend Service] → [Database]
                     ↓
                [Cache Layer]
                     ↓
                [Message Queue] → [Worker Service] → [External API]
```

## Technology Stack

### Backend
- **Language**: [TypeScript, Python, Go, etc.]
- **Framework**: [Express, FastAPI, Gin, etc.]
- **Runtime**: [Node.js 20, Python 3.12, etc.]

### Database
- **Primary Database**: [PostgreSQL 16, MongoDB 7, etc.]
  - **Rationale**: [Why this database?]
- **Caching**: [Redis 7, Memcached, etc.]
- **Search**: [Elasticsearch 8, etc. (if applicable)]

### Message Queue (if applicable)
- **System**: [RabbitMQ, Kafka, AWS SQS, etc.]
- **Use Cases**: [Async processing, event streaming, etc.]

### Infrastructure
- **Hosting**: [AWS, GCP, Azure, On-premise]
- **Container**: [Docker, Kubernetes]
- **CI/CD**: [GitHub Actions, GitLab CI, Jenkins]

## Scalability Strategy

### Horizontal Scaling
- **Services**: Stateless design for easy scaling
- **Database**: Read replicas for read-heavy workloads
- **Caching**: Distributed cache (Redis Cluster)

### Performance Targets
| Metric | Target | Strategy |
|--------|--------|----------|
| Response Time (p95) | < 200ms | Caching, DB indexing, async processing |
| Throughput | 1000 req/s | Horizontal scaling, load balancing |
| Database Query Time | < 50ms | Optimized queries, indexes, connection pooling |

### Caching Strategy
```
- **L1 Cache (Application)**: In-memory cache for hot data (TTL: 1min)
- **L2 Cache (Redis)**: Distributed cache for frequently accessed data (TTL: 5min)
- **Cache Invalidation**: Event-driven invalidation on data mutations
```

## Security Architecture

### Authentication
- **Method**: [OAuth 2.0, JWT, Session-based, etc.]
- **Provider**: [Auth0, Cognito, Custom, etc.]
- **Token Expiry**: [15min access token, 7-day refresh token]

### Authorization
- **Model**: [RBAC, ABAC, etc.]
- **Roles**: [Admin, User, Guest, etc.]
- **Permissions**: [Read, Write, Delete, etc.]

### Data Protection
- **Encryption at Rest**: AES-256
- **Encryption in Transit**: TLS 1.3
- **Sensitive Fields**: [password, SSN, credit card] → Encrypted/Hashed
- **PII Handling**: GDPR-compliant data retention and deletion

### Security Headers
```typescript
helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
  },
})
```

## Resilience Patterns

### Circuit Breaker
```typescript
const circuitBreaker = new CircuitBreaker(callExternalAPI, {
  timeout: 3000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
});
```

### Retry Strategy
```typescript
const retryConfig = {
  retries: 3,
  factor: 2,
  minTimeout: 1000,
  maxTimeout: 5000,
  onRetry: (err, attempt) => logger.warn(`Retry attempt ${attempt}`, err),
};
```

### Rate Limiting
```typescript
const rateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP',
});
```

## Deployment Architecture

### Environment Strategy
- **Development**: Local + dev cloud environment
- **Staging**: Production-like environment for testing
- **Production**: Multi-region for HA (if applicable)

### Infrastructure as Code
```yaml
# k8s-deployment.yaml (example)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: feature-service
  template:
    metadata:
      labels:
        app: feature-service
    spec:
      containers:
      - name: feature-service
        image: myorg/feature-service:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 3
```

## Migration Strategy

### Data Migration
- **Approach**: [Blue-green, Rolling, etc.]
- **Rollback Plan**: [Database backups, version compatibility]
- **Zero-Downtime**: [Backward-compatible schema changes]

### Code Migration
- **Feature Flags**: Gradual rollout with flag controls
- **A/B Testing**: Split traffic for controlled experiment

## Open Questions

- [ ] Question 1: [Needs clarification from stakeholders]
- [ ] Question 2: [Technical decision pending]

## Approval

- [ ] Engineering Lead reviewed
- [ ] Security reviewed
- [ ] DevOps reviewed
- [ ] Product Owner approved
```

---

### Risk Assessment

**Objective**: Identify security risks, compliance requirements, and mitigation strategies.

**Agent Prompt Template**:
```
Perform security and risk assessment for feature: [FEATURE_NAME].

Technical Architecture:
[Paste architecture design from Step 2]

Assess:
1. Security vulnerabilities (OWASP Top 10)
2. Data privacy concerns (PII, sensitive data)
3. Compliance requirements (GDPR, HIPAA, SOC2, PCI-DSS)
4. Authentication and authorization risks
5. API security (rate limiting, input validation)
6. Infrastructure security
7. Supply chain security (dependencies)
8. Business continuity risks

Provide:
- Risk matrix with likelihood and impact
- Mitigation strategies for each risk
- Compliance checklist
- Security requirements for implementation
```

**Risk Assessment Template**:

```markdown
# Security & Risk Assessment: [Feature Name]

## Executive Summary
- **Overall Risk Level**: [Low / Medium / High / Critical]
- **Critical Findings**: [Number of critical issues]
- **Compliance Status**: [Compliant / Needs Remediation]

## Security Analysis

### OWASP Top 10 Assessment

| Vulnerability | Risk Level | Present? | Mitigation |
|---------------|------------|----------|------------|
| A01: Broken Access Control | High | ⚠️ Yes | Implement RBAC, enforce least privilege |
| A02: Cryptographic Failures | Medium | ✅ No | TLS 1.3, AES-256 encryption implemented |
| A03: Injection | High | ⚠️ Yes | Use parameterized queries, input validation |
| A04: Insecure Design | Medium | ✅ No | Threat modeling completed |
| A05: Security Misconfiguration | Medium | ⚠️ Yes | Harden default configs, security scanning |
| A06: Vulnerable Components | Medium | ⚠️ Yes | Dependency scanning, SCA tools |
| A07: Authentication Failures | High | ✅ No | OAuth 2.0 + MFA implemented |
| A08: Software/Data Integrity | Low | ✅ No | Code signing, SBOM generation |
| A09: Logging/Monitoring Failures | Medium | ⚠️ Yes | Enhanced logging, SIEM integration |
| A10: SSRF | Low | ✅ No | Allowlist for external calls |

### Data Privacy

**PII Inventory**:
| Data Type | Collected? | Storage | Retention | Encryption | Access Control |
|-----------|------------|---------|-----------|------------|----------------|
| Email | Yes | PostgreSQL | 2 years | At rest | RBAC |
| Password | Yes | PostgreSQL | N/A | Bcrypt hashed | Admin only |
| Payment Info | Yes | Stripe (3rd party) | Per PCI-DSS | Tokenized | No access |
| IP Address | Yes | Logs | 90 days | In transit | Security team |

**GDPR Compliance**:
- [ ] Right to Access: API endpoint for user data export
- [ ] Right to Erasure: Data deletion workflow implemented
- [ ] Right to Portability: JSON export format
- [ ] Data Processing Agreement: Signed with third-party vendors
- [ ] Privacy Policy: Updated to include new feature
- [ ] Cookie Consent: Implemented for analytics

## Risk Matrix

| Risk ID | Description | Likelihood | Impact | Risk Score | Priority |
|---------|-------------|------------|--------|------------|----------|
| R-001 | SQL Injection via unsanitized input | Medium | High | **High** | P0 |
| R-002 | Unauthorized access to admin endpoints | Low | Critical | **High** | P0 |
| R-003 | DDoS attack on public API | Medium | Medium | **Medium** | P1 |
| R-004 | Dependency vulnerability (Log4j-like) | Low | High | **Medium** | P1 |
| R-005 | Data breach due to misconfigured S3 | Low | Critical | **Medium** | P1 |
| R-006 | Session hijacking | Low | Medium | **Low** | P2 |

**Risk Scoring**:
- **Critical Impact + High Likelihood** = Critical Risk (P0 - Immediate remediation)
- **High Impact + Medium Likelihood** = High Risk (P0 - Remediate before launch)
- **Medium Impact + Medium Likelihood** = Medium Risk (P1 - Remediate within sprint)
- **Low Impact + Low Likelihood** = Low Risk (P2 - Monitor and track)

## Mitigation Strategies

### R-001: SQL Injection
**Current State**: Dynamic SQL queries constructed with string concatenation

**Mitigation**:
```typescript
// ❌ VULNERABLE
const query = `SELECT * FROM users WHERE email = '${userInput}'`;

// ✅ SECURE
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [userInput]);

// ✅ SECURE (ORM)
const user = await User.findOne({ where: { email: userInput } });
```

**Action Items**:
- [ ] Audit all database queries
- [ ] Replace dynamic queries with parameterized queries
- [ ] Enable strict TypeScript mode
- [ ] Add SQL injection tests
- [ ] Configure SAST tool to detect SQL injection patterns

### R-002: Unauthorized Access
**Current State**: Admin endpoints lack proper authorization checks

**Mitigation**:
```typescript
// ✅ SECURE
const authMiddleware = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const payload = await verifyJWT(token);
    req.user = payload;

    // Check role-based access
    if (req.path.startsWith('/admin') && !payload.roles.includes('admin')) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};
```

**Action Items**:
- [ ] Implement RBAC middleware
- [ ] Audit all endpoints for authorization
- [ ] Add authorization tests
- [ ] Principle of least privilege for all roles

## Compliance Requirements

### GDPR (if applicable)
- [x] Data Processing Agreement with vendors
- [x] Privacy Policy updated
- [ ] Data Protection Impact Assessment (DPIA) completed
- [ ] Data retention policy documented
- [ ] User consent mechanisms implemented

### HIPAA (if applicable)
- [ ] BAA signed with third-party vendors
- [ ] PHI encryption at rest and in transit
- [ ] Audit logging for PHI access
- [ ] Access controls (role-based)
- [ ] Regular security risk assessments

### SOC 2 Type II (if applicable)
- [ ] Security controls documented
- [ ] Change management process
- [ ] Incident response plan
- [ ] Access reviews quarterly
- [ ] Vendor risk management

### PCI-DSS (if applicable)
- [ ] No storage of full credit card numbers
- [ ] Tokenization via Stripe/payment processor
- [ ] Network segmentation
- [ ] Regular vulnerability scanning
- [ ] Penetration testing annually

## Security Requirements for Implementation

### Must-Have (P0)
1. ✅ **Input Validation**: Validate and sanitize all user inputs
2. ✅ **Authentication**: OAuth 2.0 with JWT
3. ✅ **Authorization**: RBAC on all endpoints
4. ✅ **Encryption**: TLS 1.3 for transit, AES-256 for rest
5. ✅ **SQL Injection Prevention**: Parameterized queries only
6. ✅ **XSS Prevention**: Content Security Policy headers
7. ✅ **CSRF Protection**: CSRF tokens for state-changing operations

### Should-Have (P1)
1. **Rate Limiting**: 100 req/min per IP
2. **Security Headers**: Helmet.js with strict CSP
3. **Dependency Scanning**: Snyk/Dependabot integration
4. **Secret Management**: AWS Secrets Manager / Vault
5. **Audit Logging**: Log all authentication and authorization events

### Nice-to-Have (P2)
1. **Penetration Testing**: Annual third-party pentest
2. **Bug Bounty Program**: HackerOne/Bugcrowd
3. **WAF**: AWS WAF or Cloudflare for DDoS protection
4. **SIEM Integration**: Splunk/DataDog security monitoring

## Security Testing Plan

### Static Analysis (SAST)
- **Tool**: [SonarQube, Semgrep, etc.]
- **Schedule**: On every commit (CI/CD)
- **Blockers**: Critical/High severity findings

### Dynamic Analysis (DAST)
- **Tool**: [OWASP ZAP, Burp Suite, etc.]
- **Schedule**: Weekly in staging
- **Scope**: All API endpoints

### Dependency Scanning (SCA)
- **Tool**: [Snyk, Dependabot, etc.]
- **Schedule**: Daily
- **Action**: Auto-PR for patch versions, manual review for major versions

### Penetration Testing
- **Frequency**: Annually + before major releases
- **Scope**: Full application security assessment
- **Provider**: [Third-party security firm]

## Approval

- [ ] Security team reviewed and approved
- [ ] Compliance team reviewed (if applicable)
- [ ] Engineering lead acknowledged risks and mitigations
- [ ] Product owner approved risk/timeline trade-offs
```

---

## Phase 2: Implementation & Development

### Backend Implementation

**Agent Prompt Template**:
```
Implement backend services for feature: [FEATURE_NAME].

Technical Design:
[Paste architecture from Step 2]

Requirements:
- Build RESTful/GraphQL APIs following OpenAPI spec
- Implement business logic with proper error handling
- Integrate with database using ORM/query builder
- Add resilience patterns (circuit breakers, retries, timeouts)
- Implement multi-tier caching strategy
- Add feature flags for gradual rollout
- Follow [TDD/BDD/DDD] methodology

Security:
- Parameterized queries (SQL injection prevention)
- Input validation and sanitization
- Authentication/authorization on all endpoints
- Rate limiting

Code Structure:
- Follow clean architecture / hexagonal architecture
- Dependency injection for testability
- Separation of concerns

Deliverables:
- Production-ready backend services
- Unit tests (>80% coverage)
- API documentation (Swagger/GraphQL schema)
- Database migrations
- Feature flag configuration
```

**Implementation Checklist**:

```markdown
## Backend Implementation Checklist

### Project Setup
- [ ] Initialize project structure
- [ ] Configure TypeScript/linting/formatting
- [ ] Set up environment variables
- [ ] Configure database connection
- [ ] Set up dependency injection container

### API Layer
- [ ] Implement REST/GraphQL endpoints
- [ ] Add request validation (Joi/Zod/class-validator)
- [ ] Add authentication middleware
- [ ] Add authorization checks
- [ ] Implement error handling middleware
- [ ] Add rate limiting
- [ ] Configure CORS
- [ ] Add security headers (Helmet)
- [ ] Generate API documentation

### Business Logic
- [ ] Implement core domain models
- [ ] Implement business rules and validations
- [ ] Add transaction management
- [ ] Implement event publishers (if event-driven)
- [ ] Add logging with correlation IDs
- [ ] Implement circuit breakers for external calls
- [ ] Add retry logic with exponential backoff

### Data Layer
- [ ] Define database schema
- [ ] Create migrations (up/down)
- [ ] Implement repositories/data access
- [ ] Add database indexes
- [ ] Configure connection pooling
- [ ] Add query performance monitoring

### Caching
- [ ] Implement Redis caching layer
- [ ] Define cache keys and TTLs
- [ ] Add cache invalidation logic
- [ ] Monitor cache hit rates

### Feature Flags
- [ ] Integrate feature flag service
- [ ] Add flag checks for new code paths
- [ ] Configure rollout percentages
- [ ] Add flag documentation

### Testing
- [ ] Unit tests for business logic
- [ ] Integration tests for repositories
- [ ] API endpoint tests
- [ ] Security tests (SQL injection, XSS, etc.)
- [ ] Performance tests

### Documentation
- [ ] Inline code comments
- [ ] README with setup instructions
- [ ] API documentation (Swagger/Postman)
- [ ] Architecture decision records (ADRs)
```

### Frontend Implementation

**Agent Prompt Template**:
```
Build frontend components for feature: [FEATURE_NAME].

Backend APIs:
[Paste API endpoints from Step 4]

Requirements:
- Build responsive UI using [React/Vue/Angular/Svelte]
- Implement state management ([Redux/Zustand/Pinia/etc.])
- Integrate with backend APIs
- Handle loading, error, and empty states
- Add form validation and error messages
- Implement analytics tracking
- Add feature flag integration for A/B testing
- Follow [TDD] methodology if specified

Accessibility:
- WCAG 2.1 Level AA compliance
- Keyboard navigation
- Screen reader support
- Color contrast ratios

Performance:
- Code splitting and lazy loading
- Image optimization
- Bundle size monitoring

Deliverables:
- Production-ready frontend components
- Unit tests for components
- Integration tests for user flows
- Storybook/component documentation
```

**Implementation Checklist**:

```markdown
## Frontend Implementation Checklist

### Project Setup
- [ ] Initialize project (Vite/CRA/Next.js/etc.)
- [ ] Configure TypeScript
- [ ] Set up linting/formatting (ESLint/Prettier)
- [ ] Configure build tools
- [ ] Set up environment variables

### Component Development
- [ ] Create component structure
- [ ] Implement UI components
- [ ] Add responsive breakpoints
- [ ] Implement loading states (skeletons)
- [ ] Implement error boundaries
- [ ] Add empty states

### State Management
- [ ] Set up state management library
- [ ] Define state shape
- [ ] Implement actions/reducers
- [ ] Add selectors
- [ ] Integrate with components

### API Integration
- [ ] Create API client (axios/fetch)
- [ ] Implement request/response interceptors
- [ ] Add authentication token handling
- [ ] Implement error handling
- [ ] Add retry logic
- [ ] Cache API responses (React Query/SWR)

### Forms & Validation
- [ ] Implement form components
- [ ] Add client-side validation
- [ ] Display field-level errors
- [ ] Handle form submission
- [ ] Add loading and success states

### Analytics
- [ ] Integrate analytics SDK (Segment/Amplitude)
- [ ] Track page views
- [ ] Track user interactions
- [ ] Track conversion events
- [ ] Add custom event properties

### Feature Flags
- [ ] Integrate feature flag SDK
- [ ] Wrap new features with flag checks
- [ ] Add flag-based A/B test variants
- [ ] Track flag exposure events

### Accessibility
- [ ] Semantic HTML elements
- [ ] ARIA labels and roles
- [ ] Keyboard navigation support
- [ ] Focus management
- [ ] Screen reader testing
- [ ] Color contrast validation

### Performance
- [ ] Code splitting per route
- [ ] Lazy load components
- [ ] Image optimization (next/image, lazy loading)
- [ ] Bundle size analysis
- [ ] Lighthouse score >90

### Testing
- [ ] Unit tests for utilities/hooks
- [ ] Component tests (React Testing Library)
- [ ] Integration tests for user flows
- [ ] Visual regression tests (Chromatic/Percy)
- [ ] Accessibility tests (axe/jest-axe)
```

### Data Pipeline

**Objective**: Build ETL/ELT processes, analytics events, and data quality monitoring.

**Implementation Guide**:

```markdown
## Data Pipeline Implementation

### Event Schema Design

```typescript
// analytics-events.ts
export interface BaseEvent {
  event_id: string;          // UUID
  user_id: string | null;    // User ID or null for anonymous
  session_id: string;        // Session identifier
  timestamp: Date;           // Event timestamp (ISO 8601)
  event_name: string;        // Event name (snake_case)
  properties: Record<string, any>;  // Event-specific properties
  context: {
    page_url: string;
    referrer: string;
    user_agent: string;
    ip_address: string;
    device_type: 'mobile' | 'tablet' | 'desktop';
    locale: string;
  };
}

export interface FeatureUsedEvent extends BaseEvent {
  event_name: 'feature_used';
  properties: {
    feature_name: string;
    action: string;           // e.g., 'created', 'updated', 'deleted'
    entity_id: string;
    entity_type: string;
    metadata: Record<string, any>;
  };
}
```

### Event Publishing

```typescript
// analytics-service.ts
class AnalyticsService {
  async track(event: BaseEvent): Promise<void> {
    // Validate event schema
    const validated = EventSchema.parse(event);

    // Enrich with server-side context
    const enriched = {
      ...validated,
      server_timestamp: new Date(),
      environment: process.env.NODE_ENV,
    };

    // Publish to message queue
    await this.messageQueue.publish('analytics.events', enriched);

    // Also send to analytics platform (Segment/Amplitude)
    await this.segment.track({
      userId: enriched.user_id,
      event: enriched.event_name,
      properties: enriched.properties,
      context: enriched.context,
    });
  }
}
```

### ETL Pipeline (Batch Processing)

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-eng',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'feature_analytics_etl',
    default_args=default_args,
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    catchup=False,
)

def extract_events(**context):
    """Extract events from message queue"""
    execution_date = context['execution_date']
    events = fetch_events_from_s3(execution_date)
    return events

def transform_events(events):
    """Transform and aggregate events"""
    df = pd.DataFrame(events)

    # Data quality checks
    assert df['event_id'].nunique() == len(df), "Duplicate event IDs found"
    assert df['timestamp'].notna().all(), "Missing timestamps"

    # Aggregations
    daily_metrics = df.groupby(['feature_name', 'action']).agg({
        'event_id': 'count',
        'user_id': 'nunique',
    }).rename(columns={
        'event_id': 'event_count',
        'user_id': 'unique_users',
    })

    return daily_metrics

def load_to_warehouse(metrics):
    """Load to data warehouse"""
    metrics.to_sql(
        'feature_metrics_daily',
        con=warehouse_engine,
        if_exists='append',
        method='multi',
    )

extract_task = PythonOperator(
    task_id='extract_events',
    python_callable=extract_events,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_events',
    python_callable=transform_events,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag,
)

extract_task >> transform_task >> load_task
```

### Data Quality Monitoring

```python
# data_quality_checks.py
from great_expectations import DataContext

def validate_feature_events():
    context = DataContext()

    # Define expectations
    suite = context.create_expectation_suite("feature_events_suite")

    # Schema validation
    suite.expect_column_to_exist("event_id")
    suite.expect_column_to_exist("timestamp")
    suite.expect_column_to_exist("event_name")

    # Data quality rules
    suite.expect_column_values_to_be_unique("event_id")
    suite.expect_column_values_to_not_be_null("timestamp")
    suite.expect_column_values_to_be_in_set(
        "event_name",
        value_set=["feature_used", "feature_viewed", "feature_error"]
    )

    # Run validation
    results = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[batch],
        run_id=f"validation_{datetime.now().isoformat()}"
    )

    if not results.success:
        send_alert("Data quality validation failed", results)
        raise DataQualityError("Validation failed")
```

### Analytics Dashboard Query

```sql
-- Daily feature usage metrics
CREATE MATERIALIZED VIEW feature_metrics_daily AS
SELECT
  DATE_TRUNC('day', timestamp) as date,
  feature_name,
  action,
  COUNT(DISTINCT event_id) as event_count,
  COUNT(DISTINCT user_id) as unique_users,
  COUNT(DISTINCT session_id) as unique_sessions,
  AVG(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))) as avg_session_duration_seconds
FROM analytics_events
WHERE event_name = 'feature_used'
GROUP BY 1, 2, 3;

-- Refresh schedule
CREATE INDEX idx_feature_metrics_date ON feature_metrics_daily(date DESC);
REFRESH MATERIALIZED VIEW CONCURRENTLY feature_metrics_daily;
```
```

---

## Phase 3: Testing & Quality Assurance

### Automated Testing

(Due to length constraints, I'll create condensed versions of the remaining sections. Let me know if you'd like me to expand any specific section.)

**Testing Strategy**:
- **Unit Tests**: Business logic, utilities, pure functions (>80% coverage)
- **Integration Tests**: API endpoints, database operations
- **E2E Tests**: Critical user journeys
- **Performance Tests**: Load testing, stress testing

**Testing Checklist**:
```markdown
- [ ] Unit tests for all business logic
- [ ] API integration tests
- [ ] E2E tests for critical user flows
- [ ] Security tests (OWASP, SQL injection, XSS)
- [ ] Performance/load tests
- [ ] Accessibility tests
- [ ] Cross-browser testing
- [ ] Mobile responsiveness testing
```

### Security Validation

**Security Testing Checklist**:
```markdown
- [ ] OWASP Top 10 vulnerability scanning
- [ ] Dependency vulnerability scanning (Snyk/Dependabot)
- [ ] SQL injection testing
- [ ] XSS testing
- [ ] CSRF protection verification
- [ ] Authentication/authorization testing
- [ ] Rate limiting verification
- [ ] Penetration testing (if applicable)
```

### Performance Optimization

**Performance Targets**:
- API response time: p95 < 200ms
- Frontend load time: < 2s
- Database query time: < 50ms
- Bundle size: < 200KB (gzipped)

**Optimization Checklist**:
```markdown
- [ ] Database query optimization (indexes, query plans)
- [ ] API response caching
- [ ] Frontend code splitting
- [ ] Image optimization
- [ ] CDN configuration
- [ ] Database connection pooling
- [ ] Load testing and capacity planning
```

---

## Phase 4: Deployment & Monitoring

### Deployment Pipeline

**CI/CD Pipeline Steps**:
1. **Build**: Compile, transpile, bundle
2. **Test**: Run automated test suites
3. **Security Scan**: SAST, dependency scanning
4. **Deploy to Staging**: Automated deployment
5. **Smoke Tests**: Verify critical functionality
6. **Deploy to Production**: Gradual rollout with feature flags
7. **Monitor**: Track metrics and errors

### Observability

**Observability Stack**:
- **Metrics**: Prometheus/DataDog/New Relic
- **Logs**: ELK Stack/Splunk/CloudWatch
- **Traces**: Jaeger/Zipkin/DataDog APM
- **Alerts**: PagerDuty/OpsGenie

**Key Metrics to Monitor**:
```markdown
- Request rate, error rate, duration (RED metrics)
- CPU, memory, disk usage (USE metrics)
- Business KPIs (conversion rate, feature adoption)
- Custom metrics (feature usage, user engagement)
```

### Documentation

**Documentation Deliverables**:
```markdown
- [ ] API documentation (Swagger/Postman)
- [ ] User guides
- [ ] Deployment runbook
- [ ] Troubleshooting guide
- [ ] Architecture documentation
- [ ] Onboarding guide for new developers
- [ ] Changelog/release notes
```

---

## Quick Reference: All Phase Steps

| Phase | Step | Agent | Estimated Time |
|-------|------|-------|----------------|
| 1 | Business Analysis | Manual | 1-2 days |
| 1 | Architecture Design | architect-review | 1-2 days |
| 1 | Risk Assessment | security-auditor | 1 day |
| 2 | Backend Implementation | backend-architect | 3-7 days |
| 2 | Frontend Implementation | frontend-developer | 3-7 days |
| 2 | Data Pipeline | Manual | 1-3 days |
| 3 | Automated Testing | test-automator | 2-4 days |
| 3 | Security Validation | security-auditor | 1-2 days |
| 3 | Performance Optimization | performance-engineer | 1-3 days |
| 4 | Deployment Pipeline | deployment-engineer | 1-2 days |
| 4 | Observability | observability-engineer | 1-2 days |
| 4 | Documentation | docs-architect | 1-2 days |

**Total Timeline**: 16-37 days (depending on complexity)
