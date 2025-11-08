# Workflow Orchestration Patterns

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

Advanced workflow orchestration patterns for complex CI/CD scenarios including sequential/parallel execution, error handling, retry logic, and event-driven coordination.

---

## WorkflowOrchestrator Class

Complete TypeScript implementation for flexible workflow orchestration:

```typescript
import { EventEmitter } from 'events';

interface WorkflowStep {
  name: string;
  type: 'parallel' | 'sequential';
  steps?: WorkflowStep[];
  action?: () => Promise<any>;
  retries?: number;
  timeout?: number;  // milliseconds
  condition?: () => boolean | Promise<boolean>;
  onError?: 'fail' | 'continue' | 'retry';
}

interface WorkflowResult {
  success: boolean;
  duration: number;
  steps: StepResult[];
  errors: Error[];
}

interface StepResult {
  name: string;
  path: string;
  status: 'success' | 'failed' | 'skipped' | 'timeout';
  duration: number;
  result?: any;
  error?: Error;
  retries?: number;
}

class WorkflowOrchestrator extends EventEmitter {
  private startTime: number = 0;

  /**
   * Execute a workflow with orchestrated steps
   */
  async execute(workflow: WorkflowStep): Promise<WorkflowResult> {
    this.startTime = Date.now();
    const result: WorkflowResult = {
      success: true,
      duration: 0,
      steps: [],
      errors: []
    };

    this.emit('workflow:start', { workflow: workflow.name });

    try {
      await this.executeStep(workflow, result, workflow.name);
    } catch (error) {
      result.success = false;
      result.errors.push(error as Error);
      this.emit('workflow:failed', { error });
    } finally {
      result.duration = Date.now() - this.startTime;
      this.emit('workflow:complete', { result });
    }

    return result;
  }

  /**
   * Execute a single step (may contain nested steps)
   */
  private async executeStep(
    step: WorkflowStep,
    result: WorkflowResult,
    parentPath: string
  ): Promise<void> {
    const stepPath = `${parentPath}`;

    // Check condition
    if (step.condition) {
      const shouldExecute = await Promise.resolve(step.condition());
      if (!shouldExecute) {
        const stepResult: StepResult = {
          name: step.name,
          path: stepPath,
          status: 'skipped',
          duration: 0
        };
        result.steps.push(stepResult);
        this.emit('step:skipped', { step: step.name, path: stepPath });
        return;
      }
    }

    const stepStartTime = Date.now();
    this.emit('step:start', { step: step.name, path: stepPath });

    try {
      let stepOutput: any;

      // Execute based on type
      if (step.type === 'parallel' && step.steps) {
        stepOutput = await this.executeParallel(step.steps, result, stepPath);
      } else if (step.type === 'sequential' && step.steps) {
        stepOutput = await this.executeSequential(step.steps, result, stepPath);
      } else if (step.action) {
        stepOutput = await this.executeAction(step, stepPath);
      }

      const stepResult: StepResult = {
        name: step.name,
        path: stepPath,
        status: 'success',
        duration: Date.now() - stepStartTime,
        result: stepOutput
      };
      result.steps.push(stepResult);
      this.emit('step:complete', { step: step.name, path: stepPath, duration: stepResult.duration });

    } catch (error) {
      const stepResult: StepResult = {
        name: step.name,
        path: stepPath,
        status: 'failed',
        duration: Date.now() - stepStartTime,
        error: error as Error
      };
      result.steps.push(stepResult);
      result.errors.push(error as Error);
      this.emit('step:failed', { step: step.name, path: stepPath, error });

      // Handle error based on strategy
      if (step.onError === 'fail' || !step.onError) {
        throw error;
      } else if (step.onError === 'continue') {
        // Continue to next step
        return;
      }
    }
  }

  /**
   * Execute an action with retry and timeout
   */
  private async executeAction(step: WorkflowStep, stepPath: string): Promise<any> {
    const maxRetries = step.retries || 0;
    const timeout = step.timeout || 300000; // 5 minutes default

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          const backoffDelay = Math.min(1000 * Math.pow(2, attempt - 1), 30000);
          this.emit('step:retry', { step: step.name, attempt, delay: backoffDelay });
          await this.sleep(backoffDelay);
        }

        // Execute with timeout
        const result = await this.withTimeout(
          step.action!(),
          timeout,
          `Step ${step.name} exceeded timeout of ${timeout}ms`
        );

        return result;

      } catch (error) {
        lastError = error as Error;
        if (attempt < maxRetries) {
          continue;
        }
      }
    }

    throw lastError;
  }

  /**
   * Execute steps in parallel
   */
  private async executeParallel(
    steps: WorkflowStep[],
    result: WorkflowResult,
    parentPath: string
  ): Promise<any[]> {
    const promises = steps.map((step, index) =>
      this.executeStep(step, result, `${parentPath}.parallel[${index}]`)
    );

    return Promise.all(promises);
  }

  /**
   * Execute steps sequentially
   */
  private async executeSequential(
    steps: WorkflowStep[],
    result: WorkflowResult,
    parentPath: string
  ): Promise<any[]> {
    const results: any[] = [];

    for (let i = 0; i < steps.length; i++) {
      const stepResult = await this.executeStep(
        steps[i],
        result,
        `${parentPath}.sequential[${i}]`
      );
      results.push(stepResult);
    }

    return results;
  }

  /**
   * Execute promise with timeout
   */
  private withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    timeoutMessage: string
  ): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) =>
        setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs)
      )
    ]);
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export { WorkflowOrchestrator, WorkflowStep, WorkflowResult, StepResult };
```

---

## Complex Workflow Examples

### 1. Deployment Workflow

Three-phase deployment with pre-deployment checks, deployment, and post-deployment verification:

```typescript
import { WorkflowOrchestrator } from './orchestrator';

async function deploymentWorkflow() {
  const orchestrator = new WorkflowOrchestrator();

  // Event listeners
  orchestrator.on('workflow:start', (data) => console.log('🚀 Starting deployment:', data));
  orchestrator.on('step:complete', (data) => console.log('✅', data.step, `(${data.duration}ms)`));
  orchestrator.on('step:failed', (data) => console.error('❌', data.step, data.error.message));

  const workflow: WorkflowStep = {
    name: 'Production Deployment',
    type: 'sequential',
    steps: [
      // PHASE 1: Pre-Deployment
      {
        name: 'Pre-Deployment Checks',
        type: 'parallel',
        steps: [
          {
            name: 'Run Tests',
            type: 'sequential',
            action: async () => {
              // Run test suite
              console.log('  Running test suite...');
              await sleep(2000);
              return { tests: 150, passed: 150 };
            }
          },
          {
            name: 'Security Scan',
            type: 'sequential',
            action: async () => {
              console.log('  Running security scan...');
              await sleep(1500);
              return { vulnerabilities: 0 };
            }
          },
          {
            name: 'Database Backup',
            type: 'sequential',
            action: async () => {
              console.log('  Creating database backup...');
              await sleep(3000);
              return { backupId: 'backup-2025-11-06' };
            },
            retries: 2,
            timeout: 10000
          }
        ]
      },

      // PHASE 2: Deployment
      {
        name: 'Deploy Application',
        type: 'sequential',
        steps: [
          {
            name: 'Build Docker Image',
            type: 'sequential',
            action: async () => {
              console.log('  Building Docker image...');
              await sleep(5000);
              return { image: 'myapp:v1.2.3' };
            },
            timeout: 30000
          },
          {
            name: 'Push to Registry',
            type: 'sequential',
            action: async () => {
              console.log('  Pushing to registry...');
              await sleep(3000);
              return { digest: 'sha256:abc123...' };
            },
            retries: 3
          },
          {
            name: 'Update Kubernetes',
            type: 'sequential',
            action: async () => {
              console.log('  Updating Kubernetes deployment...');
              await sleep(4000);
              return { replicas: 3, ready: 3 };
            },
            onError: 'fail'
          }
        ]
      },

      // PHASE 3: Post-Deployment
      {
        name: 'Post-Deployment Verification',
        type: 'parallel',
        steps: [
          {
            name: 'Health Check',
            type: 'sequential',
            action: async () => {
              console.log('  Running health checks...');
              await sleep(2000);
              return { status: 'healthy' };
            },
            retries: 5,
            timeout: 10000
          },
          {
            name: 'Smoke Tests',
            type: 'sequential',
            action: async () => {
              console.log('  Running smoke tests...');
              await sleep(3000);
              return { tests: 10, passed: 10 };
            },
            retries: 3
          },
          {
            name: 'Notify Stakeholders',
            type: 'sequential',
            action: async () => {
              console.log('  Sending notifications...');
              await sleep(1000);
              return { notified: ['team@example.com', 'slack'] };
            },
            onError: 'continue'  // Don't fail deployment if notification fails
          }
        ]
      }
    ]
  };

  const result = await orchestrator.execute(workflow);
  console.log('\n📊 Deployment Summary:');
  console.log(`  Success: ${result.success}`);
  console.log(`  Duration: ${result.duration}ms`);
  console.log(`  Steps Executed: ${result.steps.length}`);
  console.log(`  Errors: ${result.errors.length}`);

  return result;
}

// Helper function
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
```

### 2. Data Pipeline Workflow

ETL pipeline with validation and error handling:

```typescript
const dataPipeline: WorkflowStep = {
  name: 'ETL Data Pipeline',
  type: 'sequential',
  steps: [
    // Extract
    {
      name: 'Extract Data',
      type: 'parallel',
      steps: [
        {
          name: 'Extract from API',
          type: 'sequential',
          action: async () => {
            // Fetch data from external API
            return { records: 10000, source: 'api' };
          },
          retries: 3,
          timeout: 60000
        },
        {
          name: 'Extract from Database',
          type: 'sequential',
          action: async () => {
            // Query database
            return { records: 25000, source: 'database' };
          },
          retries: 2
        },
        {
          name: 'Extract from S3',
          type: 'sequential',
          action: async () => {
            // Download S3 files
            return { records: 5000, source: 's3' };
          },
          retries: 3
        }
      ]
    },

    // Transform
    {
      name: 'Transform Data',
      type: 'sequential',
      steps: [
        {
          name: 'Data Validation',
          type: 'sequential',
          action: async () => {
            // Validate data schema
            return { valid: 39500, invalid: 500 };
          }
        },
        {
          name: 'Data Cleaning',
          type: 'sequential',
          action: async () => {
            // Clean and normalize data
            return { cleaned: 39500 };
          }
        },
        {
          name: 'Data Enrichment',
          type: 'sequential',
          action: async () => {
            // Enrich with additional data
            return { enriched: 39500 };
          },
          condition: async () => {
            // Only enrich if needed
            return true;
          }
        }
      ]
    },

    // Load
    {
      name: 'Load Data',
      type: 'parallel',
      steps: [
        {
          name: 'Load to Data Warehouse',
          type: 'sequential',
          action: async () => {
            // Insert into data warehouse
            return { inserted: 39500, table: 'fact_data' };
          },
          timeout: 120000
        },
        {
          name: 'Update Search Index',
          type: 'sequential',
          action: async () => {
            // Update Elasticsearch
            return { indexed: 39500 };
          },
          onError: 'continue'  // Don't fail pipeline if indexing fails
        },
        {
          name: 'Generate Report',
          type: 'sequential',
          action: async () => {
            // Create summary report
            return { report: 'pipeline-report-2025-11-06.pdf' };
          },
          onError: 'continue'
        }
      ]
    }
  ]
};
```

### 3. Microservices Release Workflow

Parallel service builds with sequential deployment:

```typescript
const microservicesRelease: WorkflowStep = {
  name: 'Microservices Release',
  type: 'sequential',
  steps: [
    // Build all services in parallel
    {
      name: 'Build Services',
      type: 'parallel',
      steps: [
        {
          name: 'Build API Service',
          type: 'sequential',
          action: async () => {
            return { service: 'api', image: 'api:v2.1.0' };
          },
          timeout: 180000
        },
        {
          name: 'Build Worker Service',
          type: 'sequential',
          action: async () => {
            return { service: 'worker', image: 'worker:v2.1.0' };
          },
          timeout: 180000
        },
        {
          name: 'Build Scheduler Service',
          type: 'sequential',
          action: async () => {
            return { service: 'scheduler', image: 'scheduler:v2.1.0' };
          },
          timeout: 180000
        }
      ]
    },

    // Deploy services sequentially (to avoid overwhelming infrastructure)
    {
      name: 'Deploy Services',
      type: 'sequential',
      steps: [
        {
          name: 'Deploy API Service',
          type: 'sequential',
          action: async () => {
            // Rolling update API service
            return { deployed: 'api', replicas: 5 };
          },
          retries: 2
        },
        {
          name: 'Deploy Worker Service',
          type: 'sequential',
          action: async () => {
            // Deploy worker service
            return { deployed: 'worker', replicas: 3 };
          },
          retries: 2
        },
        {
          name: 'Deploy Scheduler Service',
          type: 'sequential',
          action: async () => {
            // Deploy scheduler (single instance)
            return { deployed: 'scheduler', replicas: 1 };
          },
          retries: 2
        }
      ]
    },

    // Verify all services
    {
      name: 'Verify Deployment',
      type: 'parallel',
      steps: [
        {
          name: 'API Health Check',
          type: 'sequential',
          action: async () => {
            return { service: 'api', healthy: true };
          },
          retries: 10,
          timeout: 30000
        },
        {
          name: 'Worker Health Check',
          type: 'sequential',
          action: async () => {
            return { service: 'worker', healthy: true };
          },
          retries: 10,
          timeout: 30000
        },
        {
          name: 'Scheduler Health Check',
          type: 'sequential',
          action: async () => {
            return { service: 'scheduler', healthy: true };
          },
          retries: 10,
          timeout: 30000
        }
      ]
    }
  ]
};
```

### 4. Blue-Green Deployment Workflow

Zero-downtime deployment with traffic switching:

```typescript
const blueGreenDeployment: WorkflowStep = {
  name: 'Blue-Green Deployment',
  type: 'sequential',
  steps: [
    // Prepare Green Environment
    {
      name: 'Prepare Green Environment',
      type: 'sequential',
      steps: [
        {
          name: 'Deploy to Green',
          type: 'sequential',
          action: async () => {
            // Deploy new version to green environment
            return { environment: 'green', version: 'v1.0.2' };
          },
          timeout: 300000
        },
        {
          name: 'Warm Up Green',
          type: 'sequential',
          action: async () => {
            // Send warm-up traffic to green
            return { requests: 1000, success: 1000 };
          }
        }
      ]
    },

    // Verify Green Environment
    {
      name: 'Verify Green Environment',
      type: 'parallel',
      steps: [
        {
          name: 'Health Checks',
          type: 'sequential',
          action: async () => {
            return { healthy: true };
          },
          retries: 5
        },
        {
          name: 'Integration Tests',
          type: 'sequential',
          action: async () => {
            return { tests: 50, passed: 50 };
          }
        },
        {
          name: 'Performance Tests',
          type: 'sequential',
          action: async () => {
            return { p99: 150, target: 200, passed: true };
          }
        }
      ]
    },

    // Switch Traffic
    {
      name: 'Switch Traffic to Green',
      type: 'sequential',
      steps: [
        {
          name: 'Route 10% Traffic',
          type: 'sequential',
          action: async () => {
            return { green: 10, blue: 90 };
          }
        },
        {
          name: 'Monitor Metrics',
          type: 'sequential',
          action: async () => {
            return { errors: 0, latency: 145 };
          },
          timeout: 60000
        },
        {
          name: 'Route 50% Traffic',
          type: 'sequential',
          action: async () => {
            return { green: 50, blue: 50 };
          },
          condition: async () => {
            // Only proceed if 10% went well
            return true;
          }
        },
        {
          name: 'Route 100% Traffic',
          type: 'sequential',
          action: async () => {
            return { green: 100, blue: 0 };
          }
        }
      ]
    },

    // Cleanup Blue Environment
    {
      name: 'Cleanup Blue Environment',
      type: 'sequential',
      steps: [
        {
          name: 'Wait for Cooldown',
          type: 'sequential',
          action: async () => {
            await sleep(300000);  // Wait 5 minutes
            return { cooldown: 'complete' };
          }
        },
        {
          name: 'Destroy Blue',
          type: 'sequential',
          action: async () => {
            // Destroy blue environment
            return { destroyed: 'blue' };
          },
          onError: 'continue'  // Don't fail if cleanup fails
        }
      ]
    }
  ]
};
```

---

## Event-Driven Patterns

### Monitoring Integration

```typescript
// DataDog metrics
orchestrator.on('step:complete', (data) => {
  dogstatsd.timing('workflow.step.duration', data.duration, [`step:${data.step}`]);
  dogstatsd.increment('workflow.step.complete', [`step:${data.step}`]);
});

orchestrator.on('step:failed', (data) => {
  dogstatsd.increment('workflow.step.failed', [`step:${data.step}`]);
});

// Prometheus metrics
orchestrator.on('workflow:complete', (data) => {
  workflowDuration.labels(data.result.success ? 'success' : 'failure').observe(data.result.duration);
  workflowSteps.set(data.result.steps.length);
});
```

### Alerting

```typescript
// Slack notifications
orchestrator.on('workflow:failed', async (data) => {
  await sendSlackAlert({
    channel: '#deployments',
    text: `🚨 Workflow failed: ${data.error.message}`,
    color: 'danger'
  });
});

orchestrator.on('step:retry', (data) => {
  if (data.attempt > 2) {
    sendSlackAlert({
      channel: '#deployments',
      text: `⚠️ Step ${data.step} retrying (attempt ${data.attempt})`,
      color: 'warning'
    });
  }
});
```

---

## Best Practices

### 1. Idempotency
Ensure steps can be safely retried:
```typescript
{
  name: 'Deploy to Kubernetes',
  action: async () => {
    // Use declarative approach (idempotent)
    await kubectl.apply({ manifest: 'deployment.yaml' });
  },
  retries: 3
}
```

### 2. Timeouts
Always set appropriate timeouts:
```typescript
{
  name: 'External API Call',
  action: async () => {
    // Long-running operation
  },
  timeout: 60000,  // 1 minute
  retries: 2
}
```

### 3. Error Handling Strategy
Choose appropriate error handling:
```typescript
{
  name: 'Send Notification',
  action: async () => {
    // Non-critical operation
  },
  onError: 'continue'  // Don't fail workflow
}
```

---

For related documentation, see:
- [github-actions-reference.md](github-actions-reference.md)
- [gitlab-ci-reference.md](gitlab-ci-reference.md)
- [workflow-analysis-framework.md](workflow-analysis-framework.md)
