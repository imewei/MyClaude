---
name: gitlab-ci-patterns
description: Build GitLab CI/CD pipelines with multi-stage workflows, caching strategies, distributed runners, and scalable automation patterns including Docker-in-Docker builds, Kubernetes deployments, and GitOps integration. Use when implementing GitLab CI/CD pipelines in .gitlab-ci.yml files, creating multi-stage pipelines with dependencies and DAG workflows, optimizing pipeline performance with intelligent caching for node_modules, pip packages, or Maven artifacts, setting up GitLab Runners (shared, specific, or group runners) on Kubernetes or Docker, implementing Docker builds with docker:dind or Kaniko for containerless builds, deploying applications to Kubernetes clusters with kubectl or Helm from GitLab, creating infrastructure deployment pipelines with Terraform and GitLab-managed state, implementing security scanning with SAST, DAST, Dependency Scanning, or Container Scanning templates, setting up automated testing workflows for unit tests, integration tests, and code quality checks, configuring merge request pipelines with approval rules and quality gates, implementing manual deployment jobs with when: manual for production control, creating dynamic child pipelines for monorepo or multi-project deployments, setting up GitLab Pages for static site deployment and documentation hosting, implementing auto-scaling runners with Docker Machine or Kubernetes executors, configuring cache and artifact management for build optimization, setting up scheduled pipelines for nightly builds or periodic tasks, implementing GitOps workflows with GitLab Agent for Kubernetes, integrating with external services using webhooks and API triggers, or managing CI/CD variables, secrets, and environment-specific configurations. Use this skill when working with .gitlab-ci.yml configuration, pipeline stages, jobs, scripts, rules, needs, artifacts, cache, or any GitLab CI/CD-specific features.
---

# GitLab CI Patterns

Comprehensive GitLab CI/CD pipeline patterns for automated testing, building, and deployment.

## When to use this skill

- When creating or modifying .gitlab-ci.yml pipeline configuration files
- When setting up multi-stage GitLab CI/CD pipelines (build, test, deploy stages)
- When implementing Docker builds using docker:dind service or Kaniko for secure builds
- When deploying applications to Kubernetes clusters using kubectl, Helm, or GitLab Agent
- When optimizing pipeline performance with caching strategies for dependencies
- When configuring GitLab Runners (shell, Docker, Kubernetes executors) for CI/CD jobs
- When implementing infrastructure-as-code deployments with Terraform or Ansible
- When setting up security scanning with GitLab's built-in SAST, DAST, and dependency scanning
- When creating merge request pipelines with automated testing and code quality checks
- When implementing manual approval gates for production deployments
- When setting up dynamic child pipelines for monorepo or complex multi-project workflows
- When deploying static websites or documentation to GitLab Pages
- When configuring auto-scaling runners with Docker Machine or Kubernetes for high-demand workloads
- When implementing GitOps workflows with GitLab Agent for Kubernetes
- When setting up scheduled pipelines for nightly builds, weekly reports, or periodic maintenance
- When managing artifacts and dependencies between pipeline jobs
- When creating reusable pipeline templates with extends or includes
- When implementing environment-specific deployments with GitLab Environments
- When integrating external services using webhooks, triggers, or API calls
- When managing CI/CD variables, secrets (masked/protected variables), or file variables
- When implementing review apps for feature branch testing
- When setting up compliance pipelines or audit trails for regulated environments
- When troubleshooting pipeline failures, runner issues, or optimization opportunities
- When migrating from Jenkins, GitHub Actions, or other CI/CD platforms to GitLab CI

## Basic Pipeline Structure

```yaml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

build:
  stage: build
  image: node:20
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 hour
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/

test:
  stage: test
  image: node:20
  script:
    - npm ci
    - npm run lint
    - npm test
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl apply -f k8s/
    - kubectl rollout status deployment/my-app
  only:
    - main
  environment:
    name: production
    url: https://app.example.com
```

## Docker Build and Push

```yaml
build-docker:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker build -t $CI_REGISTRY_IMAGE:latest .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main
    - tags
```

## Multi-Environment Deployment

```yaml
.deploy_template: &deploy_template
  image: bitnami/kubectl:latest
  before_script:
    - kubectl config set-cluster k8s --server="$KUBE_URL" --insecure-skip-tls-verify=true
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default

deploy:staging:
  <<: *deploy_template
  stage: deploy
  script:
    - kubectl apply -f k8s/ -n staging
    - kubectl rollout status deployment/my-app -n staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy:production:
  <<: *deploy_template
  stage: deploy
  script:
    - kubectl apply -f k8s/ -n production
    - kubectl rollout status deployment/my-app -n production
  environment:
    name: production
    url: https://app.example.com
  when: manual
  only:
    - main
```

## Terraform Pipeline

```yaml
stages:
  - validate
  - plan
  - apply

variables:
  TF_ROOT: ${CI_PROJECT_DIR}/terraform
  TF_VERSION: "1.6.0"

before_script:
  - cd ${TF_ROOT}
  - terraform --version

validate:
  stage: validate
  image: hashicorp/terraform:${TF_VERSION}
  script:
    - terraform init -backend=false
    - terraform validate
    - terraform fmt -check

plan:
  stage: plan
  image: hashicorp/terraform:${TF_VERSION}
  script:
    - terraform init
    - terraform plan -out=tfplan
  artifacts:
    paths:
      - ${TF_ROOT}/tfplan
    expire_in: 1 day

apply:
  stage: apply
  image: hashicorp/terraform:${TF_VERSION}
  script:
    - terraform init
    - terraform apply -auto-approve tfplan
  dependencies:
    - plan
  when: manual
  only:
    - main
```

## Security Scanning

```yaml
include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml

trivy-scan:
  stage: test
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 1 --severity HIGH,CRITICAL $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  allow_failure: true
```

## Caching Strategies

```yaml
# Cache node_modules
build:
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/
    policy: pull-push

# Global cache
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cache/
    - vendor/

# Separate cache per job
job1:
  cache:
    key: job1-cache
    paths:
      - build/

job2:
  cache:
    key: job2-cache
    paths:
      - dist/
```

## Dynamic Child Pipelines

```yaml
generate-pipeline:
  stage: build
  script:
    - python generate_pipeline.py > child-pipeline.yml
  artifacts:
    paths:
      - child-pipeline.yml

trigger-child:
  stage: deploy
  trigger:
    include:
      - artifact: child-pipeline.yml
        job: generate-pipeline
    strategy: depend
```

## Reference Files

- `assets/gitlab-ci.yml.template` - Complete pipeline template
- `references/pipeline-stages.md` - Stage organization patterns

## Best Practices

1. **Use specific image tags** (node:20, not node:latest)
2. **Cache dependencies** appropriately
3. **Use artifacts** for build outputs
4. **Implement manual gates** for production
5. **Use environments** for deployment tracking
6. **Enable merge request pipelines**
7. **Use pipeline schedules** for recurring jobs
8. **Implement security scanning**
9. **Use CI/CD variables** for secrets
10. **Monitor pipeline performance**

## Related Skills

- `github-actions-templates` - For GitHub Actions
- `deployment-pipeline-design` - For architecture
- `secrets-management` - For secrets handling
