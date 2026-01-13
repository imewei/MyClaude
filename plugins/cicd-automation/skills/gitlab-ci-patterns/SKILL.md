---
name: gitlab-ci-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: GitLab CI/CD
description: Build GitLab CI/CD pipelines with multi-stage workflows, caching, Docker builds, Kubernetes deployments, and security scanning. Use when creating .gitlab-ci.yml pipelines, setting up runners, implementing Terraform/IaC, or configuring GitOps workflows.
---

# GitLab CI Patterns

Production-ready GitLab CI/CD pipeline patterns and automation.

---

## Pipeline Stages

| Stage | Purpose | Jobs |
|-------|---------|------|
| build | Compile, containerize | npm build, docker build |
| test | Validate | unit, integration, lint |
| deploy | Release | staging, production |

---

## Basic Pipeline

```yaml
stages: [build, test, deploy]

variables:
  DOCKER_TLS_CERTDIR: "/certs"

build:
  stage: build
  image: node:20
  script:
    - npm ci && npm run build
  artifacts:
    paths: [dist/]
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths: [node_modules/]

test:
  stage: test
  image: node:20
  script:
    - npm ci && npm run lint && npm test

deploy:
  stage: deploy
  script: kubectl apply -f k8s/
  only: [main]
  environment:
    name: production
    url: https://app.example.com
```

---

## Docker Build

```yaml
build-docker:
  stage: build
  image: docker:24
  services: [docker:24-dind]
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only: [main, tags]
```

---

## Multi-Environment Deployment

```yaml
.deploy_template: &deploy_template
  image: bitnami/kubectl:latest
  before_script:
    - kubectl config set-cluster k8s --server="$KUBE_URL"
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default

deploy:staging:
  <<: *deploy_template
  stage: deploy
  script: kubectl apply -f k8s/ -n staging
  environment: { name: staging }
  only: [develop]

deploy:production:
  <<: *deploy_template
  stage: deploy
  script: kubectl apply -f k8s/ -n production
  environment: { name: production }
  when: manual
  only: [main]
```

---

## Terraform Pipeline

```yaml
stages: [validate, plan, apply]

validate:
  stage: validate
  image: hashicorp/terraform:1.6
  script:
    - terraform init -backend=false
    - terraform validate && terraform fmt -check

plan:
  stage: plan
  image: hashicorp/terraform:1.6
  script: terraform init && terraform plan -out=tfplan
  artifacts:
    paths: [tfplan]

apply:
  stage: apply
  image: hashicorp/terraform:1.6
  script: terraform apply -auto-approve tfplan
  dependencies: [plan]
  when: manual
  only: [main]
```

---

## Security Scanning

```yaml
include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml

trivy-scan:
  stage: test
  image: aquasec/trivy:latest
  script: trivy image --exit-code 1 --severity HIGH,CRITICAL $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  allow_failure: true
```

---

## Caching Strategies

| Strategy | Key | Use Case |
|----------|-----|----------|
| Branch-based | `${CI_COMMIT_REF_SLUG}` | Dependencies per branch |
| Global | `global-cache` | Shared across branches |
| Job-specific | `job-name-cache` | Isolated per job |

```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths: [node_modules/, .cache/]
  policy: pull-push
```

---

## Dynamic Child Pipelines

```yaml
generate-pipeline:
  stage: build
  script: python generate_pipeline.py > child-pipeline.yml
  artifacts:
    paths: [child-pipeline.yml]

trigger-child:
  stage: deploy
  trigger:
    include:
      - artifact: child-pipeline.yml
        job: generate-pipeline
    strategy: depend
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Specific image tags | `node:20` not `node:latest` |
| Cache dependencies | Reduce install time |
| Artifacts for builds | Pass between stages |
| Manual production gates | `when: manual` |
| Environments for tracking | Link to URLs |
| Security scanning | Include templates |
| CI/CD variables | Masked for secrets |

---

## Checklist

- [ ] Stages defined (build, test, deploy)
- [ ] Caching configured for dependencies
- [ ] Artifacts for build outputs
- [ ] Manual approval for production
- [ ] Environments with URLs
- [ ] Security scanning enabled
- [ ] Secrets in masked variables
- [ ] Image tags pinned

---

**Version**: 1.0.5
