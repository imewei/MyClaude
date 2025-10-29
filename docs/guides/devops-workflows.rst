DevOps Workflows
================

This guide demonstrates how to combine DevOps plugins for containerization, orchestration, and continuous deployment. Learn how to build modern CI/CD pipelines and manage infrastructure as code.

Overview
--------

DevOps workflows automate the software delivery process:

- **Containerization**: :term:`Docker` for packaging applications
- **Orchestration**: :term:`Kubernetes` for managing containers at scale
- **CI/CD**: :term:`CI/CD` pipelines for automated testing and deployment
- **Monitoring**: :term:`Observability` for production systems

Multi-Plugin Workflow: Containerized Deployment Pipeline
---------------------------------------------------------

This workflow combines :doc:`/plugins/full-stack-orchestration`, :doc:`/plugins/cicd-automation`, and :doc:`/plugins/observability-monitoring` to create an end-to-end deployment pipeline.

Prerequisites
~~~~~~~~~~~~~

Before starting, ensure you have:

- Docker installed and running
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- Git repository for your application
- Understanding of :term:`Container Orchestration`

See :term:`Docker`, :term:`Kubernetes`, and :term:`CI/CD` in the :doc:`/glossary` for background.

Step 1: Containerize Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/full-stack-orchestration` to create Docker containers:

.. code-block:: dockerfile

   # Dockerfile
   FROM python:3.12-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Expose port
   EXPOSE 8000

   # Run application
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build and test locally:

.. code-block:: bash

   # Build Docker image
   docker build -t myapp:latest .

   # Run container
   docker run -p 8000:8000 myapp:latest

   # Test application
   curl http://localhost:8000/health

Step 2: Create Kubernetes Manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define Kubernetes resources for deployment:

.. code-block:: yaml

   # k8s/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: myapp
     labels:
       app: myapp
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: myapp
     template:
       metadata:
         labels:
           app: myapp
       spec:
         containers:
         - name: myapp
           image: myapp:latest
           ports:
           - containerPort: 8000
           env:
           - name: DATABASE_URL
             valueFrom:
               secretKeyRef:
                 name: db-credentials
                 key: url
           resources:
             requests:
               memory: "256Mi"
               cpu: "100m"
             limits:
               memory: "512Mi"
               cpu: "500m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8000
             initialDelaySeconds: 10
             periodSeconds: 5

Create service and ingress:

.. code-block:: yaml

   # k8s/service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: myapp
   spec:
     selector:
       app: myapp
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer

   ---
   # k8s/ingress.yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: myapp
     annotations:
       cert-manager.io/cluster-issuer: "letsencrypt-prod"
   spec:
     tls:
     - hosts:
       - myapp.example.com
       secretName: myapp-tls
     rules:
     - host: myapp.example.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: myapp
               port:
                 number: 80

Step 3: Set Up CI/CD Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/cicd-automation` to automate deployments:

.. code-block:: yaml

   # .github/workflows/deploy.yml
   name: Build and Deploy

   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   env:
     REGISTRY: ghcr.io
     IMAGE_NAME: ${{ github.repository }}

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.12'

         - name: Install dependencies
           run: |
             pip install -r requirements.txt
             pip install pytest pytest-cov

         - name: Run tests
           run: pytest --cov=app --cov-report=xml

         - name: Upload coverage
           uses: codecov/codecov-action@v3

     build:
       needs: test
       runs-on: ubuntu-latest
       permissions:
         contents: read
         packages: write
       steps:
         - uses: actions/checkout@v4

         - name: Log in to Container Registry
           uses: docker/login-action@v3
           with:
             registry: ${{ env.REGISTRY }}
             username: ${{ github.actor }}
             password: ${{ secrets.GITHUB_TOKEN }}

         - name: Extract metadata
           id: meta
           uses: docker/metadata-action@v5
           with:
             images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}

         - name: Build and push
           uses: docker/build-push-action@v5
           with:
             context: .
             push: true
             tags: ${{ steps.meta.outputs.tags }}
             labels: ${{ steps.meta.outputs.labels }}

     deploy:
       needs: build
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main'
       steps:
         - uses: actions/checkout@v4

         - name: Configure kubectl
           uses: azure/k8s-set-context@v3
           with:
             method: kubeconfig
             kubeconfig: ${{ secrets.KUBE_CONFIG }}

         - name: Deploy to Kubernetes
           run: |
             kubectl apply -f k8s/
             kubectl rollout status deployment/myapp

Step 4: Add Monitoring and Observability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/observability-monitoring` for production monitoring:

.. code-block:: yaml

   # k8s/monitoring.yaml
   apiVersion: v1
   kind: ServiceMonitor
   metadata:
     name: myapp
     labels:
       app: myapp
   spec:
     selector:
       matchLabels:
         app: myapp
     endpoints:
     - port: metrics
       interval: 30s
       path: /metrics

Add application instrumentation:

.. code-block:: python

   # app/monitoring.py
   from prometheus_client import Counter, Histogram, generate_latest
   from fastapi import Response

   # Define metrics
   request_count = Counter(
       'app_requests_total',
       'Total request count',
       ['method', 'endpoint', 'status']
   )

   request_duration = Histogram(
       'app_request_duration_seconds',
       'Request duration in seconds',
       ['method', 'endpoint']
   )

   # Metrics endpoint
   @app.get("/metrics")
   async def metrics():
       return Response(
           content=generate_latest(),
           media_type="text/plain"
       )

Expected Outcomes
~~~~~~~~~~~~~~~~~

After completing this workflow, you will have:

- Containerized application with Docker
- Kubernetes deployment with 3 replicas
- Automated CI/CD pipeline with tests
- Production monitoring with Prometheus
- Automated rollout and rollback capabilities

Workflow: Infrastructure as Code
---------------------------------

Manage cloud infrastructure with :doc:`/plugins/cicd-automation` principles.

Prerequisites
~~~~~~~~~~~~~

- Cloud provider account (AWS, GCP, Azure)
- Understanding of :term:`Terraform`
- Version control for infrastructure code

Step 1: Define Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: hcl

   # main.tf
   terraform {
     required_providers {
       aws = {
         source  = "hashicorp/aws"
         version = "~> 5.0"
       }
     }
   }

   provider "aws" {
     region = "us-west-2"
   }

   # EKS Cluster
   module "eks" {
     source  = "terraform-aws-modules/eks/aws"
     version = "19.0"

     cluster_name    = "myapp-cluster"
     cluster_version = "1.27"

     vpc_id     = module.vpc.vpc_id
     subnet_ids = module.vpc.private_subnets

     eks_managed_node_groups = {
       general = {
         desired_size = 2
         min_size     = 1
         max_size     = 5

         instance_types = ["t3.medium"]
       }
     }
   }

   # VPC
   module "vpc" {
     source  = "terraform-aws-modules/vpc/aws"
     version = "5.0"

     name = "myapp-vpc"
     cidr = "10.0.0.0/16"

     azs             = ["us-west-2a", "us-west-2b", "us-west-2c"]
     private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
     public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

     enable_nat_gateway = true
     single_nat_gateway = true
   }

Step 2: Apply Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Initialize Terraform
   terraform init

   # Plan changes
   terraform plan -out=tfplan

   # Apply changes
   terraform apply tfplan

   # Get cluster credentials
   aws eks update-kubeconfig --name myapp-cluster --region us-west-2

Step 3: Automate Infrastructure Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # .github/workflows/terraform.yml
   name: Terraform CI/CD

   on:
     push:
       branches: [main]
       paths:
         - 'terraform/**'

   jobs:
     terraform:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - name: Setup Terraform
           uses: hashicorp/setup-terraform@v3

         - name: Terraform Init
           run: terraform init

         - name: Terraform Format
           run: terraform fmt -check

         - name: Terraform Validate
           run: terraform validate

         - name: Terraform Plan
           run: terraform plan

         - name: Terraform Apply
           if: github.ref == 'refs/heads/main'
           run: terraform apply -auto-approve

Integration Patterns
--------------------

Common DevOps Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Full CI/CD Pipeline**
   :doc:`/plugins/cicd-automation` + :doc:`/plugins/full-stack-orchestration` + :doc:`/plugins/observability-monitoring`

   Complete automation from code commit to production deployment with monitoring.

**GitOps Workflow**
   :doc:`/plugins/git-pr-workflows` + :doc:`/plugins/cicd-automation` + :doc:`/plugins/full-stack-orchestration`

   Infrastructure and applications managed through Git pull requests.

**Multi-Environment Deployment**
   :doc:`/plugins/full-stack-orchestration` + :doc:`/plugins/cicd-automation` + :doc:`/plugins/quality-engineering`

   Automated promotion through dev, staging, and production environments.

Best Practices
~~~~~~~~~~~~~~

1. **Immutable Infrastructure**: Rebuild rather than modify running systems
2. **Infrastructure as Code**: Version control all infrastructure definitions
3. **Blue-Green Deployments**: Minimize downtime with parallel environments
4. **Automated Testing**: Test infrastructure changes before production
5. **Monitoring**: Implement comprehensive observability from day one
6. **Security**: Scan containers for vulnerabilities, use secrets management

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Container Fails to Start**
   - Check container logs: `kubectl logs <pod-name>`
   - Verify resource limits are sufficient
   - Ensure environment variables are set correctly
   - Check liveness/readiness probes

**Deployment Rollout Stuck**
   - Check pod status: `kubectl get pods`
   - Review events: `kubectl describe deployment <name>`
   - Verify image pull secrets
   - Check resource quotas

**CI/CD Pipeline Failures**
   - Review pipeline logs
   - Verify credentials and secrets
   - Check network connectivity to cluster
   - Ensure kubectl version compatibility

Next Steps
----------

- Explore :doc:`infrastructure-workflows` for cloud architecture
- See :doc:`/plugins/observability-monitoring` for advanced monitoring
- Review :doc:`/plugins/quality-engineering` for security testing
- Check :doc:`/categories/devops` for all DevOps plugins

Additional Resources
--------------------

- `Kubernetes Documentation <https://kubernetes.io/docs/>`_
- `Docker Best Practices <https://docs.docker.com/develop/dev-best-practices/>`_
- `GitHub Actions Guide <https://docs.github.com/en/actions>`_
- `Terraform Tutorials <https://developer.hashicorp.com/terraform/tutorials>`_

See Also
--------

- :doc:`development-workflows` - Application development patterns
- :doc:`scientific-workflows` - HPC deployment strategies
- :doc:`/integration-map` - Complete plugin compatibility matrix
