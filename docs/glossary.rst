Glossary
========

This glossary provides definitions for technical terms used throughout the Plugin Marketplace documentation. Terms are organized alphabetically and include context about their usage in various plugins.

.. glossary::
   :sorted:

   Ansible
      An open-source automation tool for configuration management, application deployment, and task automation. Ansible uses YAML-based playbooks to define infrastructure as code and can manage both cloud and on-premises systems without requiring agents on target machines.

      Related plugins: :doc:`/plugins/cicd-automation`

   BDD
      Behavior-Driven Development. A software development methodology that extends Test-Driven Development (TDD) by writing test cases in natural language that non-programmers can read. BDD focuses on the behavior of an application from the end user's perspective.

      Related plugins: :doc:`/plugins/unit-testing`, :doc:`/plugins/quality-engineering`

   CI/CD
      Continuous Integration/Continuous Deployment. A software development practice where code changes are automatically built, tested, and deployed to production. CI/CD pipelines automate the software release process, reducing manual errors and accelerating delivery.

      Related plugins: :doc:`/plugins/cicd-automation`, :doc:`/plugins/git-pr-workflows`

   Cloud Infrastructure
      The collection of hardware and software components needed to enable cloud computing, including servers, storage, networking, and virtualization technologies. Cloud infrastructure can be public, private, or hybrid.

      Related plugins: :doc:`/plugins/observability-monitoring`, :doc:`/plugins/hpc-computing`

   Container Orchestration
      The automated management, deployment, scaling, and networking of containerized applications. Container orchestration platforms like Kubernetes handle the lifecycle of containers across clusters of machines.

      Related plugins: :doc:`/plugins/full-stack-orchestration`, :doc:`/plugins/cicd-automation`

   Docker
      A platform for developing, shipping, and running applications in containers. Docker packages applications with their dependencies into standardized units called containers, ensuring consistency across different environments.

      Related plugins: :doc:`/plugins/full-stack-orchestration`, :doc:`/plugins/cicd-automation`

   GPU Computing
      The use of Graphics Processing Units (GPUs) for general-purpose computing tasks beyond graphics rendering. GPUs excel at parallel processing and are widely used for machine learning, scientific simulations, and data analysis.

      Related plugins: :doc:`/plugins/deep-learning`, :doc:`/plugins/hpc-computing`, :doc:`/plugins/jax-implementation`

   HPC
      High-Performance Computing. The use of supercomputers and parallel processing techniques to solve complex computational problems. HPC systems aggregate computing power to deliver performance far beyond typical desktop computers.

      Related plugins: :doc:`/plugins/hpc-computing`, :doc:`/plugins/julia-development`, :doc:`/plugins/molecular-simulation`

   JAX
      A Python library for high-performance numerical computing and machine learning research. JAX combines NumPy-like syntax with automatic differentiation, GPU/TPU acceleration, and just-in-time compilation via XLA.

      Related plugins: :doc:`/plugins/jax-implementation`, :doc:`/plugins/deep-learning`

   Kubernetes
      An open-source container orchestration platform for automating deployment, scaling, and management of containerized applications. Kubernetes groups containers into logical units for easy management and discovery.

      Related plugins: :doc:`/plugins/full-stack-orchestration`, :doc:`/plugins/cicd-automation`

   MCMC
      Markov Chain Monte Carlo. A class of algorithms for sampling from probability distributions using random walks. MCMC methods are fundamental in Bayesian inference, allowing estimation of posterior distributions for complex models.

      Related plugins: :doc:`/plugins/statistical-physics`, :doc:`/plugins/machine-learning`

   Message Queue
      A form of asynchronous service-to-service communication used in distributed systems. Message queues store messages sent between applications, allowing decoupled communication and improving system reliability and scalability.

      Related plugins: :doc:`/plugins/backend-development`, :doc:`/plugins/full-stack-orchestration`

   Microservices
      An architectural style that structures an application as a collection of loosely coupled, independently deployable services. Each microservice implements specific business capabilities and communicates with others via well-defined APIs.

      Related plugins: :doc:`/plugins/backend-development`, :doc:`/plugins/python-development`, :doc:`/plugins/cicd-automation`

   Observability
      The ability to measure a system's internal state based on its external outputs. In software systems, observability encompasses logging, metrics, and distributed tracing to understand system behavior and diagnose issues.

      Related plugins: :doc:`/plugins/observability-monitoring`, :doc:`/plugins/debugging-toolkit`

   ORM
      Object-Relational Mapping. A programming technique for converting data between incompatible type systems in object-oriented programming languages. ORMs provide a high-level abstraction for database operations, mapping database tables to classes.

      Related plugins: :doc:`/plugins/backend-development`, :doc:`/plugins/python-development`

   Parallel Computing
      A type of computation where multiple calculations or processes are carried out simultaneously. Parallel computing divides large problems into smaller tasks that can be solved concurrently, leveraging multiple processors or cores.

      Related plugins: :doc:`/plugins/hpc-computing`, :doc:`/plugins/julia-development`, :doc:`/plugins/molecular-simulation`

   REST API
      Representational State Transfer Application Programming Interface. An architectural style for designing networked applications using HTTP requests to access and manipulate resources. REST APIs use standard HTTP methods (GET, POST, PUT, DELETE) and are stateless.

      Related plugins: :doc:`/plugins/backend-development`, :doc:`/plugins/python-development`, :doc:`/plugins/llm-application-dev`

   SciML
      Scientific Machine Learning. An emerging field that combines scientific computing with machine learning techniques. SciML integrates physics-based models with data-driven approaches for solving complex scientific problems.

      Related plugins: :doc:`/plugins/julia-development`, :doc:`/plugins/jax-implementation`, :doc:`/plugins/statistical-physics`

   TDD
      Test-Driven Development. A software development process where tests are written before the actual code. TDD follows a cycle of writing a failing test, implementing code to pass the test, and refactoring the code for improvement.

      Related plugins: :doc:`/plugins/unit-testing`, :doc:`/plugins/quality-engineering`, :doc:`/plugins/python-development`

   Terraform
      An open-source infrastructure as code software tool that enables defining and provisioning data center infrastructure using a declarative configuration language. Terraform manages both cloud and on-premises resources.

      Related plugins: :doc:`/plugins/cicd-automation`

Additional Terms
----------------

.. glossary::
   :sorted:

   Agent
      In the context of this marketplace, an agent is an AI-powered assistant with specialized expertise in a particular domain. Agents can understand context, provide recommendations, and assist with complex technical tasks.

   Command
      A specific action or operation that a plugin provides, typically invoked through a command-line interface or programmatic API. Commands encapsulate reusable functionality for common development tasks.

   Distributed Tracing
      A method used to track requests as they flow through distributed systems and microservices. Distributed tracing helps identify performance bottlenecks and understand complex interactions between services.

      Related plugins: :doc:`/plugins/observability-monitoring`, :doc:`/plugins/backend-development`

   Plugin
      A modular software component that adds specific capabilities to the marketplace. Each plugin contains agents, commands, and skills focused on a particular technical domain or workflow.

   Skill
      A reusable capability or knowledge module provided by a plugin. Skills represent specialized expertise that can be applied across different contexts and combined with other skills for complex tasks.

   Workflow
      A sequence of connected steps or tasks that accomplish a specific goal. In this documentation, workflows often combine multiple plugins to solve complex problems spanning different technical domains.

   XLA
      Accelerated Linear Algebra. A domain-specific compiler for linear algebra that optimizes TensorFlow and JAX computations. XLA generates efficient machine code for various hardware accelerators including GPUs and TPUs.

      Related plugins: :doc:`/plugins/jax-implementation`, :doc:`/plugins/deep-learning`

Cross-References
----------------

For comprehensive information on how plugins work together, see:

- :doc:`integration-map` - Complete integration matrix showing plugin relationships
- :doc:`guides/integration-patterns` - Best practices for combining plugins
- :doc:`guides/index` - Quick-start guides for common workflows

See Also
--------

- `Sphinx Glossary Directive Documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-glossary>`_
- Plugin documentation pages for detailed technical information
- Category pages for domain-specific terminology
