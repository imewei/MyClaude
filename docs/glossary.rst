Glossary
========

This glossary provides definitions for the key technical terms used throughout the Claude Code Plugin Marketplace. It is designed to help both new and experienced developers understand the concepts and tools involved in our 3-suite architecture.

.. glossary::

   Hub Skill
      A meta-orchestrator skill registered in ``plugin.json`` that contains a routing decision tree dispatching to specialized sub-skills. Hub skills are the only skills visible in the manifest; sub-skills are discovered through hub references. See also :term:`Sub-Skill`.

   Sub-Skill
      A specialized skill directory containing a ``SKILL.md`` that is not directly registered in ``plugin.json``. Sub-skills are reached through a :term:`Hub Skill`'s routing decision tree via ``../`` relative links.

   Routing Decision Tree
      A code block inside a hub skill's ``SKILL.md`` that maps user intent to the appropriate sub-skill. The tree uses conditional logic (domain keywords, task type) to select the best sub-skill for a given request.

   Agent Team
      A pre-built configuration of 2-6 agents from one or more suites, coordinated by an orchestrator, designed for a specific workflow (e.g., incident response, Bayesian inference, PR review). See the :doc:`Agent Teams Guide <agent-teams-guide>` for all 21 templates.

   JAX
      JAX is a Python library for accelerator-oriented computing that combines Autograd and XLA for high-performance machine learning research. It provides a familiar NumPy-like API but with support for automatic differentiation, JIT compilation, and vectorization.

   Machine Learning
      Machine Learning (ML) is a subfield of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience and data.

   API
      An Application Programming Interface (API) is a set of rules and protocols that allows different software applications to communicate with each other. In this ecosystem, APIs are used by agents to interact with MCP servers and other tools.

   Python
      Python is a high-level, interpreted programming language known for its readability and versatility. It is the primary language used for developing plugins and suites in the MyClaude ecosystem.

   SciML
      Scientific Machine Learning (SciML) is an emerging field that combines traditional scientific modeling (like differential equations) with machine learning techniques to solve complex scientific and engineering problems.

   MCMC
      Markov Chain Monte Carlo (MCMC) is a class of algorithms for sampling from a probability distribution. It is widely used in scientific computing for Bayesian inference and statistical physics simulations.

   HPC
      High Performance Computing (HPC) involves the use of supercomputers and parallel processing techniques to solve complex computational problems that require massive amounts of memory and processing power.

   CI/CD
      Continuous Integration (CI) and Continuous Deployment (CD) are practices in software engineering that automate the building, testing, and deployment of code changes to ensure high quality and fast delivery.

   Kubernetes
      Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. It is a key component of modern cloud-native infrastructure.

   Docker
      Docker is a platform that uses OS-level virtualization to deliver software in packages called containers. Containers are lightweight, portable, and ensure that applications run consistently across different environments.

   Terraform
      Terraform is an open-source infrastructure as code (IaC) tool that allows developers to define and provision data center infrastructure using a high-level configuration language.

   Ansible
      Ansible is an open-source automation tool used for IT tasks such as configuration management, application deployment, and intra-service orchestration.

   GPU
      A Graphics Processing Unit (GPU) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images and, increasingly, for general-purpose parallel computing in ML and science.

   Parallel
      Parallel computing is a type of computation in which many calculations or the execution of processes are carried out simultaneously. This is essential for HPC and deep learning workloads.

   REST
      Representational State Transfer (REST) is an architectural style for providing standards between computer systems on the web, making it easier for systems to communicate with each other.

   Microservices
      Microservices is an architectural style that structures an application as a collection of small, independent services that communicate over a network. This promotes scalability and maintainability.

   TDD
      Test-Driven Development (TDD) is a software development process relying on software requirements being converted to test cases before software is fully developed, and tracking all software development by repeatedly testing the software against all test cases.

   BDD
      Behavior-Driven Development (BDD) is an agile software development process that encourages collaboration among developers, QA, and non-technical or business participants in a software project.

   Cloud
      Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user.

   Observability
      Observability is the ability to measure the internal states of a system by examining its outputs. In software, this typically involves monitoring, logging, and distributed tracing.

   ORM
      Object-Relational Mapping (ORM) is a programming technique for converting data between incompatible type systems using object-oriented programming languages.

   Message Queue
      A message queue is a form of asynchronous service-to-service communication used in serverless and microservices architectures.

   Container Orchestration
      Container orchestration is the automatic process of managing or scheduling the work of individual containers for applications based on microservices within clusters.

   Scalability
      Scalability is the property of a system to handle a growing amount of work by adding resources to the system.

   Efficiency
      Efficiency in the context of scientific computing refers to the ratio of useful work performed to the resources (time, memory, energy) consumed.

   Reliability
      Reliability is the probability that a system will perform its intended function without failure for a specified period of time under specified conditions.

   Maintainability
      Maintainability is the ease with which a software system or component can be modified to correct faults, improve performance, or other attributes, or adapt to a changed environment.

   Integration
      Integration is the process of combining individual software modules into a unified system to ensure they work together seamlessly.

   Documentation
      Technical documentation refers to any document that explains how software works, how it is built, and how it should be used.

   Optimization
      Optimization is the selection of a best element, with regard to some criterion, from some set of available alternatives. In coding, it often refers to improving performance.
