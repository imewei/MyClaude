# Deep Learning Plugin

Comprehensive deep learning expertise covering neural network theory, architecture design, multi-framework implementation, training diagnostics, and research translation.

## Agents

### neural-architecture-engineer
**Architecture Design & Multi-Framework Implementation**

- Design neural network architectures (transformers, CNNs, RNNs, attention mechanisms)
- Compare and select frameworks (Flax, Equinox, Haiku, PyTorch, Keras)
- Implement state-of-the-art architectures (BERT, GPT, ResNet, ViT)
- Design training strategies (learning rate schedules, regularization, data augmentation)
- Debug training issues (convergence, overfitting, gradient explosions)
- Optimize model performance (memory usage, inference latency)

**When to use:**
- Designing new neural network architectures
- Selecting and comparing deep learning frameworks
- Implementing SOTA architectures from papers
- Setting up complete training pipelines
- Framework migration and conversion

### neural-network-master
**Theory, Debugging & Research Translation**

- Deep theoretical understanding (optimization theory, statistical learning theory, information theory)
- Mathematical foundations (backpropagation, loss landscapes, convergence analysis)
- Training diagnostics & expert debugging (gradient issues, loss curve interpretation)
- Research paper translation (SOTA papers → practical implementation)
- Pedagogical explanations (teaching neural networks from first principles)
- Advanced topics (meta-learning, continual learning, adversarial robustness)

**When to use:**
- Understanding why neural networks behave a certain way (theory)
- Debugging complex training issues (vanishing gradients, loss plateaus)
- Translating research papers into implementation guidance
- Learning neural network concepts from first principles
- Analyzing SOTA architectures with theoretical depth
- Mathematical deep dives (deriving backpropagation, analyzing convergence)

## Agent Collaboration

### Complementary Roles

**neural-network-master** provides the "WHY":
- Theoretical understanding and mathematical foundations
- Diagnostic insights and root cause analysis
- Research insights and theoretical motivations
- Pedagogical explanations that build understanding

**neural-architecture-engineer** provides the "WHAT":
- Concrete architecture design decisions
- Framework selection and implementation
- Training pipeline setup and execution
- Production-ready model development

### Typical Workflows

#### Workflow 1: Research to Production
1. **neural-network-master**: Analyzes research paper, explains theory, identifies key components
2. **neural-architecture-engineer**: Designs architecture, selects framework, implements model
3. **jax-pro** (if JAX): Optimizes JAX transformations for performance
4. **mlops-engineer**: Deploys to production with monitoring

#### Workflow 2: Training Issue Resolution
1. **neural-network-master**: Diagnoses training issue theoretically (e.g., vanishing gradients)
2. **neural-network-master**: Proposes theoretically-grounded solutions
3. **neural-architecture-engineer**: Implements architecture changes (e.g., add residual connections)
4. **neural-architecture-engineer**: Reruns training and validates fix

#### Workflow 3: Novel Architecture Design
1. **neural-network-master**: Explains theoretical motivations for architecture ideas
2. **neural-architecture-engineer**: Designs and implements architecture variants
3. **neural-architecture-engineer**: Runs experiments and hyperparameter searches
4. **neural-network-master**: Analyzes results theoretically, guides next iterations

## Delegation Patterns

### From Deep Learning Agents to Others

**To jax-pro:**
- JAX-specific transformations (jit, vmap, pmap, scan, remat)
- Performance optimization with XLA compilation
- Multi-device parallelization strategies

**To mlops-engineer:**
- Production deployment and model serving
- MLOps infrastructure and pipelines
- Monitoring and maintenance

**To data-scientist:**
- Exploratory data analysis
- Feature engineering
- Statistical analysis and data quality

**To visualization-interface:**
- Advanced visualization (loss landscapes, t-SNE, attention maps)
- Interactive analysis tools
- Custom plotting for research

### From Others to Deep Learning Agents

**Call neural-network-master when:**
- Need theoretical understanding or mathematical explanations
- Debugging complex training issues
- Translating research papers
- Learning concepts from first principles

**Call neural-architecture-engineer when:**
- Need to design and implement architectures
- Selecting frameworks for a project
- Setting up complete training pipelines
- Implementing SOTA models from papers

## Key Features

### Theoretical Depth
- Optimization theory (loss landscapes, convergence, implicit regularization)
- Statistical learning theory (generalization, VC dimension, double descent)
- Information theory (compression, mutual information, information bottleneck)
- Representation learning (manifolds, disentanglement, inductive biases)
- Geometric deep learning (symmetries, equivariance, graph networks)

### Multi-Framework Expertise
- **Flax**: Modern JAX neural networks with Linen API, stateful training
- **Equinox**: Functional PyTorch-like JAX library with PyTree integration
- **Haiku**: DeepMind's functional approach to neural networks in JAX
- **Keras**: High-level API with JAX backend for rapid prototyping
- **PyTorch**: Reference for cross-framework understanding and migration

### Advanced Capabilities
- Training diagnostics (gradient pathologies, loss curve interpretation, convergence analysis)
- Research translation (paper decoding, SOTA architecture analysis)
- Pedagogical mastery (concept building, visualization, analogies, historical context)
- Cutting-edge topics (transformers, diffusion models, self-supervised learning, LLMs)
- Ethical AI (bias & fairness, interpretability, robustness & safety)

## Examples

### Example 1: Understanding Training Behavior
```
User: "Why does my transformer converge faster than my CNN on the same task?"

neural-network-master responds:
- Explains inductive biases: CNNs assume translation equivariance, transformers are more flexible
- Discusses optimization landscapes: Transformers can have smoother loss landscapes due to skip connections
- Analyzes attention: Self-attention provides direct paths for gradient flow
- Cites theory: Transformers have weaker inductive bias but learn patterns from data
- Provides mathematical intuition: Attention as weighted aggregation with learned weights

Result: User understands the theoretical trade-offs and can make informed architecture choices
```

### Example 2: Implementing Research Paper
```
User: "Can you help me implement 'Attention is All You Need'?"

Collaboration:
1. neural-network-master: Explains self-attention mechanism theory, positional encoding math
2. neural-network-master → neural-architecture-engineer: "Implement transformer with these essential components"
3. neural-architecture-engineer: Designs architecture, selects Flax, implements multi-head attention
4. neural-architecture-engineer: Creates training pipeline with Adam optimizer

Result: User has both theoretical understanding AND working implementation
```

### Example 3: Debugging Training Issue
```
User: "My model's loss suddenly spiked at epoch 50"

neural-network-master workflow:
1. Analyzes loss curves and gradient statistics
2. Diagnoses: Learning rate too high, escaped local minimum into high-curvature region
3. Explains theory: Loss landscape geometry, curvature, learning rate vs step size
4. Recommends: Learning rate warmup + cosine annealing
5. Delegates to neural-architecture-engineer: "Implement cosine annealing schedule"

Result: Issue resolved with theoretical understanding of root cause
```

## Installation & Usage

### Prerequisites
- Python 3.12+
- JAX (for JAX-based frameworks)
- PyTorch (optional, for cross-framework comparisons)
- Jupyter (recommended for interactive exploration)

### Agent Invocation

```python
# Use neural-network-master for theory and debugging
/ask neural-network-master "Why do residual connections help gradient flow?"

# Use neural-architecture-engineer for implementation
/ask neural-architecture-engineer "Design a vision transformer for image classification"

# Collaborative workflow
/ask neural-network-master "Explain BERT pretraining theory"
# Then use neural-architecture-engineer to implement based on guidance
```

## Resources

### Papers & Theory
- "Attention is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "Deep Residual Learning" (He et al., 2015) - ResNet and skip connections
- "Deep Double Descent" (Nakkiran et al., 2019) - Modern generalization theory
- "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019) - Pruning and sparsity

### Frameworks
- [JAX](https://github.com/google/jax) - Composable transformations of Python+NumPy
- [Flax](https://github.com/google/flax) - Neural networks in JAX
- [Equinox](https://github.com/patrick-kidger/equinox) - Functional JAX library
- [Haiku](https://github.com/deepmind/dm-haiku) - DeepMind's JAX neural networks

### Learning
- [Distill.pub](https://distill.pub/) - Interactive explanations
- [Neural Network Visualization](http://playground.tensorflow.org/) - Interactive playground
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville

## Contributing

This plugin is part of the scientific computing workflows marketplace. Contributions welcome for:
- Additional neural network architectures
- New debugging diagnostics
- Research paper translations
- Pedagogical improvements
- Framework support expansions

## License

MIT License - See repository root for details

## Authors

Wei Chen - Initial development and deep learning expertise integration

---

*Bridging theory and practice in deep learning through complementary expertise in neural network fundamentals, architecture design, and production implementation.*
