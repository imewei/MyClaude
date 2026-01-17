# Systems Programming Patterns - Assets

This directory contains diagrams and visual aids for understanding systems programming patterns.

## Diagram Generation

The diagrams in this directory are provided as ASCII art for maximum portability and can be rendered using various tools:

- **Mermaid**: For flowcharts and sequence diagrams
- **PlantUML**: For UML and architecture diagrams
- **Graphviz DOT**: For graph structures
- **ASCII art**: For simple illustrations

## Available Diagrams

### Memory Management

1. **arena-allocator-layout.md** - Visual representation of arena/bump allocator
2. **memory-pool-structure.md** - Fixed-size memory pool internal structure
3. **cache-hierarchy.md** - CPU cache levels and memory hierarchy

### Concurrency

4. **lock-free-stack.md** - Lock-free stack push/pop operations
5. **aba-problem.md** - Illustration of the ABA problem in lock-free algorithms
6. **rcu-workflow.md** - RCU read-update-reclaim cycle

### Debugging

7. **memory-error-detection.md** - How sanitizers detect memory errors
8. **profiling-workflow.md** - Step-by-step profiling methodology

## Usage

Most diagrams include both ASCII art representations and source code for rendering tools. To render:

```bash
# Mermaid (install mermaid-cli)
mmdc -i diagram.mmd -o diagram.svg

# PlantUML
plantuml diagram.puml

# Graphviz
dot -Tsvg diagram.dot -o diagram.svg
```

## Contributing

When adding new diagrams:
1. Include both source format and ASCII art fallback
2. Add description in this README
3. Reference from appropriate markdown files in references/
