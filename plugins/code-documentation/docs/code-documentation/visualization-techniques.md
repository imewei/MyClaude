# Visualization Techniques

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: Generate visual representations of code structure, flow, and algorithms

## Overview

Transform code into visual diagrams including flowcharts, class diagrams, sequence diagrams, algorithm visualizations, and recursion trees using Mermaid and other formats.

## Flow Diagram Generation

### VisualExplainer Class

```python
class VisualExplainer:
    def generate_flow_diagram(self, code_structure):
        """
        Generate Mermaid flowchart showing code execution flow

        Args:
            code_structure: Dictionary with type and function/control flow data

        Returns:
            Mermaid flowchart diagram as string
        """
        diagram = "```mermaid\nflowchart TD\n"

        if code_structure['type'] == 'function_flow':
            nodes = []
            edges = []

            for i, func in enumerate(code_structure['functions']):
                node_id = f"F{i}"
                nodes.append(f"    {node_id}[{func['name']}]")

                # Add function details
                if func.get('parameters'):
                    nodes.append(f"    {node_id}_params[/{', '.join(func['parameters'])}/]")
                    edges.append(f"    {node_id}_params --> {node_id}")

                # Add return value
                if func.get('returns'):
                    nodes.append(f"    {node_id}_return[{func['returns']}]")
                    edges.append(f"    {node_id} --> {node_id}_return")

                # Connect to called functions
                for called in func.get('calls', []):
                    called_id = f"F{code_structure['function_map'][called]}"
                    edges.append(f"    {node_id} --> {called_id}")

            diagram += "\n".join(nodes) + "\n"
            diagram += "\n".join(edges) + "\n"

        diagram += "```"
        return diagram
```

### Control Flow Diagram

```python
def generate_control_flow(self, function_node):
    """
    Generate control flow diagram for a single function

    Shows: if/else branches, loops, try/except blocks
    """
    diagram = "```mermaid\nflowchart TD\n"

    # Start node
    diagram += f"    Start([{function_node.name}])\n"

    # Parse function body
    node_counter = 0
    edges = []

    for i, stmt in enumerate(function_node.body):
        node_id = f"N{node_counter}"

        if isinstance(stmt, ast.If):
            # If-else branching
            diagram += f"    {node_id}{{{{ {ast.unparse(stmt.test)} }}}}\n"

            # True branch
            true_id = f"N{node_counter + 1}"
            diagram += f"    {true_id}[True path]\n"
            edges.append(f"    {node_id} -->|Yes| {true_id}")

            # False branch
            false_id = f"N{node_counter + 2}"
            diagram += f"    {false_id}[False path]\n"
            edges.append(f"    {node_id} -->|No| {false_id}")

            node_counter += 3

        elif isinstance(stmt, ast.While):
            # While loop
            diagram += f"    {node_id}{{{{ {ast.unparse(stmt.test)} }}}}\n"
            loop_body = f"N{node_counter + 1}"
            diagram += f"    {loop_body}[Loop body]\n"
            edges.append(f"    {node_id} -->|Continue| {loop_body}")
            edges.append(f"    {loop_body} --> {node_id}")

            node_counter += 2

        elif isinstance(stmt, ast.Return):
            # Return statement
            diagram += f"    {node_id}[Return {ast.unparse(stmt.value) if stmt.value else 'None'}]\n"
            diagram += f"    End([End])\n"
            edges.append(f"    {node_id} --> End")
            node_counter += 1

        else:
            # Regular statement
            stmt_text = ast.unparse(stmt)[:50]  # Truncate long statements
            diagram += f"    {node_id}[{stmt_text}]\n"
            node_counter += 1

    # Add edges
    diagram += "\n".join(edges) + "\n"
    diagram += "```"

    return diagram
```

## Class Diagram Generation

```python
def generate_class_diagram(self, classes):
    """
    Generate UML-style class diagram using Mermaid

    Args:
        classes: List of class dictionaries with attributes and methods

    Returns:
        Mermaid class diagram
    """
    diagram = "```mermaid\nclassDiagram\n"

    for cls in classes:
        # Class definition
        diagram += f"    class {cls['name']} {{\n"

        # Attributes
        for attr in cls.get('attributes', []):
            visibility = '+' if attr.get('public', True) else '-'
            diagram += f"        {visibility}{attr['name']} : {attr.get('type', 'Any')}\n"

        # Methods
        for method in cls.get('methods', []):
            visibility = '+' if method.get('public', True) else '-'
            params = ', '.join(method.get('params', []))
            returns = method.get('returns', 'None')
            diagram += f"        {visibility}{method['name']}({params}) : {returns}\n"

        diagram += "    }\n"

        # Relationships
        if cls.get('inherits'):
            diagram += f"    {cls['inherits']} <|-- {cls['name']}\n"

        for composition in cls.get('compositions', []):
            diagram += f"    {cls['name']} *-- {composition}\n"

        for aggregation in cls.get('aggregations', []):
            diagram += f"    {cls['name']} o-- {aggregation}\n"

        for association in cls.get('associations', []):
            diagram += f"    {cls['name']} --> {association}\n"

    diagram += "```"
    return diagram
```

## Sequence Diagram Generation

```python
def generate_sequence_diagram(self, interactions):
    """
    Generate sequence diagram showing object interactions

    Args:
        interactions: List of {from, to, message, type} dictionaries

    Returns:
        Mermaid sequence diagram
    """
    diagram = "```mermaid\nsequenceDiagram\n"

    # Extract participants
    participants = set()
    for interaction in interactions:
        participants.add(interaction['from'])
        participants.add(interaction['to'])

    # Declare participants
    for participant in sorted(participants):
        diagram += f"    participant {participant}\n"

    diagram += "\n"

    # Add interactions
    for interaction in interactions:
        from_obj = interaction['from']
        to_obj = interaction['to']
        message = interaction['message']
        msg_type = interaction.get('type', 'sync')

        if msg_type == 'sync':
            diagram += f"    {from_obj}->>+{to_obj}: {message}\n"
            if interaction.get('returns'):
                diagram += f"    {to_obj}-->>-{from_obj}: {interaction['returns']}\n"
        elif msg_type == 'async':
            diagram += f"    {from_obj}-)>{to_obj}: {message}\n"
        elif msg_type == 'note':
            diagram += f"    Note over {from_obj},{to_obj}: {message}\n"

    diagram += "```"
    return diagram
```

## Algorithm Visualization

### Sorting Algorithm Visualization

```python
class AlgorithmVisualizer:
    def visualize_sorting_algorithm(self, algorithm_name, array):
        """
        Create step-by-step visualization of sorting algorithm

        Supports: bubble_sort, quick_sort, merge_sort, insertion_sort
        """
        steps = []

        if algorithm_name == 'bubble_sort':
            steps.append("""
## Bubble Sort Visualization

**Initial Array**: {array}

### How Bubble Sort Works:
1. Compare adjacent elements
2. Swap if they're in wrong order
3. Repeat until no swaps needed

### Step-by-Step Execution:
""".format(array=array))

            # Simulate bubble sort with visualization
            arr = array.copy()
            n = len(arr)

            for i in range(n):
                swapped = False
                step_viz = f"\n**Pass {i+1}**:\n"

                for j in range(0, n-i-1):
                    # Show comparison
                    step_viz += f"Compare [{arr[j]}] and [{arr[j+1]}]: "

                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
                        step_viz += f"Swap → {arr}\n"
                        swapped = True
                    else:
                        step_viz += "No swap needed\n"

                steps.append(step_viz)

                if not swapped:
                    steps.append(f"\n✅ Array is sorted: {arr}")
                    break

        return '\n'.join(steps)
```

### Recursion Visualization

```python
def visualize_recursion(self, func_name, example_input):
    """
    Visualize recursive function calls using tree structure

    Args:
        func_name: Name of recursive function
        example_input: Input to trace

    Returns:
        ASCII tree showing call stack
    """
    viz = f"""
## Recursion Visualization: {func_name}

### Call Stack Visualization:
```
{func_name}({example_input})
│
├─> Base case check: {example_input} == 0? No
├─> Recursive call: {func_name}({example_input - 1})
│   │
│   ├─> Base case check: {example_input - 1} == 0? No
│   ├─> Recursive call: {func_name}({example_input - 2})
│   │   │
│   │   ├─> Base case check: 1 == 0? No
│   │   ├─> Recursive call: {func_name}(0)
│   │   │   │
│   │   │   └─> Base case: Return 1
│   │   │
│   │   └─> Return: 1 * 1 = 1
│   │
│   └─> Return: 2 * 1 = 2
│
└─> Return: 3 * 2 = 6
```

**Final Result**: {func_name}({example_input}) = 6
"""
    return viz

def generate_recursion_tree(self, func_name, n, func):
    """
    Generate actual recursion tree by tracing execution

    Args:
        func_name: Function name
        n: Input value
        func: Actual function to trace

    Returns:
        Tree structure showing all calls
    """
    call_tree = []

    def traced_func(x, depth=0):
        indent = "  " * depth
        call_tree.append(f"{indent}{func_name}({x})")

        if x <= 1:
            call_tree.append(f"{indent}  → base case: {x}")
            return x

        result = traced_func(x - 1, depth + 1)
        call_tree.append(f"{indent}  → return: {x} * {result} = {x * result}")
        return x * result

    traced_func(n)
    return "\n".join(call_tree)
```

## Data Structure Visualization

```python
def visualize_tree(self, tree_data):
    """
    Visualize tree data structure

    Args:
        tree_data: Dictionary with {value, left, right} structure

    Returns:
        Mermaid graph showing tree structure
    """
    diagram = "```mermaid\ngraph TD\n"

    def add_node(node, node_id="root"):
        if node is None:
            return

        # Add current node
        diagram_lines.append(f"    {node_id}[{node['value']}]")

        # Add left child
        if node.get('left'):
            left_id = f"{node_id}_L"
            diagram_lines.append(f"    {node_id} --> {left_id}")
            add_node(node['left'], left_id)

        # Add right child
        if node.get('right'):
            right_id = f"{node_id}_R"
            diagram_lines.append(f"    {node_id} --> {right_id}")
            add_node(node['right'], right_id)

    diagram_lines = []
    add_node(tree_data)
    diagram += "\n".join(diagram_lines) + "\n```"

    return diagram
```

## Architecture Diagrams

```python
def generate_architecture_diagram(self, components):
    """
    Generate system architecture diagram

    Args:
        components: List of {name, type, connections} dictionaries

    Returns:
        Mermaid architecture diagram
    """
    diagram = "```mermaid\ngraph TB\n"

    # Create subgraphs for layers
    layers = {}
    for component in components:
        layer = component.get('layer', 'default')
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(component)

    # Add components grouped by layer
    for layer, comps in layers.items():
        diagram += f"    subgraph {layer}\n"
        for comp in comps:
            comp_id = comp['id']
            comp_name = comp['name']
            comp_type = comp.get('type', 'component')

            if comp_type == 'database':
                diagram += f"        {comp_id}[({comp_name})]\n"
            elif comp_type == 'api':
                diagram += f"        {comp_id}[/{comp_name}/]\n"
            else:
                diagram += f"        {comp_id}[{comp_name}]\n"
        diagram += "    end\n"

    # Add connections
    for component in components:
        for connection in component.get('connections', []):
            diagram += f"    {component['id']} --> {connection}\n"

    diagram += "```"
    return diagram
```

## Usage Examples

### Generate Flow Diagram

```python
visualizer = VisualExplainer()

code_structure = {
    'type': 'function_flow',
    'functions': [
        {'name': 'main', 'parameters': [], 'returns': 'int', 'calls': ['process', 'save']},
        {'name': 'process', 'parameters': ['data'], 'returns': 'dict', 'calls': ['validate']},
        {'name': 'validate', 'parameters': ['data'], 'returns': 'bool', 'calls': []},
        {'name': 'save', 'parameters': ['result'], 'returns': 'None', 'calls': []}
    ],
    'function_map': {'main': 0, 'process': 1, 'validate': 2, 'save': 3}
}

diagram = visualizer.generate_flow_diagram(code_structure)
print(diagram)
```

### Generate Class Diagram

```python
classes = [
    {
        'name': 'Animal',
        'attributes': [
            {'name': 'name', 'type': 'str', 'public': True},
            {'name': 'age', 'type': 'int', 'public': True}
        ],
        'methods': [
            {'name': 'speak', 'params': [], 'returns': 'str', 'public': True}
        ]
    },
    {
        'name': 'Dog',
        'inherits': 'Animal',
        'attributes': [
            {'name': 'breed', 'type': 'str', 'public': True}
        ],
        'methods': [
            {'name': 'speak', 'params': [], 'returns': 'str', 'public': True},
            {'name': 'fetch', 'params': ['item'], 'returns': 'None', 'public': True}
        ]
    }
]

diagram = visualizer.generate_class_diagram(classes)
print(diagram)
```

### Visualize Algorithm

```python
algo_viz = AlgorithmVisualizer()

# Bubble sort
steps = algo_viz.visualize_sorting_algorithm('bubble_sort', [5, 2, 8, 1, 9])
print(steps)

# Recursion
recursion_viz = algo_viz.visualize_recursion('factorial', 3)
print(recursion_viz)
```

## Integration with Code Analysis

```python
from code_analysis_framework import CodeAnalyzer

# Analyze code
analyzer = CodeAnalyzer()
analysis = analyzer.analyze_complexity(code)

# Generate visualizations based on analysis
visualizer = VisualExplainer()

if analysis['metrics']['function_count'] > 3:
    # Generate function flow diagram
    flow = visualizer.generate_flow_diagram(extract_function_structure(code))

if analysis['patterns']:
    # Generate class diagram for design patterns
    classes = extract_class_structure(code)
    class_diagram = visualizer.generate_class_diagram(classes)
```

## Best Practices

1. **Keep diagrams focused**: Don't try to show everything in one diagram
2. **Use appropriate diagram types**:
   - Flow diagrams for logic flow
   - Class diagrams for structure
   - Sequence diagrams for interactions
3. **Add context**: Include titles and legends
4. **Limit complexity**: Break large diagrams into smaller ones
5. **Update with code**: Regenerate diagrams when code changes

## Supported Diagram Formats

- **Mermaid**: Flowcharts, class diagrams, sequence diagrams, state diagrams
- **PlantUML**: UML diagrams (alternative to Mermaid)
- **ASCII**: Simple text-based diagrams for recursion/trees
- **DOT/Graphviz**: Complex graph structures
