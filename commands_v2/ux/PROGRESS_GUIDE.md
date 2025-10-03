# Progress Tracking Guide

Comprehensive guide to using the progress tracking system.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Best Practices](#best-practices)
5. [API Reference](#api-reference)

## Introduction

The progress tracking system provides beautiful, informative progress indicators for long-running operations.

### Features

- **Rich progress bars** with multiple columns
- **Hierarchical progress** for nested operations
- **Step tracking** for sequential operations
- **Live dashboard** for real-time monitoring
- **Time estimates** (ETA and elapsed time)
- **Status indicators** (success, error, warning)

## Basic Usage

### Simple Progress Bar

```python
from ux.progress.progress_tracker import ProgressTracker

tracker = ProgressTracker()

with tracker.live_progress():
    task = tracker.add_task("Processing files", total=100)

    for i in range(100):
        # Do work...
        tracker.update(task, advance=1)

    tracker.complete(task)
```

### Progress with Custom Updates

```python
task = tracker.add_task("Processing", total=1000)

# Update to specific value
tracker.update(task, completed=500)

# Advance by amount
tracker.update(task, advance=10)

# Update description
tracker.update(task, description="Processing batch 2")
```

### Indeterminate Progress

For operations without known total:

```python
task = tracker.add_task("Analyzing code")  # No total

# Updates show count, not percentage
tracker.update(task, advance=1)
tracker.update(task, advance=1)
```

## Advanced Features

### Hierarchical Progress

Track parent and child operations:

```python
parent = tracker.add_task("Code Optimization", total=3)

# Child tasks
analyze = tracker.add_task(
    "Analyzing code",
    total=100,
    parent_id=parent
)

optimize = tracker.add_task(
    "Applying optimizations",
    total=50,
    parent_id=parent
)

test = tracker.add_task(
    "Running tests",
    total=25,
    parent_id=parent
)

# Update children
for i in range(100):
    tracker.update(analyze, advance=1)

tracker.complete(analyze)
tracker.update(parent, advance=1)  # Update parent

# Continue with other children...
```

### Step Tracking

For sequential operations:

```python
from ux.progress.step_tracker import StepTracker

tracker = StepTracker("Deployment Pipeline")

# Define steps
tracker.add_step("build", "Build application")
tracker.add_step("test", "Run tests")
tracker.add_step("deploy", "Deploy to production")

# Execute steps
tracker.start_step("build")
# Do work...
tracker.complete_step("build")

tracker.start_step("test")
# Do work...
tracker.complete_step("test")

# Print summary
tracker.print_summary()
```

### Live Dashboard

Real-time monitoring dashboard:

```python
from ux.progress.live_dashboard import LiveDashboard

dashboard = LiveDashboard()

with dashboard.live():
    # Update command status
    dashboard.update_command("optimize", "Running")

    # Track agents
    dashboard.add_agent("Scientific Agent")
    dashboard.add_agent("Quality Agent")

    # Update metrics
    dashboard.increment_cache_hits()
    dashboard.increment_operations(completed=1)

    # Do work...

    # Clean up
    dashboard.remove_agent("Scientific Agent")
    dashboard.remove_agent("Quality Agent")
    dashboard.update_command("optimize", "Complete")
```

### Status Indicators

Mark tasks with different statuses:

```python
from ux.progress.progress_tracker import ProgressStatus

# Success
tracker.complete(task, ProgressStatus.SUCCESS)

# Error
tracker.complete(task, ProgressStatus.ERROR)

# Warning
tracker.complete(task, ProgressStatus.WARNING)

# Skipped
tracker.complete(task, ProgressStatus.SKIPPED)
```

### Metadata

Store additional information:

```python
task = tracker.add_task(
    "Processing",
    total=100,
    files_processed=0,
    errors_found=0
)

tracker.update(
    task,
    advance=1,
    files_processed=10,
    errors_found=2
)
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with tracker.live_progress():
    # Work here
    pass

# Bad - might leave progress display open
tracker.live_progress()
# Work here
```

### 2. Complete Tasks

Always mark tasks as complete:

```python
task = tracker.add_task("Processing", total=100)

try:
    for i in range(100):
        tracker.update(task, advance=1)
    tracker.complete(task, ProgressStatus.SUCCESS)
except Exception:
    tracker.complete(task, ProgressStatus.ERROR)
    raise
```

### 3. Meaningful Descriptions

```python
# Good
task = tracker.add_task("Analyzing Python files in src/")

# Bad
task = tracker.add_task("Processing")
```

### 4. Appropriate Granularity

```python
# Good - update every 1% (for 10000 items)
if i % 100 == 0:
    tracker.update(task, completed=i)

# Bad - update for every item (too frequent)
for i in range(10000):
    tracker.update(task, advance=1)
```

### 5. Use Step Tracking for Sequences

```python
# Good - clear sequence
tracker = StepTracker("Operation")
tracker.add_step("step1", "First step")
tracker.add_step("step2", "Second step")

# Bad - trying to track steps with progress
task1 = tracker.add_task("Step 1")
task2 = tracker.add_task("Step 2")
```

### 6. Handle Errors Gracefully

```python
tracker = StepTracker("Process")
tracker.add_step("step1", "Step 1")
tracker.add_step("step2", "Step 2")

tracker.start_step("step1")

try:
    # Work...
    tracker.complete_step("step1")
except Exception as e:
    tracker.fail_step("step1", str(e))
    # Skip remaining steps
    tracker.skip_step("step2", "Skipped due to earlier failure")
    raise
```

## API Reference

### ProgressTracker

#### Constructor

```python
ProgressTracker(
    console=None,              # Rich console instance
    enabled=True,              # Enable/disable tracking
    show_time=True,            # Show elapsed time
    show_percentage=True,      # Show percentage
    show_eta=True,             # Show ETA
    refresh_rate=10           # Updates per second
)
```

#### Methods

##### add_task

```python
add_task(
    description: str,          # Task description
    total: Optional[int] = None,  # Total items (None for indeterminate)
    parent_id: Optional[str] = None,  # Parent task ID
    **metadata                # Additional metadata
) -> str  # Returns task ID
```

##### update

```python
update(
    task_id: str,              # Task ID
    completed: Optional[int] = None,  # Set completed count
    advance: Optional[int] = None,    # Advance by amount
    description: Optional[str] = None,  # Update description
    **metadata                # Update metadata
)
```

##### complete

```python
complete(
    task_id: str,              # Task ID
    status: ProgressStatus = ProgressStatus.SUCCESS  # Final status
)
```

##### live_progress

```python
live_progress() -> ContextManager
# Returns context manager for live updates
```

### StepTracker

#### Constructor

```python
StepTracker(
    name: str,                 # Operation name
    console=None,              # Rich console instance
    enabled=True              # Enable/disable tracking
)
```

#### Methods

##### add_step

```python
add_step(
    step_id: str,              # Step ID
    name: str,                 # Step name
    description: str = "",     # Step description
    **metadata                # Additional metadata
) -> Step
```

##### start_step

```python
start_step(step_id: str)
```

##### complete_step

```python
complete_step(
    step_id: str,
    status: StepStatus = StepStatus.COMPLETED,
    error: Optional[str] = None
)
```

##### fail_step

```python
fail_step(
    step_id: str,
    error: str
)
```

##### skip_step

```python
skip_step(
    step_id: str,
    reason: str = ""
)
```

### LiveDashboard

#### Constructor

```python
LiveDashboard(
    console=None,              # Rich console instance
    enabled=True,              # Enable/disable dashboard
    refresh_rate=4            # Updates per second
)
```

#### Methods

##### update_command

```python
update_command(
    command: str,              # Command name
    status: str               # Status (Running, Complete, etc.)
)
```

##### add_agent / remove_agent

```python
add_agent(agent_name: str)
remove_agent(agent_name: str)
```

##### increment_cache_hits / increment_cache_misses

```python
increment_cache_hits(count: int = 1)
increment_cache_misses(count: int = 1)
```

##### increment_operations

```python
increment_operations(
    completed: int = 0,
    failed: int = 0
)
```

## Examples

See [examples/progress_example.py](examples/progress_example.py) for complete examples.

## Troubleshooting

### Progress not showing

1. Check if rich is installed: `pip install rich`
2. Check if terminal supports rich output
3. Verify `enabled=True` in constructor

### Progress updates too slow

Increase refresh rate:

```python
tracker = ProgressTracker(refresh_rate=20)  # 20 updates/sec
```

### Progress bar flickering

Decrease refresh rate:

```python
tracker = ProgressTracker(refresh_rate=5)  # 5 updates/sec
```

### Memory usage growing

Clear completed tasks:

```python
tracker.clear()
```

## Integration

### With asyncio

```python
import asyncio

async def process_items(tracker, task_id, items):
    for item in items:
        await process_item(item)
        tracker.update(task_id, advance=1)

with tracker.live_progress():
    task = tracker.add_task("Processing", total=100)
    asyncio.run(process_items(tracker, task, items))
```

### With multiprocessing

```python
from multiprocessing import Pool, Manager

def worker(task_id, item, counter):
    result = process(item)
    counter.value += 1
    return result

with Manager() as manager:
    counter = manager.Value('i', 0)

    with tracker.live_progress():
        task = tracker.add_task("Processing", total=100)

        with Pool() as pool:
            results = [
                pool.apply_async(worker, (task, item, counter))
                for item in items
            ]

            while counter.value < len(items):
                tracker.update(task, completed=counter.value)
                time.sleep(0.1)
```

## Further Reading

- [Error Handling Guide](ERROR_GUIDE.md)
- [Recommendation Guide](RECOMMENDATION_GUIDE.md)
- [Main README](README.md)