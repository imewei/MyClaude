---
name: type-driven-design
version: "2.1.0"
description: Master type-driven design in Python using Protocols, Generics, and static analysis. Use when designing library interfaces, implementing structural typing, using Generic types for reusable components, or enforcing strict type safety with pyright/mypy.
---

# Type-Driven Design in Python

Treat type hints as a contract to build more robust and self-documenting systems.

## Expert Agent

For complex architecture design or strict type-safety enforcement, delegate to:

- **`python-pro`**: Expert in structural typing, generics, and Python systems architecture.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## 1. Structural Typing (Protocols)

Use `typing.Protocol` to define interfaces based on behavior rather than inheritance.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DataSink(Protocol):
    def write(self, data: bytes) -> int: ...
    def flush(self) -> None: ...

class FileSink:
    def write(self, data: bytes) -> int:
        # Implementation
        return len(data)
    def flush(self) -> None:
        pass

def process_to_sink(sink: DataSink, data: bytes):
    sink.write(data)
    sink.flush()
```

## 2. Generics and TypeVars

Use `TypeVar` and `Generic` to create components that work across multiple types while preserving type information.

```python
from typing import TypeVar, Generic, Sequence

T = TypeVar("T")

class Repository(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> Sequence[T]:
        return self._items

# Usage
user_repo = Repository[User]()
user_repo.add(User(id=1))  # Type-checked
```

## 3. Strict Static Analysis

Always configure your project for strict type checking in `pyproject.toml`.

```toml
[tool.pyright]
typeCheckingMode = "strict"
reportMissingTypeStubs = false

[tool.mypy]
strict = true
warn_unreachable = true
pretty = true
```

## 4. Patterns and Best Practices

| Pattern | Description |
|---------|-------------|
| **NewType** | Create distinct types for the same underlying data (e.g., `UserId = NewType("UserId", int)`). |
| **Annotated** | Add metadata to types for runtime validation (e.g., with Pydantic). |
| **Literal** | Restrict values to a specific set of literals. |
| **TypeGuard** | Define functions that narrow types for the type checker. |

## Checklist

- [ ] Protocols used instead of abstract base classes where appropriate.
- [ ] Generics used to preserve type information in reusable components.
- [ ] `pyright` or `mypy --strict` passes without errors.
- [ ] IO boundaries are validated using Pydantic or similar.
- [ ] `NewType` used to prevent accidental mixing of primitive values.
