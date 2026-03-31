---
name: documentation-standards
version: "2.2.1"
description: Guidelines for high-quality technical documentation, including API specs, READMEs, and internal runbooks.
---

# Documentation Standards

Expert guide for maintaining clear, accurate, and useful technical documentation.

## 1. Documentation Types

- **README**: The entry point for any project. Should include setup, usage, and contribution guides.
- **API Specs**: Detailed documentation of endpoints, payloads, and error codes (e.g., OpenAPI/Swagger).
- **Runbooks**: Actionable guides for responding to specific alerts or maintaining systems.
- **Architecture Decision Records (ADR)**: Documentation of significant design choices and their rationale.

## 2. Best Practices

- **Clarity**: Use clear, concise language. Avoid jargon where possible.
- **Visuals**: Use diagrams (Mermaid, C4) to explain complex flows.
- **Accuracy**: Keep docs close to the code. Treat stale documentation as a bug.
- **In-Code Docs**: Use standard formats (JSDoc, Docstrings) for public APIs.

## 3. Documentation Checklist

- [ ] Setup instructions are tested and work from a fresh clone.
- [ ] All public API parameters and return values are documented.
- [ ] Examples are provided for common use cases.
- [ ] Diagrams accurately reflect the current system state.
- [ ] Internal links are valid and not broken.
