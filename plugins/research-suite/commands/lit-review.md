---
name: lit-review
description: Structured literature review — topic scan, claim extraction, evidence synthesis, gap identification, and output as summary, table, or annotated bibliography.
argument-hint: "[--topic \"query\"] [--scope narrow|broad] [--output summary|table|annotated-bib]"
allowed-tools: ["Read", "Write", "WebSearch", "WebFetch"]
---

# /lit-review — Literature Review

Routes to `research-expert` via `research-suite:research-practice` hub.

## Usage

```
/lit-review --topic "physics-informed neural networks for fluid dynamics" --scope broad --output table
/lit-review --topic "Bayesian UDE parameter estimation" --scope narrow --output annotated-bib
/lit-review --topic "JAX-MD molecular dynamics" --scope narrow --output summary
```

## What This Does

1. Searches for papers matching `--topic`
2. Extracts key claims, methods, and results from top sources
3. Synthesizes evidence and identifies research gaps
4. Formats output as `--output` type

## Output Types

| `--output` | Format |
|---|---|
| `summary` | 3-5 paragraph narrative synthesis |
| `table` | Markdown table: paper, method, key result, limitation |
| `annotated-bib` | BibTeX entries with 2-sentence annotation each |

## Token Strategy

PRISMA/GRADE checklist templates load only for `--scope broad` reviews. Narrow reviews use lightweight claim extraction only.
