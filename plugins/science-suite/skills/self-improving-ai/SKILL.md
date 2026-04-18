---
name: self-improving-ai
description: Research-framework overview of self-improving AI systems — a routing skill that surveys four improvement families (inference-time scaling, self-refinement loops, autonomous research agents, evolutionary program search) and routes to depth-skill companions for tool-specific details. Use when choosing between approaches, when the user is surveying the field, or when an LLM system must improve its own outputs but the right family is not yet decided. Use proactively when the user mentions self-improving AI, test-time compute, inference-time scaling, best-of-N, tree-of-thoughts, MCTS for reasoning, Self-Refine, STaR, Reflexion, AlphaEvolve, autonomous research agents, or multi-agent research loops (AutoGen, CrewAI, LangGraph, smolagents). For the DSPy programmatic-prompt-optimization path specifically, see `dspy-basics`. For Constitutional AI / RLAIF / DPO / KTO / PPO training specifically, see `rlaif-training`.
---

# Self-Improving AI

Frameworks and techniques that let an LLM system get better at a task without a human rewriting prompts or hand-labelling more data.

## Expert Agents

For self-improving AI system design, prompt optimization, and autonomous research loops, delegate to:

- **`ai-engineer`** (primary): Production LLM applications, RAG, agentic AI, prompt programs.
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`
- **`prompt-engineer`** (secondary): Prompt technique, chain-of-thought, constitutional principles, evaluation harnesses.
  - *Location*: `plugins/science-suite/agents/prompt-engineer.md`
- **`research-expert`** (tertiary): Autonomous-research-loop design, experiment-planning agents, literature-synthesis pipelines.
  - *Location*: `plugins/research-suite/agents/research-expert.md`

---

## The four families of self-improvement

Pick the family that matches the signal you can cheaply measure:

1. **Programmatic prompt optimization** — treat prompts as compilable programs. An optimizer searches over prompt wordings, few-shot exemplars, module structures, and decoding parameters to maximize a downstream metric on a small training set.
2. **Inference-time scaling** — trade extra compute at inference for quality. Best-of-N with a verifier, tree-of-thoughts, MCTS over reasoning steps, beam search reranked by a reward model.
3. **Self-refinement loops** — generate → critique → revise. The model is its own verifier or shares one with a downstream checker. Self-Refine, STaR (rationalization bootstrap), Reflexion (verbal RL).
4. **Constitutional AI / RLAIF** — AI-generated preference labels drive a learned reward model or direct preference optimization. Replaces the expensive human-labelled pair stage of RLHF.

A fifth nascent family — **evolutionary search over prompt programs** (AlphaEvolve, OpenEvolve) — combines genetic programming with LLM-as-mutation-operator and is currently the state of the art for coding / math-proof domains.

---

## Tool catalog

| Family | Package | Key API |
|---|---|---|
| Programmatic prompts | **`dspy`** | `dspy.Signature`, `dspy.Module`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.Retrieve`, `dspy.Predict`, `dspy.Evaluate`, optimizers `MIPROv2`, `BootstrapFewShotWithRandomSearch`, `BetterTogether`, `COPRO` |
| Inference-time search | **`langchain`** + `RunnableParallel`, **`guidance`** constrained decoding, hand-rolled best-of-N | `with_structured_output`, `RunnableParallel`, `select()` |
| Tree search / MCTS over reasoning | **`tot`**, **`llm-guided-tot`**, hand-rolled on `asyncio` | rollouts, UCB1 selection, reward-model backup |
| Self-refine loops | hand-rolled with a verifier + `max_iter` cap | `while not verified: critique → revise`; verifier can be another LLM call, a unit test, or a domain oracle |
| RLAIF / Constitutional AI | **`trl`** (PPO / DPO / KTO), **`trlx`**, **`axolotl`** | `PPOTrainer`, `DPOTrainer`, `KTOTrainer`, AI-as-judge labelers |
| Autonomous research loop | **`autogen`**, **`crewai`**, **`langgraph`**, **`smolagents`** | multi-agent loops, tool-calling, memory + planner + critic roles, cyclical state graphs |
| Evolutionary prompt-program search | **AlphaEvolve** (DeepMind, paper), **OpenEvolve** (community) | population of prompt-programs, LLM-driven mutation, fitness = downstream metric |
| Evaluation harness (needed by all families) | **`lm-eval-harness`**, **`promptfoo`**, **`deepeval`**, **`braintrust`** | task-metric glue; A/B evaluation; regression tests for prompt programs |

---

## Minimal pattern — DSPy + MIPROv2 optimizer

```python
import dspy

# Configure the LM backend once
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class AnswerQuestion(dspy.Signature):
    """Given a question and relevant context, produce a short answer."""
    context  = dspy.InputField()
    question = dspy.InputField()
    answer   = dspy.OutputField(desc="short answer")

class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(AnswerQuestion)
    def forward(self, context, question):
        return self.predict(context=context, question=question)

# Evaluate + compile with MIPROv2 against a small labelled trainset
metric  = lambda ex, pred, trace=None: ex.answer.lower() in pred.answer.lower()
program = QAProgram()
compiled = dspy.MIPROv2(metric=metric, auto="medium").compile(
    program, trainset=trainset, valset=valset,
)
# `compiled` is a drop-in replacement with optimized instructions + few-shots
```

## Minimal pattern — Self-Refine loop

```python
def self_refine(generate_fn, critique_fn, is_good, max_iter=5):
    """Generic Self-Refine: generate → critique → revise until verified."""
    output = generate_fn()
    for _ in range(max_iter):
        if is_good(output):
            return output
        feedback = critique_fn(output)
        output = generate_fn(feedback=feedback)
    return output                         # return best effort after budget
```

The verifier (`is_good`) can be: a unit-test pass/fail, a reward-model score above a threshold, a rubric scored by another LLM call, or a domain oracle (e.g., code executes without error, proof type-checks, JSON matches a schema).

---

## When to reach for which family

| Symptom / constraint | Family |
|---|---|
| Prompt-tuning friction eats a huge fraction of dev time | **Programmatic prompts** (DSPy) |
| A verifier exists (unit tests, schema check, judge LLM) and inference budget can grow | **Inference-time search** (best-of-N) or **Self-Refine** |
| The task has multi-step reasoning with branching | **Tree-of-thoughts / MCTS** |
| Human preference labels are expensive; you have an AI judge you trust | **RLAIF / DPO** |
| You need the model to safely reject / transform adversarial inputs | **Constitutional AI** (principle-based critiques) |
| The task is long-horizon research / coding with many tools | **Autonomous loop** (AutoGen / CrewAI / LangGraph) |
| Code / math benchmark with clear metric, willing to burn GPU | **Evolutionary prompt-program search** (AlphaEvolve / OpenEvolve) |

---

## Depth companions

This skill is the taxonomy-level orchestrator. For in-depth patterns on the two most mature families, see the dedicated depth-skills:

- **[DSPy Basics](../dspy-basics/SKILL.md)** — Signatures, Modules, ChainOfThought / ReAct / Retrieve, MIPROv2 / BootstrapFewShot / COPRO / BetterTogether optimizers, evaluation design.
- **[RLAIF Training](../rlaif-training/SKILL.md)** — DPO / KTO / IPO / PPO with `trl`, Constitutional AI critique loops, preference generation (LLM-as-judge / constitutional / self-rewarding), reward-hacking failure modes.

The other two families (inference-time scaling and autonomous research loops) are covered in-line above; they do not yet have enough stable API surface to warrant dedicated depth skills.

## Composition with neighboring skills

- **LLM evaluation** — every self-improving system needs a metric; design the evaluation harness before the optimizer. See `llm-evaluation`.
- **Prompt engineering** — the atomic-level technique side; DSPy / RLAIF sit on top of it. See the `prompt-engineer` agent.
- **LLM applications** — RAG, agentic, multi-modal patterns that self-improving AI enhances. See `llm-application-patterns`.
- **Agent systems** — multi-agent coordination primitives that autonomous research loops build on. See `multi-agent-coordination`.
- **Research methodology** — applying self-improving AI to literature review, hypothesis generation, experiment design. See `research-methodology` in the `research-suite` plugin (or its hub `research-practice`).

---

## Common pitfalls

- **Reward hacking** — the optimizer finds a way to maximize the metric without solving the underlying task. Always keep a held-out metric the optimizer never sees.
- **Verifier drift** — an LLM-as-judge verifier changes behaviour between runs; pin the verifier model or use a cheaper, more deterministic check where possible.
- **Self-Refine plateaus** — many tasks stop improving after 2–3 refine rounds. Measure the marginal-benefit curve before setting `max_iter`.
- **RLAIF reward gaming** — DPO is more stable than PPO on AI-generated labels; prefer DPO / KTO when starting.
- **Autonomous-loop cost blowup** — multi-agent loops can burn enormous token budgets in a single task. Set a hard turn and token cap per session.

---

## Checklist

- [ ] Built a programmatic evaluation harness before reaching for any optimizer
- [ ] Verified the metric is hard to game (held-out variant, adversarial examples)
- [ ] Picked the family from the symptom table, not from enthusiasm about the newest paper
- [ ] For DSPy / prompt-program optimization: chose `MIPROv2` (or `COPRO` for cheaper runs) with a labelled trainset ≥ 20 examples
- [ ] For Self-Refine: measured the marginal-benefit curve and capped `max_iter`
- [ ] For RLAIF: started with DPO / KTO over PPO; kept a reference model for regularization
- [ ] For autonomous research loops: set hard turn / token caps; saved session traces for audit
- [ ] Logged all optimization runs with `lm-eval-harness` / `promptfoo` / `braintrust` for regression tracking
