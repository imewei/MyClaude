---
name: self-improving-agents
description: Design closed-loop self-improvement workflows for LLM agents using reflection-driven refinement, automatic prompt optimization (DSPy, TextGrad), self-consistency ensembles, and constitutional self-critique. Use when building agents that iteratively improve their own prompts, reasoning chains, or policies from their own trajectories, when the user mentions self-improving, meta-prompting, auto-prompt, DSPy, TextGrad, PromptBreeder, self-critique loops, or RLAIF-style alignment. Pair with reflection-framework for the underlying meta-cognitive audit and with memory-system-patterns for the failure-trace retrieval that makes improvement learned rather than per-turn.
---

# Self-Improving Agents

Workflows where an LLM agent *modifies its own configuration* — prompts, reasoning strategies, tool-use policies, or even model weights — based on feedback from its own outputs. The closed loop distinguishes self-improvement from one-shot reflection: the improved artifact is retained and reused.

## Expert Agent

For self-improving agent design, prompt-optimization programming models, and reflective loop architecture, delegate to:

- **`context-specialist`**: LLM application patterns, prompt optimization, loop architecture, safety constraints.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`
- **`reasoning-engine`** (secondary): Reflection, bias audit, reasoning-quality scoring that feeds the improvement loop.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`

---

## Two improvement regimes — pick the right one

| Regime | What it modifies | When to use |
|--------|-----------------|-------------|
| **Intra-task reflection** | Nothing persists — agent re-drafts within a single turn | Cheap to try, no training data, no rollback risk |
| **Learned improvement** | Prompts / few-shots / chain-of-thought templates persisted across runs | You have a labeled eval set and can iterate offline |
| **RLAIF / preference optimization** | Model weights via RL from AI feedback | You have compute budget and deployment control over the base model |

**Default**: start with intra-task reflection. Graduate to learned improvement only after you have a stable eval set that pins down what "improvement" means.

---

## Pattern 1 — Reflection-Refine-Validate loop (intra-task)

The simplest self-improvement pattern: draft → critique → refine → verify. See `reflection-framework` for the critique rubrics.

```
draft  ← agent(task, prompt)
crit   ← critic(draft, rubric)
if crit.score < threshold:
    refined ← agent(task, prompt + draft + crit.feedback)
    verify  ← validator(refined)
    if not verify.pass:
        escalate to human
    else:
        return refined
else:
    return draft
```

Cost: 2–3× per turn vs. a single draft. Benefit: catches self-contradictions, unsupported claims, missed requirements. **Do not run reflection on every turn** — reserve for high-stakes outputs where the cost is justified (see `reflection-framework` for the trigger criteria).

---

## Pattern 2 — Self-consistency ensembles

Sample N independent reasoning chains, then aggregate (majority vote for classification; self-critique over the set for free-form). Drastically reduces variance without modifying any prompt.

```python
def self_consistency(agent, task, n=5, temperature=0.9):
    chains = [agent(task, temperature=temperature) for _ in range(n)]
    # Classification: majority vote on final answer
    # Free-form:      feed all chains to a critic that selects / synthesizes
    return aggregate(chains)
```

When to use: factual / numerical / logical tasks where the right answer is unique. Avoid for creative tasks where diversity is the point.

---

## Pattern 3 — Automatic prompt optimization (DSPy)

`dspy` treats prompts as *programs* and optimizes them by feeding labeled examples through a metric and searching for the best few-shot selection, instruction phrasing, or chain-of-thought decomposition. Unlike hand-rolled RLAIF, it needs no gradient access — it only calls the inference API.

```python
import dspy

class Fact(dspy.Signature):
    question: str = dspy.InputField()
    answer:   str = dspy.OutputField()
    source:   str = dspy.OutputField()

class FactPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(Fact)
    def forward(self, question):
        return self.qa(question=question)

def metric(gold, pred, trace=None):
    return int(pred.answer.strip().lower() == gold.answer.strip().lower())

from dspy.teleprompt import MIPROv2
compiled = MIPROv2(metric=metric, auto="medium").compile(FactPipeline(), trainset=train_examples)
compiled.save("fact_pipeline.json")
```

Key optimizers to know:

| Optimizer | What it tunes | When to reach for it |
|-----------|---------------|----------------------|
| `BootstrapFewShot` | Few-shot demonstrations | Starting point — cheap, no instruction rewriting |
| `BootstrapFewShotWithRandomSearch` | Few-shot demos via random search | When the first bootstrap under-fits |
| `COPRO` | Instruction strings | Instruction phrasing is the bottleneck, few-shots already fine |
| `MIPRO / MIPROv2` | Joint few-shot + instruction optimization via a Bayesian surrogate | Production default — best-quality-per-API-call, handles non-trivial metrics |
| `BootstrapFinetune` | Distills optimized pipeline into a smaller model | Deployment — shrink the compiled prompt into a fine-tuned student |

DSPy's key insight: the compiled artifact is a *program*, not a prompt string. You can recompile when the base model changes without hand-rewriting prompts.

---

## Pattern 4 — Textual-gradient optimization (TextGrad)

`textgrad` computes a "textual gradient" — a natural-language critique — and backpropagates it through a computation graph of LLM calls. Useful when the feedback signal is itself a string ("this answer was too long", "the reasoning skipped step 3"), not a scalar metric.

```python
import textgrad as tg

engine = tg.get_engine("claude-sonnet-4-6")
tg.set_backward_engine(engine, override=True)

sys_prompt = tg.Variable("You are a careful scientific reasoner.",
                          requires_grad=True, role_description="system prompt")

def forward(question, answer_gold):
    pred    = tg.BlackboxLLM(engine, system_prompt=sys_prompt)(
                  tg.Variable(question, requires_grad=False, role_description="question"))
    loss_fn = tg.TextLoss("Critique vs gold answer — specific and terse.", engine=engine)
    return loss_fn(pred, tg.Variable(answer_gold, requires_grad=False, role_description="gold"))

optimizer = tg.TGD(parameters=[sys_prompt])
for question, gold in train_pairs:
    loss = forward(question, gold); loss.backward()
    optimizer.step(); optimizer.zero_grad()

print(sys_prompt.value)   # the evolved prompt
```

TextGrad is most effective when the "loss" is qualitative (style, coverage, coherence) rather than exact-match accuracy — that's where scalar metrics fail and DSPy's discrete search is awkward.

---

## Pattern 5 — Evolutionary prompt search

`PromptBreeder` and `EvoPrompt` evolve a population of prompts via LLM-authored mutations and crossovers, selecting survivors on a metric. Useful when the search space is large and DSPy's Bayesian optimizer gets stuck in a local optimum.

```
initialize population of N prompts
for generation g in 1..G:
    for each prompt p:
        score[p] ← mean_metric(p, eval_set)
    select top-k survivors
    mutate: ask an LLM to rewrite each survivor with a tweak strategy
    crossover: ask an LLM to blend pairs of survivors
    population ← survivors ∪ mutations ∪ crossovers
return argmax(score)
```

Cost is O(G · N · eval_size) API calls — expensive, so reserve for the one-time compile of a production prompt, not for per-request optimization.

---

## Pattern 6 — Constitutional self-critique (safety + quality)

Anthropic's Constitutional AI pattern: the agent critiques its own output against a written list of principles ("is this helpful?", "is this honest?", "does this respect the user's autonomy?") and rewrites if any principle is violated. Differs from reflection-framework in two ways:

1. The critique criteria are **explicit text principles**, not domain-agnostic rubrics.
2. The loop is **self-applied** — same model critiques itself, no separate judge.

```python
PRINCIPLES = [
    "The response must be grounded in the provided context or acknowledge uncertainty.",
    "The response must not exceed 500 words unless explicitly requested.",
    "Code examples must be runnable without external setup.",
]

def constitutional_refine(agent, task):
    draft = agent(task)
    for principle in PRINCIPLES:
        critique = agent(f"Does the following response satisfy: '{principle}'?\n\n{draft}")
        if "no" in critique.lower()[:20]:
            draft = agent(f"Rewrite this response to satisfy '{principle}':\n\n{draft}")
    return draft
```

Pairs naturally with `safety-guardrails` (the hard-constraint enforcement layer) — constitutional critique is the soft layer, guardrails are the hard layer.

---

## Memory-augmented improvement

Any of the above patterns can be *amortized* by storing failure traces and retrieving them for future similar tasks. See `memory-system-patterns` for the retrieval layer. The loop becomes:

```
on_task(task):
    past_failures ← memory.retrieve(task)
    prompt        ← base_prompt + past_failures
    draft         ← agent(task, prompt)
    if critic(draft) < threshold:
        memory.store(task, failure=draft, fix=…)
    return draft
```

This turns per-turn reflection into learned improvement without any explicit training — the agent accumulates a corrective trajectory library over time.

---

## Composition with neighboring skills

- **Reflection framework** — the meta-cognitive audit rubrics that drive Patterns 1 and 6. See `reflection-framework`.
- **Prompt engineering patterns** — the production-grade prompt templates that DSPy / TextGrad optimize. See `prompt-engineering-patterns`.
- **Memory system patterns** — the retrieval layer for amortized learned improvement. See `memory-system-patterns`.
- **Safety guardrails** — the hard-constraint layer that pairs with soft constitutional critique. See `safety-guardrails`.
- **Reasoning frameworks** — the CoT / First-Principles decomposition that reflection scores. See `reasoning-frameworks`.
- **thinkfirst** — upstream interview-first clarification; run once before compiling any optimized prompt. See `thinkfirst`.

---

## Checklist

- [ ] Defined a concrete metric (scalar or text-loss) before starting an optimization loop
- [ ] Held out a *separate* eval set from the training set — never compile on the eval set
- [ ] Picked the simplest pattern that fits (reflection < self-consistency < DSPy < TextGrad < RLAIF)
- [ ] Capped the loop cost (`max_iters`, `max_api_calls`) — self-improvement is easy to run forever
- [ ] Stored the compiled artifact (`compiled.save()`, evolved prompt string) and versioned it alongside the base model
- [ ] Re-evaluated after any base-model change — compiled prompts are model-specific
- [ ] For production: paired soft constitutional critique with hard `safety-guardrails` enforcement
- [ ] For learned improvement: logged failure traces to memory so future runs benefit
