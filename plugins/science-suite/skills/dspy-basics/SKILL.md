---
name: dspy-basics
description: Build programmatic prompt pipelines with DSPy — Signatures, Modules, ChainOfThought, ReAct, Retrieve, and optimizers (MIPROv2, BootstrapFewShot, COPRO, BetterTogether). Use when hand-tuning prompts is eating dev time, when you need a labelled-metric-driven optimizer to improve prompts, when the pipeline has multiple LLM calls that should compose as typed modules, or when you need deterministic regression tests on prompt programs. Use proactively when the user mentions DSPy, dspy.Signature, dspy.Module, ChainOfThought, ReAct, MIPRO, MIPROv2, BootstrapFewShot, COPRO, BetterTogether, programmatic prompts, or prompt compilation.
---

# DSPy Basics

The depth-skill companion to `self-improving-ai` for the **programmatic prompt optimization** family. Stanford's DSPy treats prompts as compilable programs: you declare typed `Signature`s, compose `Module`s, and let an optimizer search over prompt wording, few-shot exemplars, and module structure to maximize a downstream metric on a labelled trainset.

## Expert Agents

- **`ai-engineer`** (primary): Production LLM applications, prompt programs, tool calling.
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`
- **`prompt-engineer`** (secondary): Prompt technique, chain-of-thought, evaluation design.
  - *Location*: `plugins/science-suite/agents/prompt-engineer.md`

---

## Core abstractions

| Concept | Role |
|---|---|
| **`Signature`** | Typed I/O contract for an LLM call — inputs, outputs, and the task docstring. The docstring becomes part of the prompt. |
| **`Module`** | A callable wrapping one or more Signatures. The optimizer replaces the prompts inside `Module`s, not the code around them. |
| **`Predict`** | The base Module — one LLM call per Signature, no reasoning scaffold. |
| **`ChainOfThought`** | Adds a `rationale` field and prompts the model to think before answering. |
| **`ReAct`** | Thought-action-observation loop with tool calls. |
| **`Retrieve`** | Retriever primitive that fits into a pipeline alongside Predict / CoT / ReAct. |
| **`Optimizer`** | A compiler that improves a Module against a metric on a labelled trainset (`MIPROv2`, `BootstrapFewShot`, `COPRO`, `BetterTogether`). |
| **`Evaluate`** | Metric harness — runs the Module over a devset and reports aggregate scores. |

---

## Minimal pattern — Signature + ChainOfThought

```python
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class AnswerWithContext(dspy.Signature):
    """Answer the question using only the provided context."""
    context  = dspy.InputField(desc="relevant passages")
    question = dspy.InputField()
    answer   = dspy.OutputField(desc="short factual answer")

class RAGProgram(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = retriever
        self.predict  = dspy.ChainOfThought(AnswerWithContext)

    def forward(self, question):
        passages = self.retrieve(question)
        return self.predict(context=passages, question=question)

program = RAGProgram(my_retriever)
prediction = program(question="When was the transformer paper published?")
print(prediction.rationale, prediction.answer)
```

The `Signature` docstring and field descriptions are part of the prompt. The optimizer will later rewrite the instruction, swap in better few-shot exemplars, or add a rationale template — without touching the `forward` method.

---

## Compiling with MIPROv2

```python
from dspy.evaluate import Evaluate

def em_metric(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

trainset = [dspy.Example(question=q, answer=a).with_inputs("question")
            for q, a in train_pairs]
valset   = [dspy.Example(question=q, answer=a).with_inputs("question")
            for q, a in val_pairs]

optimizer = dspy.MIPROv2(
    metric          = em_metric,
    auto            = "medium",      # "light" / "medium" / "heavy" — budget
    num_threads     = 8,
)
compiled = optimizer.compile(program, trainset=trainset, valset=valset)

evaluate = Evaluate(devset=valset, metric=em_metric, num_threads=8)
print(evaluate(compiled))
compiled.save("rag_compiled.json")         # persistable, version-controllable
```

**What MIPROv2 actually does**: it proposes candidate instructions and few-shot exemplars for each predictor in the program, then uses Bayesian optimization over the candidate space to find a combination that maximizes `metric` on the validation set. Heavier budgets (`auto="heavy"`) run more proposals.

---

## Optimizer selection

| Optimizer | When |
|---|---|
| **`BootstrapFewShot`** | Cheapest; bootstraps few-shot examples by running the uncompiled program on the trainset and keeping the ones that pass the metric |
| **`BootstrapFewShotWithRandomSearch`** | Adds random search over the few-shot exemplar set — still cheap, often beats `BootstrapFewShot` |
| **`MIPROv2`** | Bayesian optimization over instructions + exemplars + demonstrations; the current state-of-the-art general-purpose optimizer |
| **`COPRO`** | Cheap instruction-only optimizer; best when the few-shot budget is already tight |
| **`BetterTogether`** | Combines instruction tuning with module-structure search — picks the best of several drop-in variants |
| **`KNNFewShot`** | Nearest-neighbor few-shot retrieval at inference time; good when the trainset is large and diverse |

Start with `BootstrapFewShot` for smoke tests, move to `MIPROv2(auto="light")` once the metric is trusted, escalate to `auto="medium"` or `"heavy"` only when the dev-set gain justifies the token spend.

---

## ReAct and tool use

```python
class DefinitionLookup(dspy.Tool):
    """Look up a term in a curated glossary."""
    def __init__(self, glossary: dict[str, str]):
        super().__init__(name="define", desc="look up a technical term")
        self.glossary = glossary

    def __call__(self, term: str) -> str:
        return self.glossary.get(term.lower(), f"no definition for '{term}'")

react = dspy.ReAct(
    signature="question -> answer",
    tools=[DefinitionLookup(glossary={"nuts": "No-U-Turn Sampler"})],
    max_iters=5,
)
result = react(question="What does NUTS stand for in MCMC?")
```

DSPy `ReAct` interleaves thought → tool-call → observation automatically; the tool catalog and a `max_iters` cap are the two main tuning knobs. **Never** wire a tool that accepts free-form code (`eval`, `exec`, raw shell) from the LM without a sandbox — treat LM tool calls as untrusted input.

---

## Evaluation design is the hardest part

The optimizer is only as good as the metric. Before reaching for any compiler:

1. **Build the devset first** — hand-label 20–100 examples. More is better, but 20 is enough to start.
2. **Pick a metric that's hard to game** — exact-match is fine for QA; for open-ended tasks, use an LLM-judge metric (`dspy.evaluate.metrics.answer_exact_match` or a custom LM-as-judge).
3. **Hold out a test set the optimizer never sees** — measure generalization on this set after compilation. A large train↔test gap means the optimizer overfit the metric.
4. **Use `dspy.Evaluate` not hand-rolled loops** — it handles parallelism, caching, and traces consistently.

---

## Common pitfalls

- **Metric rewards verbosity** — if your metric is token F1 or substring match, the optimizer will produce long answers. Penalize length explicitly.
- **Few-shot exemplars from the trainset leak into eval** — DSPy uses distinct train / val / test splits; never pass the test set as the trainset.
- **LM caching across optimizer runs** — `dspy.settings.configure(cache=True)` is often a win but hides non-determinism bugs; turn it off during metric design.
- **Compiled program is a JSON blob, not Python code** — save / load via `compiled.save(...)` and version-control the JSON, not the uncompiled `Module`.
- **`auto="heavy"` can burn $$$** — budget caps: most useful with smaller LMs (`openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`) or with a local `dspy.LM("ollama/qwen2.5-coder:7b")`.

---

## Composition with neighboring skills

- **Self-improving AI** — parent orchestrator; DSPy is the programmatic-prompt family. See `self-improving-ai`.
- **RLAIF training** — peer depth-skill covering the Constitutional AI / RLAIF family. See `rlaif-training`.
- **LLM evaluation** — build the metric harness before reaching for any compiler. See `llm-evaluation`.
- **RAG implementation** — DSPy's `Retrieve` primitive drops into RAG pipelines. See `rag-implementation`.
- **Prompt engineer** — atomic-level technique side; DSPy sits on top of it. See the `prompt-engineer` agent.

---

## Checklist

- [ ] Wrote the `Signature` docstring as if the final prompt must be self-explanatory
- [ ] Split data into train / val / test before writing any optimizer call
- [ ] Built a held-out test set the optimizer never sees
- [ ] Chose the metric deliberately and checked a few examples for gameability
- [ ] Started with `BootstrapFewShot` for smoke test, escalated to `MIPROv2(auto="light")` only after the metric was trusted
- [ ] Measured train→test generalization gap after compilation
- [ ] Saved the compiled program as JSON and versioned it
- [ ] Capped `max_iters` for `ReAct` to prevent runaway tool loops
- [ ] Disabled LM caching during metric-design sessions to catch non-determinism
- [ ] Treated all LM tool-call arguments as untrusted input (no `eval`, no raw shell)
