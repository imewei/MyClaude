---
name: rlaif-training
description: Train LLMs with AI-generated preference labels via RLAIF, Constitutional AI, DPO, KTO, and PPO using the trl library. Covers AI-as-judge label generation, constitutional critique loops, reward model training, DPO vs KTO vs PPO trade-offs, and common failure modes (reward hacking, preference collapse, reference-model drift). Use when human preference labels are too expensive, when you have a trusted judge LLM, when a constitution (principle list) can express the desired behaviour, or when fine-tuning an open-weight model on AI-generated pairs. Use proactively when the user mentions RLAIF, Constitutional AI, DPO, KTO, IPO, PPO, trl, trlx, reward model, AI-as-judge, preference optimization, or self-rewarding.
---

# RLAIF and Preference Optimization

The depth-skill companion to `self-improving-ai` for the **Constitutional AI / RLAIF** family. Replaces the expensive human-labelled pair stage of RLHF with AI-generated preference labels, either from a trusted judge LLM or from a constitution (a list of principles the model critiques itself against).

## Expert Agents

- **`ai-engineer`** (primary): Production LLM applications, fine-tuning pipelines, preference optimization.
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`
- **`ml-expert`** (secondary): Reward model training, optimizer selection, training-loop debugging.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`

---

## The pipeline

| Stage | Role |
|---|---|
| **1. SFT** | Supervised fine-tune on instruction / response pairs. Baseline model for everything that follows. |
| **2. Preference generation** | For each prompt, sample two (or more) responses from the SFT model. Label which is better — via human (RLHF), AI judge (RLAIF), or constitutional critique. |
| **3. Reward modelling** *(PPO only)* | Train a reward model on the preference pairs via pairwise logistic loss (Bradley-Terry). |
| **4. Policy optimization** | Update the policy with PPO (reward-model-driven), or directly on the preference pairs with DPO / KTO / IPO. |
| **5. Evaluation** | Measure the policy against held-out prompts with an eval harness and a separate judge. |

**Constitutional AI** inserts a critique step between stages 1 and 2: the SFT model is asked to critique its own response against a list of principles ("is this helpful, is this harmless, is this honest?"), revise, and the revised response becomes the preferred one in the pair.

---

## Tool catalog (`trl`)

Hugging Face `trl` is the production reference — `trlx` (CarperAI) is an alternative with more RL-leaning abstractions.

| Trainer | Loss | Needs reward model? | Use when |
|---|---|---|---|
| **`PPOTrainer`** | Clipped surrogate policy gradient against a reward model | **Yes** | Classic RLHF; maximum flexibility; most unstable |
| **`DPOTrainer`** | Direct preference optimization (closed-form reward-free loss) | No | Default for most teams now — stable, simple, no reward-model training stage |
| **`KTOTrainer`** | Kahneman-Tversky Optimization (unpaired binary preferences) | No | When you only have "this response is good / bad", not pairs |
| **`IPOTrainer`** | Identity Preference Optimization (avoids DPO's overconfidence bias) | No | When DPO over-drives preferences beyond the training margin |
| **`ORPOTrainer`** | Odd-ratio preference optimization + SFT loss in one step | No | Drop-in replacement for SFT when preference pairs are available early |
| **`CPOTrainer`** | Contrastive preference optimization | No | Translation / generation tasks with a quality delta signal |
| **`RewardTrainer`** | Pairwise logistic (Bradley-Terry) on preference pairs | — | Build the reward model for later PPO / best-of-N |

**Prefer DPO (or KTO / ORPO) over PPO when starting.** PPO requires a reward model, is sensitive to hyperparameter choice, and tends to collapse on distributionally-shifted rollouts. DPO trains directly on preference pairs with one loss and is dramatically more stable.

---

## Minimal pattern — DPO with trl

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_id  = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
policy    = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)     # frozen reference

# Preference dataset shape: {prompt, chosen, rejected}
dataset = Dataset.from_list([
    {"prompt": p, "chosen": c, "rejected": r}
    for p, c, r in preference_pairs
])

config = DPOConfig(
    output_dir            = "./dpo-run",
    beta                  = 0.1,              # KL weight vs reference model
    learning_rate         = 5e-7,
    num_train_epochs      = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    bf16                  = True,
)

trainer = DPOTrainer(
    model          = policy,
    ref_model      = ref_model,
    args           = config,
    train_dataset  = dataset,
    processing_class = tokenizer,
)
trainer.train()
trainer.save_model("./dpo-run/final")
```

The `beta` parameter controls how far the policy can drift from the reference model. Lower `beta` → more aggressive preference following but higher risk of policy collapse. `0.1` is a safe starting point.

---

## Generating preferences without humans

Three common patterns, in order of increasing supervision cost:

### A. LLM-as-judge (cheapest)

```python
JUDGE_PROMPT = """Given a user question and two candidate responses, return "A", "B",
or "tie" based on which is more helpful, honest, and harmless.

Question: {prompt}

Response A: {response_a}
Response B: {response_b}

Your verdict (A / B / tie):"""

def llm_judge(prompt, a, b, judge_llm):
    verdict = judge_llm(JUDGE_PROMPT.format(prompt=prompt, response_a=a, response_b=b))
    if "A" in verdict: return a, b
    if "B" in verdict: return b, a
    return None                                # skip ties
```

**Always** use a different / stronger model as the judge than the policy being trained. Self-judging leads to collapse — the policy learns to game its own idiosyncrasies.

### B. Constitutional AI (Anthropic pattern)

```python
CONSTITUTION = [
    "Responses should be helpful and directly answer the user's question.",
    "Responses should avoid harmful, biased, or deceptive content.",
    "Responses should acknowledge uncertainty when facts are unclear.",
]

CRITIQUE = """Review your response against this principle:
{principle}

Original response: {response}

Does the response violate this principle? If yes, write a revised response."""

def constitutional_pair(prompt, model):
    raw = model(prompt)
    revised = raw
    for principle in CONSTITUTION:
        out = model(CRITIQUE.format(principle=principle, response=revised))
        if "revised:" in out.lower():
            revised = out.split("revised:", 1)[1].strip()
    return {"prompt": prompt, "chosen": revised, "rejected": raw}
```

### C. Self-rewarding (Meta pattern)

Use the same model to both generate candidate responses and score them — but with a separate prompt for scoring. Periodically update the scoring prompt with the best responses found so far. Works when the scoring signal is clear (code passes tests, math is correct, schema matches) and the base model is strong enough to act as a reasonable judge.

---

## Reward hacking — the central failure mode

The policy optimizer is a search process. Any loophole in the reward signal will be found. Common failure modes:

| Symptom | Cause | Fix |
|---|---|---|
| Responses get longer and more hedged | Judge rewards verbosity | Penalize length explicitly; use paired margin judges |
| Model starts refusing everything | Constitution overweights harmlessness | Balance with helpfulness principle; include positive examples |
| Judge score increases but held-out eval drops | Policy overfit the judge's quirks | Rotate judges; use ensemble of judges |
| KL to reference model explodes | `beta` too low in DPO | Raise `beta`; check sampled rollouts for distributional shift |
| Policy collapses to a single response | PPO reward model is piecewise-constant | Switch to DPO / KTO; use a calibrated reward model |
| Training loss → 0 but eval unchanged | Preference pairs are trivial to separate | Harder-negative mining; more diverse sampling temperature |

**The single best defence is a held-out evaluation on prompts the optimizer never sees, scored by a separate judge.**

---

## Composition with neighboring skills

- **Self-improving AI** — parent orchestrator; RLAIF is the Constitutional-AI family. See `self-improving-ai`.
- **DSPy basics** — peer depth-skill covering the programmatic-prompt family. See `dspy-basics`.
- **LLM evaluation** — build the held-out eval + judge **before** the training loop. See `llm-evaluation`.
- **ML engineering production** — serving, quantization, and deployment for the trained policy. See `ml-engineering-production`.
- **Experiment tracking** — DPO runs need W&B / MLflow tracking for `loss/chosen`, `loss/rejected`, `rewards/margins`. See `experiment-tracking`.

---

## Checklist

- [ ] Chose DPO / KTO / IPO / ORPO as the default; only fell back to PPO with a specific reason
- [ ] Built a held-out eval set with a different judge before starting training
- [ ] Generated preference pairs with a distinct / stronger model as judge (never self-judging)
- [ ] Set `beta` to a safe starting value (~0.1 for DPO); monitored KL to reference model
- [ ] Checked preference-pair difficulty — not all pairs should be trivially separable
- [ ] Logged `rewards/margins`, `loss/chosen`, `loss/rejected` per step to catch collapse early
- [ ] Ran the policy against held-out prompts with a separate judge and compared to the reference model
- [ ] Rotated judges or used an ensemble to detect judge-specific gaming
- [ ] Documented the constitution (if Constitutional AI) alongside the training run for reproducibility
