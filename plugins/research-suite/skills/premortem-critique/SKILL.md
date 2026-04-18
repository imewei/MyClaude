---
name: premortem-critique
description: Stage 8 of the research-spark pipeline. Writes a failure narrative for the full plan, clusters failure modes by root cause, identifies the cheapest early signal for each cluster, and loops back to insert those signals as concrete milestones in the Stage 7 plan. Also generates a one-page summary for external critique and runs a simulated reviewer critique across multiple archetypes. Triggers on phrases like "do a premortem", "write the failure narrative", "stress-test the plan", "what could go wrong", "prepare a one-page summary for external review", "run a simulated reviewer on the full proposal", "what would kill this project", or after Stage 7 completes. The success criterion for this stage is not that it produces a good document; it is that the Stage 7 plan is different after running it. If nothing changed upstream, the premortem was not serious.
---

# premortem-critique

Stage 8. The critical feedback arc. Stress-tests the plan before resources are committed.

## Why this stage exists

The usual way projects fail is not that the plan was bad but that the plan was never seriously critiqued until reviewers saw it. Premortem is internal critique before the external reviewers get there. More importantly, it is the one stage whose output is measured by what it changes upstream.

The success criterion is not that this document is well-written. It is that the Stage 7 plan is different after running this stage. If the plan is unchanged, either the plan was already optimal (rare) or the premortem was shallow. The loop back to Stage 7 is where the value lives; everything else is preparation for it.

## Prerequisites

Every prior artifact exists: `01_spark.md` through `07_plan.md`, the predicted observable file, the Stage 6 code repo.

## Workflow

### 1. Write the failure narrative (past tense)

Assume the project has failed in two years. Not "is having difficulties." Failed. Paper was not written, or was written and rejected and never revised, or was written and published and turned out to be wrong. Write the narrative of how it failed.

Past tense, specific months, named incidents. "We ran the measurement in month 7 and saw no gap collapse" is a premortem sentence. "We might run the measurement and might see nothing" is not.

Write 2-3 narratives, each exploring a different failure mode. See `templates/premortem_template.md` for structure and a worked example.

The tense discipline matters because future-conditional sentences let the imagination hedge. Past tense forces specificity: which month, which sample, which measurement, which person's reasoning failed at which decision point.

### 2. Cluster failure modes by root cause

From the narratives, extract root categories:

- **Theoretical error.** Framework was wrong. Assumption was hidden. Dropped term mattered. Symmetry was violated.
- **Measurement artifact.** Signal seen was not the signal predicted because of an instrumental effect.
- **Scale or resource limit.** Plan could not be executed at the required scope. Ran out of beamtime, money, or collaborator bandwidth.
- **External dependency.** A collaborator did not deliver. A sample did not arrive. A library changed and broke the pipeline.
- **Timeline mismatch.** Physics was right, measurement was possible, but the sequence did not converge before funding ran out or someone else published.

Each cluster has distinct mitigations. Conflating them produces generic "we will be careful" responses that help nobody.

### 3. Identify the cheapest early signal per cluster

For each cluster, find the cheapest diagnostic that in month 1 or 2 would reveal the cluster's failure was brewing.

Patterns:
- *Theoretical error* early signal: a limit check that was rationalized rather than investigated, or an OOD test for a gray-box component that was deferred
- *Measurement artifact* early signal: a control sample producing a signal the theory forbids
- *Scale / resource limit* early signal: first week of beamtime producing data at below-planned rate
- *External dependency* early signal: first collaborator milestone running late
- *Timeline* early signal: early decision points slipping by more than 2 weeks each

### 4. Loop back to Stage 7

Most important step. Each early signal becomes a concrete milestone inserted into `07_plan.md`.

- Open `07_plan.md`
- For each early signal, write a milestone with a date and a pass/fail criterion
- Add to the timeline and the risk register
- Update the artifact

Then return to `08_premortem.md` and record what was inserted. The record is evidence that the premortem caused a change. If no changes were made, state that explicitly with a reason; "the plan was already optimal" needs justification that names which risks are already covered and how.

### 5. Generate the one-page summary

Use `../_research-commons/templates/onepage.md`. A single-page summary for an external skeptical collaborator. This is what the user will actually send someone outside the group for independent critique.

It is not a condensed proposal. It is designed to elicit critique: states the claim, the test, the risks, and the ask with no hedging. Hedged one-pagers produce vague feedback.

### 6. Simulated reviewer critique

`references/simulated_reviewer_prompts.md` provides four archetypes: theorist, experimentalist, applications-focused, statistician. Each attacks a different surface of the plan.

For a first premortem, run at least theorist and experimentalist. For mature plans, run all four. For plans destined for a specific reviewing body, emulate that body's known composition.

Each archetype produces 3-5 paragraphs of critique; you respond paragraph by paragraph with accept (revise the plan), acknowledge (explain why the current plan is defensible anyway), or rebut (cite specific evidence). Critique without response is a flag; either address it or mark it as residual unresolved risk.

If an archetype's critique reads as "interesting work, a few questions", the calibration is off. Re-prompt with the harsher tone in the prompts file.

### 7. Write, lint, mark complete

`artifacts/08_premortem.md` structure: failure narratives; cluster analysis; early signals per cluster; milestones inserted into Stage 7 with explicit diff (before → after); one-page summary; simulated reviewer critiques with responses; residual unresolved risks acknowledged openly. Style-lint. Update `_state.yaml` to mark the stack complete.

## Failure modes worth naming

- **Writing the narrative in future conditional tense.** "We might encounter..." produces generic warnings. Past tense forces specificity.
- **A cluster analysis that puts everything in "theoretical error."** Most real project failures are a mix of causes interacting; if one cluster is doing all the work, the narrative was too narrow.
- **Skipping the loop back to Stage 7.** A premortem that does not change the plan is not a premortem; it is a writing exercise. If Step 4 produced no milestones, Steps 1-3 were too soft.
- **Soft simulated reviewers.** The archetypes are tuned adversarial. If the output is soft, re-prompt. Skepticism from a simulated reviewer is cheap; skepticism from a real one after the grant is submitted is not.

## Templates and references

- `templates/premortem_template.md`: failure-narrative structure with a worked example
- `references/simulated_reviewer_prompts.md`: four archetype prompts with calibration guidance
- Shared: `../_research-commons/templates/onepage.md`, `../_research-commons/templates/reviewer2_persona.md`

## What happens after Stage 8

This is the last pipeline stage. After it, the project has a full artifact trail, an executable plan with early signals built in as milestones, and an acknowledged list of residual risks. The orchestrator marks the project "plan ready" in `_state.yaml`.

Execution is outside the refinement pipeline's scope. Common follow-ons: proposal writing (using artifacts as source material), beamtime application (using the capability map and plan), internal presentation (using the one-pager as backbone), external review solicitation (sending the one-pager to the skeptical collaborator the premortem imagined).

If during execution the plan needs revision, the orchestrator can re-enter any stage. Artifacts are versioned.
