---
name: physical-learning-systems
description: Physical and energy-based learning in disordered and soft-matter systems — coupled learning, contrastive Hebbian learning in physical (mechanical/electrical) networks, plasticity and memory formation in disordered materials, and Hopfield-style energy-based learning applied to physical substrates. Distinct from classical ML algorithms — this is learning as a physical/statistical-mechanics phenomenon, not a learning algorithm implemented in software.
---

# Physical Learning Systems

Learning as a physical phenomenon: materials and networks that adapt their own structure in response to training signals, not software implementing a learning algorithm.

## Expert Agent

- **`statistical-physicist`** — Physical/energy-based learning theory in disordered and soft-matter systems.

## Scope Boundary

This skill covers systems where the *physical substrate itself* (a mechanical network's spring stiffnesses, an electrical network's conductances, a material's internal disorder) is the thing being trained — the learning rule is implemented by physics (local rules acting on physical degrees of freedom), not by a digital algorithm. For learning algorithms implemented in software (backprop, gradient descent, classical ML), see `science-suite:machine-learning` or `science-suite:deep-learning-hub` — those are a different subject even when the vocabulary overlaps.

## Coupled Learning

A physical network (e.g. a network of variable resistors or elastic springs) is subjected to two boundary conditions: a "free" state (task input only) and a "clamped" state (task input plus a nudge toward the desired output). Local physical elements adjust (e.g. resistance decreases where current differs most between the two states) to reduce the discrepancy — this is a physically-realizable analog of contrastive Hebbian learning, requiring no global error backpropagation since each element only needs locally-available information (the two states of its own local variables).

## Contrastive Hebbian Learning & Energy-Based Models

Hopfield-style energy-based models store patterns as local minima of an energy function; learning adjusts the energy landscape so desired patterns become minima. Contrastive Hebbian learning trains via the difference between a "free" phase (network relaxes without a target) and a "clamped" phase (network relaxes with the target imposed) — mathematically related to coupled learning above but originating from the neural-network/statistical-mechanics literature (Boltzmann machines) rather than the materials-physics literature.

## Plasticity & Memory in Disordered Materials

Disordered materials (glasses, granular packings, amorphous solids) can exhibit memory effects: their response to a perturbation depends on their loading history in ways that let them "remember" prior training protocols. Cyclic loading can create memories analogous to those studied in the glass/jamming literature (see `science-suite:glass-and-collective-dynamics`) — this is the connection between glassy physics and physical learning: both concern how disordered systems' internal degrees of freedom retain a trace of past states.

## Delegation

For the underlying glassy/jamming dynamics that produce memory effects, see `science-suite:glass-and-collective-dynamics`. For the network-dynamics/dynamical-systems side of plasticity rules (treating them as attractor formation in a dynamical system), delegate to `nonlinear-dynamics-expert`. For classical ML learning algorithms (the software kind), see `science-suite:machine-learning`.
