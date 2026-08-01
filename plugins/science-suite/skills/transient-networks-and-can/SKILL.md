---
name: transient-networks-and-can
description: Transient network rheology — physical gels (reversible non-covalent crosslinks) and covalent adaptable networks (vitrimers) with bond-exchange kinetics. Use when modeling stress relaxation in self-healing gels, vitrimers, or any material with reversible/exchangeable crosslinks, including sticky Rouse and Green-Tobolsky models.
---

# Transient Networks & Covalent Adaptable Networks (CAN)

Modeling materials whose network connectivity itself relaxes over time.

## Expert Agent

- **`continuum-mechanics-engineer`** — Transient-network constitutive modeling and bond-kinetics fitting.

## Two Classes of Transient Network

| Class | Crosslink Type | Relaxation Mechanism |
|-------|-----------------|------------------------|
| Physical networks | Reversible non-covalent (H-bonding, ionic, hydrophobic) | Bond lifetime — a crosslink breaks and may or may not reform at the same location |
| Covalent adaptable networks (vitrimers) | Permanent covalent connectivity, but bonds exchange via a catalyzed reaction | Bond-exchange reaction rate — network topology changes but connectivity (crosslink density) stays constant |

This distinction matters: physical networks can flow and dissolve (crosslink density itself drops if bonds don't reform), while a well-designed vitrimer maintains constant crosslink density and instead behaves like a viscosity that depends on exchange-reaction rate — this is why vitrimers can be reprocessed like thermoplastics while retaining thermoset-like properties at use temperature.

## Sticky Rouse Model (Physical Networks)

Extends the Rouse model of polymer dynamics by adding "sticker" groups that transiently associate. Relaxation occurs on two timescales: fast Rouse-like motion between stickers, and slow terminal relaxation gated by sticker unbinding. The terminal relaxation time scales with the sticker lifetime, not the polymer's intrinsic Rouse time, for strongly associating systems.

## Green-Tobolsky Model (Bond-Exchange Kinetics)

Treats bond breaking/reformation as a first-order kinetic process. For a single exchange rate, stress relaxation follows:

```
σ(t) = σ₀ exp(-t/τ_exchange)
```

where `τ_exchange` is set by the bond-exchange reaction rate (typically Arrhenius in temperature: `τ_exchange = τ₀ exp(Ea/RT)`). This gives vitrimers their signature Arrhenius (not WLF) temperature dependence of viscosity — a diagnostic distinguishing vitrimer behavior from conventional polymer melt rheology.

## Diagnostic Checklist

- [ ] Confirm crosslink density is constant across the temperature/time range studied (vitrimer) vs. dropping (physical network dissociation)
- [ ] Check whether stress relaxation follows Arrhenius (vitrimer, bond-exchange-limited) or WLF (conventional polymer, reptation/segmental-motion-limited) temperature dependence — this is the primary experimental signature distinguishing the two mechanisms
- [ ] For multi-exchange-rate systems (multiple bond types), a single Green-Tobolsky exponential under-fits — use a distribution of exchange rates (analogous to the Prony series extension of a single Maxwell element)

## Delegation

For the underlying stochastic bond-kinetics derivation (rather than the engineering constitutive fit), delegate to `statistical-physicist`. For fitting the resulting relaxation data to a broader constitutive framework, see `science-suite:constitutive-equations`.
