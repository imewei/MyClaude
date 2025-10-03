# 🚀 New Agents Quick Reference Guide

**10 New Scientific Characterization Agents** | **Fast Lookup** | **Last Updated:** 2025-09-29

---

## 📊 AT-A-GLANCE AGENT SELECTOR

### Need Particle Size? → **Light Scattering**
**Agent:** `light-scattering-optical-expert`
**Use When:** Particle sizing (1 nm - 10 μm), diffusion, molecular weight
**Speed:** <5 minutes | **Cost:** $ | **Sample:** Solution, dispersion

### Need Atomic Structure? → **Electron Microscopy**
**Agent:** `electron-microscopy-diffraction-expert`
**Use When:** Atomic resolution, nanostructure, chemical mapping
**Speed:** Hours-days | **Cost:** $$$ | **Sample:** Solid, thin (~100 nm)

### Need Electronic Properties? → **DFT Expert**
**Agent:** `electronic-structure-dft-expert`
**Use When:** Bandgaps, DOS, elastic constants, phonons, surfaces
**Speed:** Minutes-hours | **Cost:** Compute | **Sample:** Calculated structure

### Need Chemical ID? → **Spectroscopy**
**Agent:** `spectroscopy-expert`
**Use When:** Molecular identification, functional groups, dynamics
**Speed:** Minutes | **Cost:** $-$$ | **Sample:** Solid/liquid/gas

### Need Crystal Structure? → **Crystallography**
**Agent:** `crystallography-diffraction-expert`
**Use When:** Phase ID, lattice parameters, crystallinity
**Speed:** Minutes-hours | **Cost:** $-$$ | **Sample:** Powder/crystal

### Need Surface Analysis? → **Characterization Master**
**Agent:** `materials-characterization-master`
**Use When:** Multi-technique coordination, surface composition, topography
**Speed:** Hours | **Cost:** $$ | **Sample:** Solid surfaces

### Need AI Discovery? → **Materials Informatics**
**Agent:** `materials-informatics-ml-expert`
**Use When:** High-throughput screening, property prediction, design
**Speed:** Minutes-hours | **Cost:** Compute | **Sample:** Virtual

### Need Interface Properties? → **Surface Science**
**Agent:** `surface-interface-science-expert`
**Use When:** Adsorption, QCM-D, SPR, wettability, thin films
**Speed:** Minutes-hours | **Cost:** $-$$ | **Sample:** Surfaces/interfaces

### Need Mechanical Properties? → **Rheologist**
**Agent:** `rheologist`
**Use When:** Viscosity, G'/G'', DMA, tensile, peel strength
**Speed:** Minutes-hours | **Cost:** $ | **Sample:** Bulk material

### Need Molecular Dynamics? → **Simulation**
**Agent:** `simulation-expert`
**Use When:** MD predictions, MLFFs, structure-property from atomistic
**Speed:** Hours-days | **Cost:** Compute | **Sample:** Virtual

---

## 🔍 DECISION TREE

```
START: What do you need to know?
│
├─ Size/Shape?
│  ├─ 1 nm - 10 μm in solution → LIGHT SCATTERING
│  ├─ Atomic resolution, morphology → ELECTRON MICROSCOPY
│  └─ Crystal structure, phases → CRYSTALLOGRAPHY
│
├─ Composition/Chemistry?
│  ├─ Surface composition (0-10 nm) → CHARACTERIZATION MASTER (XPS)
│  ├─ Molecular identification → SPECTROSCOPY (IR/Raman/NMR)
│  └─ Elemental mapping → ELECTRON MICROSCOPY (EDX/EELS)
│
├─ Electronic/Optical?
│  ├─ Bandgap, DOS → DFT EXPERT
│  ├─ Optical absorption → SPECTROSCOPY (UV-Vis)
│  └─ Plasmons, photoluminescence → ELECTRON MICROSCOPY (EELS/CL)
│
├─ Mechanical/Rheological?
│  ├─ Viscosity, G', G'' → RHEOLOGIST (rheometry)
│  ├─ Tensile strength, modulus → RHEOLOGIST (mechanical testing)
│  ├─ High-frequency (GHz) → LIGHT SCATTERING (Brillouin)
│  └─ Nanoindentation → CHARACTERIZATION MASTER
│
├─ Dynamics/Kinetics?
│  ├─ Diffusion coefficients → LIGHT SCATTERING (DLS)
│  ├─ Molecular dynamics → SIMULATION (MD)
│  ├─ Phase transitions → SPECTROSCOPY (BDS), DFT, CHARACTERIZATION MASTER (DSC)
│  └─ Reaction kinetics → SPECTROSCOPY (time-resolved)
│
├─ Interface/Surface?
│  ├─ Adsorption kinetics → SURFACE SCIENCE (QCM-D)
│  ├─ Biomolecular binding → SURFACE SCIENCE (SPR)
│  ├─ Surface energy → SURFACE SCIENCE, CHARACTERIZATION MASTER
│  └─ Thin films → CHARACTERIZATION MASTER (ellipsometry), ELECTRON MICROSCOPY
│
├─ Structure-Property Prediction?
│  ├─ From DFT → DFT EXPERT (properties from electronic structure)
│  ├─ From MD → SIMULATION (properties from atomistic models)
│  └─ From ML → MATERIALS INFORMATICS (property prediction, discovery)
│
└─ Multi-Technique Integration?
   └─ CHARACTERIZATION MASTER (coordinates all techniques)
```

---

## ⚡ COMMON WORKFLOWS

### Workflow 1: Polymer Characterization
**Question:** "What are the molecular weight, size, and viscoelastic properties?"

**Agent Sequence:**
1. **Light Scattering** → DLS size distribution, SLS molecular weight
2. **Rheologist** → G', G'', tan δ, Tg from DMA
3. **Simulation** → MD to correlate MW with rheology
4. **Cross-Validate** → Compare experimental with MD predictions

**Expected Time:** 1 day
**Deliverables:** Size, MW, Rg, G'(ω), Tg, molecular model

---

### Workflow 2: Nanoparticle Analysis
**Question:** "What is the size, shape, composition, and crystal structure?"

**Agent Sequence:**
1. **Light Scattering** → DLS hydrodynamic radius (fast screening)
2. **Electron Microscopy** → TEM size/shape, SAED crystal structure, EDX composition
3. **Crystallography** → XRD phase identification, lattice parameters
4. **Characterization Master** → XPS surface composition, AFM topography

**Expected Time:** 2-3 days
**Deliverables:** Size distribution, morphology, crystal structure, composition profile

---

### Workflow 3: Battery Material Discovery
**Question:** "Find new cathode materials with high voltage and fast Li⁺ diffusion"

**Agent Sequence:**
1. **Materials Informatics** → ML screening of candidate compositions
2. **DFT Expert** → Calculate voltage, bandgap, Li diffusion barriers
3. **Crystallography** → XRD validation of synthesized materials
4. **Spectroscopy** → EIS for charge transfer resistance, ionic conductivity
5. **Close Loop** → Retrain ML models with new data

**Expected Time:** 2-4 weeks (with synthesis)
**Deliverables:** Optimized material, predicted properties, experimental validation

---

### Workflow 4: Thin Film Characterization
**Question:** "What are the thickness, composition, roughness, and optical properties?"

**Agent Sequence:**
1. **Characterization Master** → Coordinate multi-technique analysis
2. **Electron Microscopy** → TEM cross-section for thickness, interfaces
3. **Spectroscopy** → XPS for composition and depth profiling
4. **Characterization Master** → Ellipsometry for thickness/optical, AFM for roughness
5. **Surface Science** → Contact angle for wettability

**Expected Time:** 1 week
**Deliverables:** Thickness, composition profile, roughness, optical constants, wettability

---

### Workflow 5: Soft Matter Structure-Dynamics
**Question:** "How does structure evolve under flow?"

**Agent Sequence:**
1. **Light Scattering** → DLS for equilibrium size, SLS for structure factor
2. **Rheologist** → Rheo-SAXS/SANS under flow (delegate to neutron/xray experts)
3. **Simulation** → MD/DPD to model flow-induced alignment
4. **Correlation Function Expert** → Theoretical interpretation

**Expected Time:** 1-2 weeks
**Deliverables:** g(r), S(k) vs. shear rate, flow curve, MD snapshots

---

## 🎯 AGENT COMBINATIONS (SYNERGY TRIPLETS)

### Triplet 1: Scattering Hub
**Agents:** Light Scattering + Neutron + X-ray
**Use When:** Need multi-scale structure (Å to μm)
**Example:** Hierarchical self-assembly (micelles → clusters → gels)

### Triplet 2: Mechanical Properties
**Agents:** Rheologist + Simulation + Light Scattering (Brillouin)
**Use When:** Link molecular structure to mechanical properties
**Example:** Polymer entanglement → viscosity → modulus

### Triplet 3: Electronic Structure
**Agents:** DFT + Electron Microscopy (EELS) + Spectroscopy (XPS)
**Use When:** Electronic structure from theory + experiment
**Example:** Bandgap engineering in semiconductors

### Triplet 4: AI-Driven Discovery
**Agents:** Materials Informatics + DFT + Characterization Master
**Use When:** Accelerated materials discovery
**Example:** 100 candidates → 10 DFT → 3 synthesize/validate

### Triplet 5: Interface Analysis
**Agents:** Surface Science + Characterization Master + DFT
**Use When:** Surface reactions, adsorption, catalysis
**Example:** QCM-D kinetics + XPS composition + DFT binding energy

---

## 📋 QUICK COMMAND REFERENCE

### Invoke Single Agent
```bash
# Example: DLS particle sizing
Claude: "Use light-scattering-optical-expert to analyze DLS data at 90°, extract size distribution"
```

### Multi-Agent Workflow
```bash
# Example: Polymer characterization
Claude: "1) Use light-scattering-expert for DLS size and SLS molecular weight
         2) Use rheologist for frequency sweep at 25°C
         3) Use simulation-expert for MD to predict viscosity
         4) Cross-validate results"
```

### Delegation Pattern
```bash
# Example: Characterization master coordinates
Claude: "Use materials-characterization-master to design strategy for thin film analysis.
         Delegate to electron-microscopy for cross-section, spectroscopy for XPS,
         and ellipsometry for optical properties"
```

---

## ⏱️ TIME & COST ESTIMATES

| Agent | Typical Time | Cost | Sample Prep |
|-------|-------------|------|-------------|
| **Light Scattering** | 5-30 min | $ | Dilution, filtration |
| **Electron Microscopy** | 2 hours - 2 days | $$$ | TEM grid, FIB, ultramicrotomy |
| **DFT Expert** | 10 min - 24 hours | Compute ($$) | Structure file only |
| **Spectroscopy** | 5 min - 2 hours | $-$$ | Minimal (pellet, solution) |
| **Crystallography** | 15 min - 4 hours | $-$$ | Powder, single crystal |
| **Characterization Master** | Varies | $-$$$ | Depends on techniques |
| **Materials Informatics** | 1 hour - 1 week | Compute ($-$$) | None (virtual) |
| **Surface Science** | 30 min - 4 hours | $-$$ | Clean surface, thin film |
| **Rheologist** | 15 min - 4 hours | $ | Bulk sample (mL-scale) |
| **Simulation** | 1 hour - 1 week | Compute ($-$$$) | Structure file, force field |

**Cost Legend:** $ (<$100), $$ ($100-$1000), $$$ (>$1000) per measurement

---

## 🚨 TROUBLESHOOTING

### "Which agent should I use?"
→ Use Decision Tree above or describe your question to Claude

### "Agent not responding as expected"
→ Check agent name spelling, verify tools available, provide more context

### "Need multiple techniques"
→ Use **Characterization Master** to coordinate, or sequence agents manually

### "Results don't match across agents"
→ Normal! Each technique has different sensitivity. Use Cross-Validation sections.

### "Want faster results"
→ Start with **Light Scattering** (fastest), escalate to EM/DFT if needed

### "Computational resources limited"
→ **Materials Informatics** → screen first, then **DFT** for top candidates only

---

## 📚 FURTHER READING

- **Full Agent Descriptions:** See individual `.md` files in `.claude/agents/`
- **Integration Testing:** `INTEGRATION_TESTING.md`
- **Deployment Strategy:** Main README (3-phase plan)
- **Synergy Matrix:** Main README (triplet patterns)

---

## 🎓 LEARNING PATH

### Beginner (Week 1)
**Deploy:** Light Scattering, Rheologist, Spectroscopy
**Try:** Polymer characterization workflow
**Goal:** Understand agent invocation and delegation

### Intermediate (Month 1)
**Deploy:** All Phase 1 + Phase 2 agents
**Try:** Multi-technique workflows (nanoparticles, thin films)
**Goal:** Master multi-agent coordination

### Advanced (Month 3)
**Deploy:** All 33 agents (23 existing + 10 new)
**Try:** AI-driven discovery, closed-loop optimization
**Goal:** Design custom workflows for research

---

## 🔄 VERSION HISTORY

**v1.0** (2025-09-29)
- Initial quick reference for 10 new agents
- Decision tree and common workflows
- Agent combinations and time estimates

---

*Quick reference enables fast agent selection and workflow design. For comprehensive details, see individual agent files. For deployment guidance, see 3-phase strategic plan in main documentation.*