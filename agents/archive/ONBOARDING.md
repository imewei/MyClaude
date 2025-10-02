# 🎓 User Onboarding Guide: 10 New Scientific Agents

**Welcome!** This guide will help you deploy and use the 10 new scientific characterization agents in 30 minutes.

**Target Audience:** Researchers, scientists, engineers using Claude Code for materials characterization
**Prerequisites:** Claude Code installed, `.claude/agents/` directory exists
**Time Required:** 30 minutes (quick start) to 2 hours (comprehensive)

---

## 🚀 QUICK START (30 Minutes)

### Step 1: Verify Installation (5 minutes)

#### Check Agent Files Exist
```bash
cd ~/.claude/agents/
ls -1 | grep -E "(light-scattering|electron-microscopy|electronic-structure|spectroscopy|crystallography|materials-characterization|materials-informatics|surface-interface|rheologist|simulation)"
```

**Expected Output:** 10 agent files listed

#### Verify Agent File Format
```bash
head -6 light-scattering-optical-expert.md
```

**Expected Output:**
```yaml
--
name: light-scattering-optical-expert
description: Light scattering and optical methods expert...
tools: Read, Write, Edit, MultiEdit, Bash, ...
model: inherit
--
```

✅ **Checkpoint:** All 10 files present with valid YAML frontmatter

---

### Step 2: Test First Agent (10 minutes)

#### Invoke Light Scattering Agent
**Scenario:** You have DLS data from a 90° measurement and want particle size.

**Claude Invocation:**
```
Please use the light-scattering-optical-expert agent to help me:

I have dynamic light scattering (DLS) data measured at 90° for a polymer solution.
The autocorrelation function shows a single exponential decay with a decay time
of approximately 5 milliseconds. The temperature was 25°C, and the solvent is
water (viscosity 0.89 mPa·s, refractive index 1.33).

Can you:
1. Calculate the diffusion coefficient
2. Estimate the hydrodynamic radius using Stokes-Einstein
3. Assess the quality of the data
```

**Expected Agent Response:**
- Calculation of diffusion coefficient D from decay time
- Hydrodynamic radius Rh calculation
- Assessment of single exponential = monodisperse sample
- Recommendation for follow-up (SLS for MW, multi-angle DLS for shape)

**Success Criteria:**
- ✅ Agent provides quantitative analysis
- ✅ Physical reasoning sound
- ✅ Recommendations actionable

---

### Step 3: Test Multi-Agent Workflow (15 minutes)

#### Scenario: Polymer Characterization
**Goal:** Characterize a polymer sample (molecular weight, size, glass transition)

**Step 3a: Light Scattering for Size/MW**
```
Use light-scattering-optical-expert:

I need to characterize a polymer sample. I have:
- DLS data showing Rh ~ 25 nm
- SLS data at multiple angles (30°, 60°, 90°, 120°, 150°)

From SLS:
- Zimm plot shows Mw ~ 500 kDa
- Radius of gyration Rg ~ 22 nm
- Second virial coefficient A2 ~ 2×10⁻⁴ mol·mL/g²

Interpret these results. What do Rg/Rh ratio and A2 tell us?
```

**Expected:** Agent explains Rg/Rh ~ 0.88 suggests compact structure (sphere/branched), positive A2 indicates good solvent.

---

**Step 3b: Rheology for Viscoelasticity**
```
Use rheologist agent:

For the same polymer (Mw = 500 kDa), I ran frequency sweeps from 0.01 to 100 rad/s:
- At low frequency: G'' > G' (liquid-like)
- Crossover at ω ~ 1 rad/s
- At high frequency: G' > G'' (solid-like)
- Plateau modulus G°N ~ 1 MPa

Also ran DMA temperature sweep:
- Tan δ peak at 85°C

Analyze the viscoelastic behavior and determine Tg.
```

**Expected:** Agent identifies entangled polymer, calculates entanglement MW from plateau modulus, confirms Tg = 85°C.

---

**Step 3c: Cross-Validation**
```
Compare the results from light-scattering and rheology agents.

Light scattering: Mw = 500 kDa, Rh = 25 nm, Rg = 22 nm
Rheology: G°N = 1 MPa, Tg = 85°C

Are these consistent? What can we conclude about the polymer structure?
```

**Expected:** Cross-validation confirms entangled polymer with Mw > Me, compact structure, good agreement between techniques.

---

✅ **Checkpoint:** Successfully invoked 2 agents and cross-validated results

---

## 📚 COMPREHENSIVE ONBOARDING (2 Hours)

### Module 1: Phase 1 Agents (Foundation)

#### Agent 1: Light Scattering & Optical Expert
**Use Cases:**
- DLS: Particle sizing, diffusion, aggregation kinetics
- SLS: Molecular weight, radius of gyration, second virial coefficient
- MALS: SEC-MALS for polymer MW distributions
- Raman: Chemical fingerprinting, crystallinity
- Brillouin: Elastic moduli at GHz frequencies

**Practice Exercise:**
```
Scenario: You measured DLS at multiple temperatures (15, 25, 35, 45°C).
The hydrodynamic radius increases with temperature: 20, 25, 32, 45 nm.

Question: What physical processes could explain this?
Task: Use light-scattering-optical-expert to interpret and recommend follow-up.
```

**Expected Learning:**
- Temperature-dependent aggregation
- Need for SLS to distinguish MW change vs. aggregation
- Potential phase transition (LCST behavior)

---

#### Agent 2: Electron Microscopy & Diffraction Expert
**Use Cases:**
- TEM: Atomic-resolution imaging, crystal structure
- SEM: Surface morphology, composition mapping (EDX)
- STEM: Z-contrast imaging, single-atom detection
- EELS: Electronic structure, bonding
- 4D-STEM: Strain mapping, orientation

**Practice Exercise:**
```
Scenario: You have HRTEM images of gold nanoparticles showing lattice fringes.
FFT shows spots corresponding to d111 = 2.35 Å.

Question: Confirm the crystal structure and estimate crystallite size.
Task: Use electron-microscopy-diffraction-expert for analysis.
```

**Expected Learning:**
- Lattice spacing identification (FCC gold d111 = 2.36 Å)
- FFT analysis for crystal structure
- Complementarity with XRD

---

#### Agent 3: Electronic Structure & DFT Expert
**Use Cases:**
- Band structure calculations
- DOS, PDOS for electronic properties
- Elastic constants, phonons
- Surface energies, defect formation energies
- On-the-fly ML force fields

**Practice Exercise:**
```
Scenario: You want to calculate the bandgap of anatase TiO2.

Question: Which functional should I use? What k-point grid?
Task: Use electronic-structure-dft-expert to design calculation.
```

**Expected Learning:**
- PBE underestimates gaps → use HSE06 or GW
- k-point convergence testing needed
- Comparison with experimental UV-Vis (3.2 eV)

---

#### Agent 9: Rheologist
**Use Cases:**
- Oscillatory shear: G', G'', tan δ
- Steady shear: Viscosity curves
- DMA: Glass transition, storage/loss modulus
- Extensional rheology: Filament stretching, CaBER
- Mechanical testing: Tensile, compression, peel

**Practice Exercise:**
```
Scenario: Frequency sweep shows G' ~ ω² at low ω, G' ~ ω⁰ at high ω.

Question: What does this tell us about the material?
Task: Use rheologist to interpret and fit Maxwell model.
```

**Expected Learning:**
- Low ω: Terminal regime (liquid-like, G' ~ ω²)
- High ω: Plateau (entangled, elastic)
- Maxwell model captures relaxation times

---

#### Agent 10: Simulation Expert
**Use Cases:**
- Classical MD: Structure, dynamics, transport properties
- ML force fields: DFT accuracy at MD speed
- HOOMD-blue: Soft matter, anisotropic particles
- DPD: Mesoscale hydrodynamics
- NEMD: Viscosity, thermal conductivity

**Practice Exercise:**
```
Scenario: You want to predict the viscosity of a polymer melt at 200°C.

Question: Which method (classical MD vs. MLFF)? What ensemble?
Task: Use simulation-expert to design workflow.
```

**Expected Learning:**
- Use NEMD with shear flow for viscosity
- MLFF if reactive or complex chemistry
- NPT ensemble for pressure control
- Compare with rheometry (rheologist)

---

### Module 2: Phase 2 Agents (Enhancement)

#### Agent 4: Spectroscopy Expert
**Key Techniques:**
- IR/Raman: Vibrational modes, functional groups
- NMR: Structure elucidation, dynamics
- EIS: Electrochemical impedance (batteries, corrosion)
- BDS: Dielectric spectroscopy (polymer dynamics, Tg)

**Practice Exercise:**
```
Scenario: Battery shows high charge-transfer resistance in EIS.
Nyquist plot: large semicircle at high frequency.

Question: Diagnose the issue and recommend fixes.
Task: Use spectroscopy-expert for EIS interpretation.
```

---

#### Agent 5: Crystallography & Diffraction Expert
**Key Techniques:**
- XRD: Phase identification, Rietveld refinement
- PDF: Local structure in amorphous/nano materials
- Synchrotron: High-resolution, time-resolved

**Practice Exercise:**
```
Scenario: XRD shows broad peaks. Is it nanocrystalline or amorphous?

Task: Use crystallography-diffraction-expert to distinguish via PDF analysis.
```

---

#### Agent 6: Materials Characterization Master
**Role:** Multi-technique coordinator

**Practice Exercise:**
```
Scenario: Characterize a thin film on silicon substrate.

Task: Use materials-characterization-master to design strategy.
Expected: Delegate to TEM (cross-section), XPS (composition), ellipsometry (thickness).
```

---

### Module 3: Phase 3 Agents (Advanced)

#### Agent 7: Materials Informatics & ML Expert
**Capabilities:**
- ML property prediction
- High-throughput DFT screening
- Active learning, Bayesian optimization
- Generative models (VAE, diffusion)

**Practice Exercise:**
```
Scenario: Find new thermoelectric materials with ZT > 2.

Task: Use materials-informatics-ml-expert to screen Materials Project,
      predict ZT with ML, delegate top 10 to DFT for validation.
```

---

#### Agent 8: Surface & Interface Science Expert
**Capabilities:**
- QCM-D: Real-time mass/viscoelasticity
- SPR: Biomolecular binding kinetics
- Surface energy, wettability

**Practice Exercise:**
```
Scenario: QCM-D shows mass increase but ΔD also increases.

Question: Is the film rigid or viscoelastic?
Task: Use surface-interface-science-expert to analyze.
```

---

## 🔗 INTEGRATION WORKFLOWS

### Workflow 1: Nanoparticle Complete Characterization
**Agents:** Light Scattering → Electron Microscopy → Crystallography

**Steps:**
1. **Light Scattering** (5 min): DLS for quick size screening
2. **Electron Microscopy** (2 hours): TEM for morphology, SAED for crystal structure
3. **Crystallography** (30 min): XRD for bulk phase identification

**Deliverable:** Size distribution, morphology, crystal structure

---

### Workflow 2: Battery Material Discovery
**Agents:** Materials Informatics → DFT → Spectroscopy

**Steps:**
1. **Materials Informatics** (1 day): Screen 1000 candidates, predict voltage
2. **DFT** (1 week): Calculate top 20 candidates (voltage, bandgap, diffusion)
3. **Spectroscopy** (1 week): Synthesize top 3, measure EIS

**Deliverable:** Optimized cathode material with experimental validation

---

### Workflow 3: Polymer Processing Optimization
**Agents:** Rheologist → Simulation → Light Scattering

**Steps:**
1. **Rheologist** (1 hour): Measure η(γ̇), extensional viscosity
2. **Simulation** (1 day): MD to predict viscosity from molecular structure
3. **Light Scattering** (30 min): Verify no degradation (MW stable)

**Deliverable:** Processing window (temperature, shear rate) for extrusion

---

## 🛠️ TROUBLESHOOTING

### Problem 1: Agent Not Invoked
**Symptoms:** Claude doesn't use specified agent

**Solutions:**
- ✅ Verify agent name spelling: `light-scattering-optical-expert` (not `light-scattering`)
- ✅ Explicitly request: "Use [agent-name] to..."
- ✅ Check agent file exists in `~/.claude/agents/`

---

### Problem 2: Results Don't Match Across Agents
**Symptoms:** DLS size ≠ TEM size, MD viscosity ≠ rheometry

**Solutions:**
- ✅ **Normal!** Different techniques have different sensitivities
- ✅ DLS measures hydrodynamic radius (solvation shell), TEM measures core
- ✅ MD captures molecular, rheology measures bulk (may have aggregates)
- ✅ Use cross-validation sections in agent outputs

---

### Problem 3: Agent Recommends Unavailable Tool
**Symptoms:** Agent suggests VASP calculation but no license

**Solutions:**
- ✅ Specify constraints: "Use open-source DFT (Quantum ESPRESSO)"
- ✅ Alternative methods: "Use ML force field instead of DFT"
- ✅ Delegate: "Use simulation-expert with MLFF trained on DFT data"

---

### Problem 4: Too Many Options, Confused
**Symptoms:** Don't know which agent to use

**Solutions:**
- ✅ Use **Decision Tree** in `AGENTS_QUICKREF.md`
- ✅ Start with **Characterization Master** to design strategy
- ✅ Describe problem to Claude: "I have X, want to know Y"

---

## 📊 PROGRESS CHECKLIST

### Quick Start Complete ✅
- [ ] Verified all 10 agent files installed
- [ ] Successfully invoked light-scattering-optical-expert
- [ ] Ran multi-agent workflow (light scattering + rheologist)
- [ ] Cross-validated results from 2+ agents

### Comprehensive Onboarding Complete ✅
- [ ] Practiced with all Phase 1 agents (1, 2, 3, 9, 10)
- [ ] Explored Phase 2 agents (4, 5, 6)
- [ ] Tested Phase 3 agents (7, 8)
- [ ] Completed 3 integration workflows

### Ready for Production ✅
- [ ] Designed custom workflow for research project
- [ ] Successfully cross-validated 3+ techniques
- [ ] Consulted `AGENTS_QUICKREF.md` for fast lookup
- [ ] Read `INTEGRATION_TESTING.md` for quality assurance

---

## 🎓 LEARNING RESOURCES

### Documentation
- **Quick Reference:** `AGENTS_QUICKREF.md` - Fast agent selection
- **Integration Testing:** `INTEGRATION_TESTING.md` - Quality assurance
- **Deployment Strategy:** Main README - 3-phase plan

### Example Workflows
- See `AGENTS_QUICKREF.md` → Common Workflows section
- Polymer characterization (DLS + rheology + MD)
- Nanoparticle analysis (DLS + TEM + XRD)
- Battery discovery (ML + DFT + EIS)

### Advanced Topics
- **Multi-Agent Orchestration:** See synergy triplets in main docs
- **Active Learning:** Materials Informatics agent workflows
- **Closed-Loop Optimization:** ML → DFT → Experiments → Retrain

---

## 🚀 NEXT STEPS

### Immediate (Today)
1. ✅ Complete Quick Start (30 min)
2. ✅ Test your research data with appropriate agent
3. ✅ Bookmark `AGENTS_QUICKREF.md` for fast lookup

### Short-Term (This Week)
1. ✅ Try 3 different agents relevant to your research
2. ✅ Design a multi-agent workflow for your project
3. ✅ Run `INTEGRATION_TESTING.md` checklist

### Long-Term (This Month)
1. ✅ Deploy Phase 2 & 3 agents as needed
2. ✅ Develop custom workflows for routine tasks
3. ✅ Share best practices with colleagues

---

## 📞 SUPPORT

### Questions or Issues?
- Consult `AGENTS_QUICKREF.md` for quick answers
- Check `INTEGRATION_TESTING.md` for testing guidance
- Review individual agent files for detailed capabilities

### Feedback?
- Document successful workflows for future users
- Report issues or limitations for improvement
- Suggest new features or enhancements

---

## 🎉 SUCCESS CRITERIA

**You've successfully onboarded when you can:**
- ✅ Invoke any of the 10 agents confidently
- ✅ Design multi-agent workflows for your research
- ✅ Cross-validate results across techniques
- ✅ Troubleshoot common issues independently
- ✅ Navigate documentation efficiently

**Welcome to the 33-agent scientific characterization ecosystem!**

---

*Onboarding guide ensures rapid adoption and productive usage of the 10 new agents. For questions, consult quick reference guide or individual agent documentation.*