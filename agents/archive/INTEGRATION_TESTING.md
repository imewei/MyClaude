# Agent Integration Testing Checklist

**Purpose:** Systematic validation of 10 new agents integration with existing 23-agent ecosystem

**Date Created:** 2025-09-29
**Status:** Ready for Testing

---

## Phase 1: File Validation

### Agent File Integrity
- [ ] `light-scattering-optical-expert.md` - Parse YAML, verify tools
- [ ] `electron-microscopy-diffraction-expert.md` - Parse YAML, verify tools
- [ ] `electronic-structure-dft-expert.md` - Parse YAML, verify tools
- [ ] `rheologist.md` - Parse YAML, verify tools
- [ ] `simulation-expert.md` - Parse YAML, verify tools
- [ ] `spectroscopy-expert.md` - Parse YAML, verify tools
- [ ] `crystallography-diffraction-expert.md` - Parse YAML, verify tools
- [ ] `materials-characterization-master.md` - Parse YAML, verify tools
- [ ] `materials-informatics-ml-expert.md` - Parse YAML, verify tools
- [ ] `surface-interface-science-expert.md` - Parse YAML, verify tools

### YAML Frontmatter Validation
```bash
# Run for each agent file
grep -A 5 "^--$" agent-file.md | head -6

# Expected output:
# --
# name: agent-name
# description: ...
# tools: Read, Write, ...
# model: inherit
# --
```

### Naming Conflicts Check
- [ ] No duplicate agent names in `.claude/agents/`
- [ ] No tool specification conflicts
- [ ] Agent names follow kebab-case convention

---

## Phase 2: Multi-Agent Collaboration Testing

### Cross-Validation Workflows

#### Test 1: Light Scattering → SANS Cross-Validation
**Scenario:** Compare DLS particle size with SANS form factor

```python
# Invoke light-scattering-optical-expert
Task: "Analyze DLS data at 90°, extract size distribution"
Expected: Rh ~ 50 nm (example)

# Delegate to neutron-soft-matter-expert
Task: "Compare with SANS P(k), extract particle size"
Expected: Rg ~ 45 nm → Rh = Rg × √(5/3) ~ 58 nm

# Validation
Tolerance: |DLS_Rh - SANS_Rh| < 20%
Status: [ ] Pass [ ] Fail
```

#### Test 2: Rheologist → Simulation Viscosity Validation
**Scenario:** Validate MD-predicted viscosity with rheometry

```python
# Invoke simulation-expert
Task: "Run NEMD simulation, calculate viscosity at T=300K"
Expected: η_MD ~ 100 mPa·s (example)

# Invoke rheologist
Task: "Measure steady shear viscosity at T=300K, γ̇=0.1 s⁻¹"
Expected: η_exp ~ 95 mPa·s

# Validation
Tolerance: |η_MD - η_exp| < 15%
Status: [ ] Pass [ ] Fail
```

#### Test 3: DFT → Spectroscopy Frequency Validation
**Scenario:** Compare DFT-calculated IR frequencies with experiments

```python
# Invoke electronic-structure-dft-expert
Task: "Calculate IR frequencies for molecule X using VASP"
Expected: ν_C=O ~ 1720 cm⁻¹ (DFT)

# Invoke spectroscopy-expert
Task: "Measure FTIR spectrum, identify C=O stretch"
Expected: ν_C=O ~ 1735 cm⁻¹ (experiment)

# Validation
Tolerance: |ν_DFT - ν_exp| < 30 cm⁻¹
Status: [ ] Pass [ ] Fail
```

#### Test 4: Electron Microscopy → XRD Structure Validation
**Scenario:** Validate TEM lattice spacing with XRD d-spacing

```python
# Invoke electron-microscopy-diffraction-expert
Task: "Measure lattice spacing from HRTEM FFT"
Expected: d_111 ~ 2.35 Å (TEM)

# Invoke crystallography-diffraction-expert
Task: "Refine XRD pattern, extract d_111 spacing"
Expected: d_111 ~ 2.36 Å (XRD)

# Validation
Tolerance: |d_TEM - d_XRD| < 0.05 Å
Status: [ ] Pass [ ] Fail
```

#### Test 5: Materials Characterization Master → Multi-Technique Integration
**Scenario:** Coordinator delegates to multiple agents

```python
# Invoke materials-characterization-master
Task: "Design characterization strategy for thin film"

# Expected Delegations:
1. Delegate to electron-microscopy-expert: "TEM cross-section for thickness"
2. Delegate to spectroscopy-expert: "XPS for composition"
3. Delegate to surface-interface-science-expert: "Contact angle for wettability"

# Validation
- [ ] Correct agent selection
- [ ] Proper delegation syntax
- [ ] Results integrated correctly
Status: [ ] Pass [ ] Fail
```

---

## Phase 3: Tool Availability Verification

### Python Libraries
- [ ] `numpy`, `scipy`, `matplotlib`, `pandas` - Core scientific
- [ ] `jupyter` - Interactive notebooks
- [ ] `lmfit` - Curve fitting
- [ ] `ase`, `pymatgen` - Materials science
- [ ] `hyperspy` - Electron microscopy
- [ ] `mdanalysis`, `mdtraj` - MD analysis
- [ ] `scikit-learn`, `jax` - Machine learning

### Specialized Software (Optional)
- [ ] `lammps`, `gromacs` - MD simulation
- [ ] VASP, Quantum ESPRESSO - DFT (commercial/research)
- [ ] Materials Project API - Database access
- [ ] HyperSpy - EM data analysis

**Note:** Missing tools should not block agent loading, only limit specific functionality.

---

## Phase 4: Phased Deployment Validation

### Phase 1: Foundation (Agents 1, 2, 3, 9, 10)
**Target Coverage:** 80-90%

- [ ] Light scattering agent loads successfully
- [ ] Electron microscopy agent loads successfully
- [ ] DFT agent loads successfully
- [ ] Rheologist loads successfully
- [ ] Simulation agent loads successfully
- [ ] All Phase 1 agents accessible via agent selection
- [ ] No conflicts with existing 23 agents

**Integration Test:**
```python
# Workflow: Polymer characterization
1. light-scattering-expert: DLS sizing
2. rheologist: Viscoelastic properties
3. simulation-expert: MD to predict properties
4. Cross-validate results

Status: [ ] Pass [ ] Fail
```

### Phase 2: Enhancement (Agents 4, 5, 6)
**Target Coverage:** 95%

- [ ] Spectroscopy agent integrates with Phase 1
- [ ] Crystallography agent integrates with Phase 1
- [ ] Materials characterization master coordinates all agents
- [ ] No new conflicts introduced

**Integration Test:**
```python
# Workflow: Battery material analysis
1. crystallography-expert: XRD phase identification
2. spectroscopy-expert: EIS for charge transfer
3. dft-expert: Calculate voltage, diffusion barriers
4. materials-characterization-master: Integrate findings

Status: [ ] Pass [ ] Fail
```

### Phase 3: Advanced (Agents 7, 8)
**Target Coverage:** 100%

- [ ] Materials informatics agent functional
- [ ] Surface science agent functional
- [ ] Complete ecosystem operational
- [ ] AI-driven discovery workflows enabled

**Integration Test:**
```python
# Workflow: Closed-loop materials discovery
1. materials-informatics-expert: Predict candidates
2. dft-expert: Screen with DFT
3. materials-characterization-master: Validate top candidates
4. Close loop: Retrain models

Status: [ ] Pass [ ] Fail
```

---

## Phase 5: Performance & Scalability

### Agent Loading Time
- [ ] All 10 agents load in <5 seconds
- [ ] No memory issues with 33 total agents loaded
- [ ] Agent selection interface responsive

### Concurrent Operations
- [ ] Multiple agents can be invoked in parallel
- [ ] No resource conflicts between agents
- [ ] Delegation chains function correctly

### Error Handling
- [ ] Graceful failure if agent file missing
- [ ] Clear error messages for tool unavailability
- [ ] Proper fallback for missing dependencies

---

## Phase 6: Regression Testing

### Existing Agent Compatibility
- [ ] All 23 existing agents still functional
- [ ] No naming conflicts introduced
- [ ] No tool specification conflicts
- [ ] Existing workflows unaffected

### Sample Test: Neutron Scattering Agent
```python
# Verify existing neutron-soft-matter-expert still works
Task: "Analyze SANS data, extract Rg"
Expected: Functional, same behavior as before

Status: [ ] Pass [ ] Fail
```

---

## Phase 7: Documentation Validation

### Agent Documentation Quality
- [ ] Each agent has clear "When to Invoke" section
- [ ] Problem-solving methodologies defined
- [ ] Multi-agent collaboration documented
- [ ] Technology stacks comprehensive
- [ ] Applications and examples provided

### Cross-References
- [ ] Agents properly reference each other
- [ ] Delegation patterns clear
- [ ] Integration workflows documented

---

## Phase 8: User Acceptance Testing

### Usability
- [ ] Agent names intuitive
- [ ] Descriptions clear
- [ ] Tool requirements transparent
- [ ] Examples helpful

### Workflows
- [ ] Common workflows (DLS + rheology + MD) functional
- [ ] Multi-agent orchestration works smoothly
- [ ] Results integrate seamlessly

---

## Testing Execution Log

| Date | Tester | Phase | Result | Notes |
|------|--------|-------|--------|-------|
| YYYY-MM-DD | Name | Phase 1 | ✅/❌ | Comments |
| | | Phase 2 | ✅/❌ | |
| | | Phase 3 | ✅/❌ | |

---

## Rollback Procedures

### If Critical Issues Found

**Immediate Actions:**
1. Document issue in detail
2. Backup current `.claude/agents/` directory
3. Remove problematic agent files
4. Restore previous stable state
5. Report to development team

**Rollback Script:**
```bash
# Backup current state
cp -r ~/.claude/agents ~/.claude/agents_backup_$(date +%Y%m%d_%H%M%S)

# Remove new agents
cd ~/.claude/agents
rm -f light-scattering-optical-expert.md
rm -f electron-microscopy-diffraction-expert.md
rm -f electronic-structure-dft-expert.md
rm -f rheologist.md
rm -f simulation-expert.md
rm -f spectroscopy-expert.md
rm -f crystallography-diffraction-expert.md
rm -f materials-characterization-master.md
rm -f materials-informatics-ml-expert.md
rm -f surface-interface-science-expert.md

# Verify system stability
# Re-test existing 23 agents
```

---

## Success Criteria

**Testing Complete When:**
- ✅ All checkboxes marked
- ✅ All integration tests pass
- ✅ No critical issues identified
- ✅ User acceptance validated
- ✅ Documentation approved

**Sign-Off:**
- Tester: _________________ Date: _________
- Reviewer: _________________ Date: _________
- Approver: _________________ Date: _________

---

*Integration testing ensures the 10 new agents work harmoniously with the existing 23-agent ecosystem, providing reliable, comprehensive materials characterization without conflicts or regressions.*