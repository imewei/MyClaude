# Common Workflows

Practical examples of multi-technique characterization workflows.

## Workflow 1: Nanoparticle Size Characterization

Compare size measurements from multiple techniques (cross-validation).

```python
from characterization_master import CharacterizationMaster

master = CharacterizationMaster()

result = master.execute({
    'workflow_type': 'nanoparticle_analysis',
    'sample_info': {
        'type': 'gold_nanoparticles',
        'synthesis': 'citrate_reduction'
    }
})

# Access cross-validation results
validation = result.data['cross_validation']
print(f"Overall agreement: {validation['overall_agreement']:.2%}")
print(f"Validation passed: {validation['validation_passed']}")

for v in validation['validations']:
    print(f"\n{v['validation_type']}:")
    print(f"  Techniques: {v['techniques']}")
    print(f"  Agreement: {v['agreement_score']:.2%}")
```

## Workflow 2: Polymer Characterization

Use the Characterization Master to run a comprehensive polymer analysis.

```python
from characterization_master import CharacterizationMaster

master = CharacterizationMaster()

# Run predefined polymer characterization workflow
result = master.execute({
    'workflow_type': 'polymer_characterization',
    'sample_info': {
        'name': 'Polystyrene-100k',
        'concentration': '1 mg/mL',
        'solvent': 'toluene'
    }
})

if result.success:
    # Access integrated report
    report = result.data['integrated_report']
    print(f"Techniques used: {', '.join(report['techniques_used'])}")
    print(f"\nKey findings:")
    for finding in report['key_findings']:
        print(f"  - {finding}")

    # Access individual results
    individual_results = result.data['individual_results']
    for technique, data in individual_results.items():
        print(f"\n{technique}: {data['status']}")
```

## Workflow 3: AI-Driven Materials Discovery

Use Materials Informatics for property prediction.

```python
from materials_informatics_agent import MaterialsInformaticsAgent

agent = MaterialsInformaticsAgent()

# Predict properties for candidate materials
result = agent.execute({
    'task': 'property_prediction',
    'structures': [
        {'composition': 'LiFePO4'},
        {'composition': 'LiCoO2'},
        {'composition': 'LiMn2O4'}
    ],
    'target_properties': ['band_gap', 'formation_energy', 'ionic_conductivity']
})

if result.success:
    for i, pred in enumerate(result.data['predictions']):
        print(f"\nMaterial {i+1}:")
        for prop, values in pred['predictions'].items():
            print(f"  {prop}: {values['value']:.3f} ± {values['uncertainty']:.3f}")
            print(f"  Confidence: {values['confidence']:.2%}")
```

## Workflow 4: Surface Science Analysis

Measure biomolecular interactions using SPR.

```python
from surface_science_agent import SurfaceScienceAgent

agent = SurfaceScienceAgent()

result = agent.execute({
    'technique': 'SPR',
    'concentrations': [10, 25, 50, 100, 200],  # nM
    'parameters': {
        'temperature': 298,
        'flow_rate': 50  # µL/min
    }
})

if result.success:
    kinetics = result.data['kinetic_analysis']
    print(f"kon: {kinetics['kon_M_inv_s']:.2e} M⁻¹s⁻¹")
    print(f"koff: {kinetics['koff_s_inv']:.2e} s⁻¹")
    print(f"KD: {kinetics['KD_nM']:.2f} nM")
    print(f"Affinity: {result.data['affinity_classification']}")
```

## Workflow 5: Scattering Validation (Synergy Triplet)

Cross-validate particle sizes using SANS → MD → DLS.

```python
from agent_orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator()

# Execute synergy triplet: SANS → MD → DLS validation
workflow = orchestrator.create_synergy_triplet(
    'scattering_validation',
    sans_data=sans_result,
    structure=structure_file
)

result = orchestrator.execute_workflow(workflow)

# Check cross-validation
print(f"SANS Rg: {result.sans_result['Rg_nm']} nm")
print(f"MD predicted size: {result.md_result['size_nm']} nm")
print(f"DLS Rh: {result.dls_result['Rh_nm']} nm")
print(f"Agreement score: {result.validation['agreement']:.2%}")
```

## Integration Patterns

### Pattern 1: Scattering Validation
```
SANS/SAXS → MD Simulation → Light Scattering (DLS)
  S(q), Rg  →  Validate S(q)  →  Validate particle size
```

### Pattern 2: Structure-Property-Processing
```
DFT → MD Simulation → Rheology
Elastic constants → Viscosity prediction → Experimental validation
```

### Pattern 3: Composition-Structure-Electronics
```
Spectroscopy → Crystallography → DFT
Molecular ID → Crystal structure → Electronic properties
```
