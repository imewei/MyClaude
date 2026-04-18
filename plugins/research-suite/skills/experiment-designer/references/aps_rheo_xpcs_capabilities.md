# APS Rheo-XPCS capability baseline (8-ID-I)

Baseline capability values for quick margin checks during instrument-capability-map construction. **This file is an illustrative template; Wei should replace the values with current measured baselines from recent beamtime runs.**

The point of having this file in the skill stack is to avoid looking up the same numbers every time a capability map is built, and to create an institutional record of how capabilities evolve as the beamline is upgraded.

## Current baseline (as of [date to be updated])

Fill in from most recent confirmed beamtime:

### Temporal

- Frame rate (standard): [fill in, e.g., up to X Hz at Y detector]
- Frame rate (fast mode): [fill in]
- Minimum exposure time per frame: [fill in]
- Maximum continuous acquisition: [fill in, limited by storage or beam stability]

### Spatial / coherence

- Coherence length (typical): [fill in]
- Beam focus (smallest spot achievable): [fill in]
- q-range (accessible): [fill in, low-q and high-q limits]

### Flux and detection

- Photon flux at sample (typical): [fill in]
- Count rate saturation: [fill in]
- Readout noise floor: [fill in]

### Rheology integration

- Rheo-XPCS cell availability: [availability status, approximate wait time if not immediate]
- Supported geometries: [list]
- Shear rate range (stable): [fill in, lower and upper]
- Torque resolution: [fill in]
- Gap setting range: [fill in]

## Limitations and known issues

[Running list of known limitations to watch for in any capability map]

- [example: "beam damage at flux > X photons/s/μm^2 for aqueous silica suspensions"]
- [example: "rheology gap setting below 200 μm known to have parallelism issues"]

## Recent beamtime records

Pointers to where recent performance data live:

- [path or DOI to most recent beamline performance note]
- [path to Wei's own recent XPCS data with measured noise floor for comparison]

## How to use this file

When building an instrument capability map in the experiment-designer stage:

1. Open this file for the baseline
2. For each dimension in the capability map, copy the relevant number from here
3. Adjust if recent beamtime experience suggests a tighter or looser bound
4. Note any adjustment in the capability map with a one-line reason

## Update cadence

After every Wei beamtime run, update the "recent beamtime records" section with a pointer to the data. Update the baseline section when performance has demonstrably shifted from what is recorded here.
