# Framing variants

Every spark can be framed at least three ways. The framing changes what the pipeline optimizes for downstream. Pick the framing that matches the methodological tradition you can most readily execute in.

## Mechanistic framing

Leads with what causes what.

**Template:** "When [conditions], [mechanism] produces [observable]. This has not been recognized because [prior blind spot]. The signature is [falsifiable measurement]."

**Worked example (battery slurry):**
> In concentrated battery-electrode slurries, the eigenvalue spacing of the linearized stress-response operator collapses before bulk rheological indicators show the approach to flocculation. The collapse has not been used as an early warning because ensemble averaging in standard oscillatory rheology obscures the spectral structure. Under Rheo-XPCS, the spectral gap should drop by more than an order of magnitude at least 30 seconds before the storage modulus crossover.

## Predictive framing

Leads with what can be forecast.

**Template:** "Given [inputs], we predict [output] to within [precision]. Current methods achieve [baseline precision or cannot predict at all]. The new element that enables this is [mechanism or method]."

**Worked example (same spark, different frame):**
> Given a two-minute window of Rheo-XPCS two-time correlation functions, we can predict the onset of flocculation in battery-electrode slurries at least 30 seconds ahead with a false-positive rate below 5%. Existing predictors based on G'/G'' crossover have zero lead time by construction. The new element is the spectral gap of the stress-response operator, extracted by operator-valued regression from the XPCS tensor.

## Design-oriented framing

Leads with what can be built or controlled.

**Template:** "By [manipulation], we can achieve [target state]. The route is [mechanism or design principle]. The test is [measurement demonstrating the target state was reached]."

**Worked example (same spark, third frame):**
> By modulating shear history in the last two minutes of slurry processing, we can delay flocculation onset by at least 60 seconds at fixed solids loading. The route is feedback control on the spectral gap of the stress-response operator, which we show drops monotonically as flocculation approaches. The test is repeated cycles showing gap-informed shear intervention extends the processable window relative to a gap-blind baseline.

## Which framing to pick

- Pick **mechanistic** if your strongest contribution is theoretical insight into why something happens.
- Pick **predictive** if your strongest contribution is a forecast with quantified precision, regardless of whether the mechanism is fully understood.
- Pick **design-oriented** if your strongest contribution is a working intervention, regardless of whether the mechanism is fully understood.

Most sparks fit more than one framing. Pick the one that makes the downstream work most tractable given what you can actually measure and build. The rejected framings are preserved in the artifact because they often resurface in the paper's Discussion section.
