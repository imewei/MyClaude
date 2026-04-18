# Dimensional analysis worksheet

Buckingham Pi-theorem workflow. Use it on the final governing equation(s) from the derivation. The dimensionless groups that fall out are the parameters Stage 6 will sweep.

## Step 1: List all variables

For every variable that appears in the governing equations, list:
- Symbol
- Physical meaning
- Dimensions (in M, L, T, and if needed Q for charge, Theta for temperature)

Example:
| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| $\eta$ | viscosity | M L^{-1} T^{-1} |
| $\rho$ | density | M L^{-3} |
| $U$ | velocity scale | L T^{-1} |
| $L$ | length scale | L |
| $k_B T$ | thermal energy | M L^2 T^{-2} |

## Step 2: Count variables and dimensions

- Number of variables: n
- Number of independent dimensions represented: k
- Number of dimensionless groups: n - k

If n - k disagrees with the number you expected, either you are missing a variable or you have an unnecessary one. Either way, investigate before proceeding.

## Step 3: Choose repeating variables

Select k variables that together span all k dimensions. These become the "repeating set" used to non-dimensionalize the others.

For fluid problems, a common choice is $\{\rho, U, L\}$; for colloidal problems, $\{k_B T, a, \eta\}$ where a is particle radius.

## Step 4: Form the dimensionless groups

For each non-repeating variable, find the exponents on the repeating set that cancel dimensions. This produces one dimensionless group per non-repeating variable.

Example (Reynolds number): from $\{\rho, U, L\}$ and the non-repeating $\eta$:
$$\Pi = \eta \cdot \rho^a U^b L^c$$
Solving for (a, b, c) to kill M, L, T: a = -1, b = -1, c = -1, giving $\Pi = \eta / (\rho U L) = 1/\text{Re}$.

## Step 5: Interpret and name each group

Every dimensionless group should get a physical interpretation. Canonical names when they exist:
- Reynolds: inertia vs viscous
- Péclet: advection vs diffusion
- Weissenberg: elastic vs viscous
- Deborah: relaxation time vs observation time
- Damköhler: reaction vs transport

Unnamed groups usually represent ratios specific to the problem. Name them explicitly in the paper's notation.

## Step 6: Check against the derivation

Rewrite the governing equation in dimensionless form using the groups. Every coefficient in the dimensionless equation should be a Pi-group or a pure number.

If a coefficient remains dimensional, either:
- A variable is missing from your list (add it and redo)
- An assumption is hiding a dimensional scale (add it to the assumptions list)

## Worked example: Generalized Stokes-Einstein

Variables: MSD $\langle \Delta r^2 \rangle$ (L^2), time $t$ (T), thermal energy $k_B T$ (ML^2 T^{-2}), probe radius $a$ (L), viscoelastic modulus $G^*$ (ML^{-1} T^{-2}).

n = 5, k = 3 (M, L, T), so n - k = 2 groups.

Repeating set: $\{k_B T, a, t\}$.

Groups:
- $\Pi_1 = \langle \Delta r^2 \rangle / a^2$ (dimensionless MSD)
- $\Pi_2 = G^* a^3 / (k_B T)$ (dimensionless modulus)

The GSE relation in dimensionless form links $\Pi_1$ and $\Pi_2$ with no remaining dimensional coefficients.

## When things go wrong

- **Negative group count:** You have more dimensions than variables, which usually means you have listed redundant or derived variables. Reduce.
- **Zero group count:** The problem is fully determined by dimensional analysis alone; there is no theoretical scaffolding to build, only a numerical coefficient to measure.
- **Group with dimensions left over:** A variable was miscounted. Recheck dimensions.
- **Groups that do not appear anywhere in the derivation:** Either they are irrelevant (drop them) or the derivation missed a dependence (revisit).
