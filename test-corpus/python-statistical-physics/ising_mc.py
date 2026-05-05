"""2D Ising model Monte Carlo simulation using the Metropolis algorithm."""

import numpy as np
from numpy.typing import NDArray


def initialize_lattice(L: int, seed: int = 42) -> NDArray[np.int8]:
    """Initialize an L x L spin lattice with random +/-1 spins."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


def compute_energy(lattice: NDArray[np.int8], J: float = 1.0) -> float:
    """Compute total Hamiltonian H = -J sum_{<ij>} s_i s_j."""
    L = lattice.shape[0]
    right = np.roll(lattice, -1, axis=1)
    down = np.roll(lattice, -1, axis=0)
    return -J * np.sum(lattice * right + lattice * down)


def compute_magnetization(lattice: NDArray[np.int8]) -> float:
    """Compute magnetization per spin m = (1/N) sum_i s_i."""
    return np.mean(lattice)


def metropolis_step(
    lattice: NDArray[np.int8],
    beta: float,
    J: float,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """Perform one full Metropolis sweep over the lattice."""
    L = lattice.shape[0]
    for _ in range(L * L):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        s = lattice[i, j]
        neighbors = (
            lattice[(i + 1) % L, j]
            + lattice[(i - 1) % L, j]
            + lattice[i, (j + 1) % L]
            + lattice[i, (j - 1) % L]
        )
        delta_E = 2.0 * J * s * neighbors
        if delta_E <= 0 or rng.random() < np.exp(-beta * delta_E):
            lattice[i, j] = -s
    return lattice


def run_simulation(
    L: int = 32,
    temperatures: NDArray[np.float64] | None = None,
    n_equilibrate: int = 1000,
    n_measure: int = 5000,
    J: float = 1.0,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Run Ising MC simulation across temperatures, measuring thermodynamic observables."""
    if temperatures is None:
        T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))  # Onsager critical temperature
        temperatures = np.linspace(1.0, 4.0, 30)

    rng = np.random.default_rng(seed)
    N = L * L
    energies = np.zeros(len(temperatures))
    magnetizations = np.zeros(len(temperatures))
    specific_heats = np.zeros(len(temperatures))
    susceptibilities = np.zeros(len(temperatures))

    for t_idx, T in enumerate(temperatures):
        beta = 1.0 / T
        lattice = initialize_lattice(L, seed=rng.integers(0, 2**31))

        for _ in range(n_equilibrate):
            lattice = metropolis_step(lattice, beta, J, rng)

        E_samples = np.zeros(n_measure)
        M_samples = np.zeros(n_measure)
        for n in range(n_measure):
            lattice = metropolis_step(lattice, beta, J, rng)
            E_samples[n] = compute_energy(lattice, J) / N
            M_samples[n] = np.abs(compute_magnetization(lattice))

        energies[t_idx] = np.mean(E_samples)
        magnetizations[t_idx] = np.mean(M_samples)
        specific_heats[t_idx] = beta**2 * np.var(E_samples) * N
        susceptibilities[t_idx] = beta * np.var(M_samples) * N

    return {
        "temperatures": temperatures,
        "energy": energies,
        "magnetization": magnetizations,
        "specific_heat": specific_heats,
        "susceptibility": susceptibilities,
    }
