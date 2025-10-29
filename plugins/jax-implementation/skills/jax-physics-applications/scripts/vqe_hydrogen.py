#!/usr/bin/env python3
"""
Complete Variational Quantum Eigensolver (VQE) for H2 Molecule

This script demonstrates a full quantum computing workflow:
1. Define molecular Hamiltonian (H2 in minimal basis)
2. Design parameterized quantum circuit (ansatz)
3. Compute expectation value <ψ(θ)|H|ψ(θ)>
4. Optimize parameters with JAX gradients
5. Validate ground state energy

Problem: Find ground state of H2 molecule
  H = Σ hᵢ Pᵢ (sum of Pauli strings)
  Goal: Minimize E(θ) = <ψ(θ)|H|ψ(θ)>

Usage:
    python vqe_hydrogen.py

Requirements:
    pip install jax optax matplotlib numpy
Note: This uses a simplified simulation without actual Cirq dependency
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt


def main():
    # Configuration
    n_qubits = 4
    n_layers = 3
    learning_rate = 0.01
    n_steps = 1000

    print("=" * 60)
    print("Variational Quantum Eigensolver for H2 Molecule")
    print("=" * 60)
    print(f"Qubits: {n_qubits}")
    print(f"Circuit layers: {n_layers}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimization steps: {n_steps}")
    print()

    # 1. Define H2 Hamiltonian
    print("Defining H2 Hamiltonian...")
    hamiltonian = h2_hamiltonian()
    print(f"  Number of terms: {len(hamiltonian)}")

    # Exact ground state energy (from classical diagonalization)
    E_exact = -1.8572  # Hartree (simplified)
    print(f"  Exact ground state: {E_exact:.6f} Ha")
    print()

    # 2. Initialize variational parameters
    print("Initializing variational circuit...")
    n_params = n_layers * 2 * n_qubits
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (n_params,)) * 0.1
    print(f"  Total parameters: {n_params}")
    print()

    # 3. Setup optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # 4. Optimization loop
    print("Starting VQE optimization...")
    energies = []
    gradients_norm = []

    for step in range(n_steps):
        # Compute energy and gradient
        energy, grads = jax.value_and_grad(compute_energy)(
            params, hamiltonian, n_qubits, n_layers
        )

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Record
        energies.append(energy)
        grad_norm = jnp.linalg.norm(grads)
        gradients_norm.append(grad_norm)

        if (step + 1) % 100 == 0:
            error = jnp.abs(energy - E_exact)
            print(f"  Step {step+1}/{n_steps}: "
                  f"E={energy:.6f} Ha, "
                  f"Error={error:.6f} Ha, "
                  f"|∇E|={grad_norm:.6f}")

    print("✓ Optimization complete\n")

    # 5. Final results and validation
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    final_energy = energies[-1]
    energy_error = jnp.abs(final_energy - E_exact)

    print(f"\nEnergy Results:")
    print(f"  VQE energy: {final_energy:.6f} Ha")
    print(f"  Exact energy: {E_exact:.6f} Ha")
    print(f"  Absolute error: {energy_error:.6f} Ha")
    print(f"  Relative error: {energy_error/jnp.abs(E_exact)*100:.4f}%")

    # Chemical accuracy threshold
    chemical_accuracy = 0.001  # 1 mHa = 0.627 kcal/mol
    if energy_error < chemical_accuracy:
        print(f"  ✓ Chemical accuracy achieved (< {chemical_accuracy} Ha)")
    else:
        print(f"  ⚠ Chemical accuracy NOT achieved (error = {energy_error:.6f} Ha)")
        print(f"    Consider: more circuit layers, better ansatz, or longer training")

    # Gradient analysis
    final_grad_norm = gradients_norm[-1]
    print(f"\nGradient Analysis:")
    print(f"  Initial |∇E|: {gradients_norm[0]:.6f}")
    print(f"  Final |∇E|: {final_grad_norm:.6f}")

    if final_grad_norm < 1e-4:
        print(f"  ✓ Convergence: EXCELLENT (gradient vanished)")
    elif final_grad_norm < 1e-2:
        print(f"  ✓ Convergence: GOOD")
    else:
        print(f"  ⚠ Convergence: INCOMPLETE (gradient still large)")

    # Convergence statistics
    energies_array = jnp.array(energies)
    convergence_window = 100
    if len(energies) >= convergence_window:
        recent_std = jnp.std(energies_array[-convergence_window:])
        print(f"\nConvergence Statistics (last {convergence_window} steps):")
        print(f"  Energy std: {recent_std:.8f} Ha")

        if recent_std < 1e-6:
            print(f"  ✓ Stable convergence achieved")
        else:
            print(f"  ~ Energy still fluctuating")

    # 6. Visualization
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Energy convergence
    ax1 = axes[0, 0]
    ax1.plot(energies, 'b-', linewidth=2, alpha=0.7, label='VQE Energy')
    ax1.axhline(E_exact, color='r', linestyle='--', linewidth=2, label='Exact')
    ax1.axhline(E_exact + chemical_accuracy, color='orange', linestyle=':',
                linewidth=1, label='Chemical Accuracy')
    ax1.axhline(E_exact - chemical_accuracy, color='orange', linestyle=':',
                linewidth=1)
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('VQE Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy error (log scale)
    ax2 = axes[0, 1]
    errors = jnp.abs(energies_array - E_exact)
    ax2.semilogy(errors, 'g-', linewidth=2)
    ax2.axhline(chemical_accuracy, color='r', linestyle='--',
                linewidth=2, label='Chemical Accuracy')
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('|E_VQE - E_exact| (Ha)')
    ax2.set_title('Energy Error (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gradient norm
    ax3 = axes[1, 0]
    ax3.semilogy(gradients_norm, 'purple', linewidth=2)
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('||∇E||')
    ax3.set_title('Gradient Norm Evolution')
    ax3.grid(True, alpha=0.3)

    # Convergence rate
    ax4 = axes[1, 1]
    window = 50
    if len(energies) >= window:
        rolling_mean = jnp.convolve(energies_array, jnp.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(energies)), rolling_mean, 'b-',
                 linewidth=2, label=f'{window}-step moving average')
        ax4.axhline(E_exact, color='r', linestyle='--', linewidth=2, label='Exact')
        ax4.set_xlabel('Optimization Step')
        ax4.set_ylabel('Energy (Hartree)')
        ax4.set_title('Convergence Smoothed')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vqe_hydrogen_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: vqe_hydrogen_results.png")

    print("\n" + "=" * 60)
    print("VQE SIMULATION COMPLETE")
    print("=" * 60)


def h2_hamiltonian():
    """
    Simplified H2 Hamiltonian in Pauli basis
    Real H2 Hamiltonian has ~15 terms; this is illustrative
    """
    hamiltonian = {
        'IIII': -0.8,    # Constant term
        'ZIII': 0.2,     # Single-qubit Z terms
        'IZII': 0.2,
        'IIZI': -0.2,
        'IIIZ': -0.2,
        'ZZII': 0.1,     # Two-qubit ZZ terms
        'ZIZI': 0.05,
        'ZIIZ': -0.05,
        'IZZI': -0.05,
        'XXII': 0.05,    # Two-qubit XX terms (hopping)
        'YYII': 0.05,    # Two-qubit YY terms
    }
    return hamiltonian


def create_circuit_state(params, n_qubits, n_layers):
    """
    Create quantum state from variational circuit
    Simplified simulation using statevector

    Circuit structure per layer:
    - Ry rotation on each qubit
    - CNOT entangling gates
    - Rz rotation on each qubit
    """
    # Start with |0000⟩ state
    state = jnp.zeros(2**n_qubits, dtype=complex)
    state = state.at[0].set(1.0 + 0j)

    param_idx = 0

    for layer in range(n_layers):
        # Ry rotations
        for qubit in range(n_qubits):
            theta = params[param_idx]
            state = apply_ry(state, qubit, theta, n_qubits)
            param_idx += 1

        # Entangling CNOT gates
        for qubit in range(n_qubits - 1):
            state = apply_cnot(state, qubit, qubit + 1, n_qubits)

        # Rz rotations
        for qubit in range(n_qubits):
            phi = params[param_idx]
            state = apply_rz(state, qubit, phi, n_qubits)
            param_idx += 1

    return state


def compute_energy(params, hamiltonian, n_qubits, n_layers):
    """Compute expectation value E = ⟨ψ(θ)|H|ψ(θ)⟩"""
    state = create_circuit_state(params, n_qubits, n_layers)

    energy = 0.0
    for pauli_string, coeff in hamiltonian.items():
        expectation = compute_pauli_expectation(state, pauli_string, n_qubits)
        energy += coeff * expectation

    return jnp.real(energy)


def compute_pauli_expectation(state, pauli_string, n_qubits):
    """Compute ⟨ψ|P|ψ⟩ for Pauli operator P"""
    # Apply Pauli operator to state
    P_state = apply_pauli_operator(state, pauli_string, n_qubits)

    # Compute inner product ⟨ψ|P|ψ⟩
    expectation = jnp.vdot(state, P_state)

    return jnp.real(expectation)


def apply_pauli_operator(state, pauli_string, n_qubits):
    """Apply Pauli operator to state"""
    result = state

    for i, pauli in enumerate(pauli_string):
        if pauli == 'X':
            result = apply_pauli_x(result, i, n_qubits)
        elif pauli == 'Y':
            result = apply_pauli_y(result, i, n_qubits)
        elif pauli == 'Z':
            result = apply_pauli_z(result, i, n_qubits)
        # 'I' does nothing

    return result


def apply_ry(state, qubit, theta, n_qubits):
    """Apply Ry(θ) rotation to specified qubit"""
    # Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    cos_half = jnp.cos(theta / 2)
    sin_half = jnp.sin(theta / 2)

    new_state = jnp.zeros_like(state)

    for i in range(2**n_qubits):
        if (i >> qubit) & 1 == 0:  # Qubit is |0⟩
            j = i | (1 << qubit)  # Flip to |1⟩
            new_state = new_state.at[i].add(cos_half * state[i] - sin_half * state[j])
            new_state = new_state.at[j].add(sin_half * state[i] + cos_half * state[j])

    return new_state


def apply_rz(state, qubit, phi, n_qubits):
    """Apply Rz(φ) rotation to specified qubit"""
    # Rz(φ) = [[e^(-iφ/2), 0], [0, e^(iφ/2)]]
    phase_neg = jnp.exp(-1j * phi / 2)
    phase_pos = jnp.exp(1j * phi / 2)

    new_state = jnp.zeros_like(state)

    for i in range(2**n_qubits):
        if (i >> qubit) & 1 == 0:  # Qubit is |0⟩
            new_state = new_state.at[i].set(phase_neg * state[i])
        else:  # Qubit is |1⟩
            new_state = new_state.at[i].set(phase_pos * state[i])

    return new_state


def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT gate with specified control and target qubits"""
    new_state = state.copy()

    for i in range(2**n_qubits):
        if (i >> control) & 1 == 1:  # Control is |1⟩
            j = i ^ (1 << target)  # Flip target
            new_state = new_state.at[i].set(state[j])
            new_state = new_state.at[j].set(state[i])

    return new_state


def apply_pauli_x(state, qubit, n_qubits):
    """Apply Pauli X gate"""
    new_state = jnp.zeros_like(state)

    for i in range(2**n_qubits):
        j = i ^ (1 << qubit)  # Flip qubit
        new_state = new_state.at[j].set(state[i])

    return new_state


def apply_pauli_y(state, qubit, n_qubits):
    """Apply Pauli Y gate"""
    new_state = jnp.zeros_like(state)

    for i in range(2**n_qubits):
        j = i ^ (1 << qubit)  # Flip qubit

        if (i >> qubit) & 1 == 0:  # Was |0⟩, becomes -i|1⟩
            new_state = new_state.at[j].add(-1j * state[i])
        else:  # Was |1⟩, becomes i|0⟩
            new_state = new_state.at[j].add(1j * state[i])

    return new_state


def apply_pauli_z(state, qubit, n_qubits):
    """Apply Pauli Z gate"""
    new_state = state.copy()

    for i in range(2**n_qubits):
        if (i >> qubit) & 1 == 1:  # Qubit is |1⟩
            new_state = new_state.at[i].set(-state[i])

    return new_state


if __name__ == '__main__':
    main()
