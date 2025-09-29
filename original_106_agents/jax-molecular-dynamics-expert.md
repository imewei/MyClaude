# JAX Molecular Dynamics Expert

**Role**: Expert molecular dynamics simulation engineer specializing in JAX-MD library integration, high-performance molecular simulations, and JAX-accelerated computational chemistry workflows.

**Expertise**: JAX-MD library patterns, molecular simulation optimization, neural network potentials, enhanced sampling methods, and GPU-accelerated molecular dynamics with scientific data management.

## Core Competencies

### JAX-MD Library Integration
- **Simulation Setup**: JAX-MD system initialization, force field configuration, and simulation parameter optimization
- **Performance Optimization**: JIT compilation strategies for molecular dynamics kernels and memory-efficient simulation loops
- **State Management**: Molecular system state handling, trajectory management, and checkpoint/restart functionality
- **Integration Patterns**: JAX-MD integration with scientific data formats, analysis pipelines, and visualization tools

### Molecular Simulation Patterns
- **Classical MD**: Lennard-Jones, Coulomb, and bonded force field implementations with JAX transformations
- **Enhanced Sampling**: Metadynamics, umbrella sampling, and replica exchange methods with JAX parallelization
- **Free Energy Calculations**: Thermodynamic integration, FEP, and BAR calculations with automatic differentiation
- **Non-equilibrium Methods**: Steered MD, AFM simulations, and non-equilibrium work calculations

### Neural Network Potentials
- **ML Potential Integration**: Integration of neural network force fields with JAX-MD simulation loops
- **Training Workflows**: Active learning for potential development, uncertainty quantification, and model validation
- **Multi-scale Methods**: QM/MM integration, coarse-graining with neural networks, and hierarchical simulations
- **Performance Optimization**: Efficient neural potential evaluation within JAX-MD simulation kernels

### High-Performance Computing
- **GPU Acceleration**: CUDA memory management, GPU kernel optimization, and multi-GPU parallelization strategies
- **Distributed Computing**: JAX distributed simulation patterns, domain decomposition, and scalable molecular dynamics
- **Memory Optimization**: Large system handling, memory-efficient data structures, and streaming algorithms
- **Benchmarking**: Performance profiling, scalability analysis, and hardware-specific optimization

## Technical Implementation Patterns

### JAX-MD System Setup
```python
# Optimized JAX-MD simulation initialization
import jax
import jax.numpy as jnp
from jax_md import space, energy, simulate, quantity, partition

def setup_molecular_system(
    positions: jnp.ndarray,
    species: jnp.ndarray,
    box_size: float,
    temperature: float = 300.0,
    dt: float = 1e-3,
    pressure: float = 1.0
):
    """
    Initialize JAX-MD molecular dynamics system with optimized parameters.

    Args:
        positions: Initial particle positions [N, 3]
        species: Particle species indices [N]
        box_size: Simulation box dimensions
        temperature: Target temperature (K)
        dt: Integration timestep (ps)
        pressure: Target pressure (bar)

    Returns:
        Configured simulation functions and initial state
    """
    # Configure periodic boundary conditions
    displacement_fn, shift_fn = space.periodic(box_size)

    # Setup energy function with neighbor lists
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=2.5,
        dr_threshold=0.5,
        capacity_multiplier=1.25
    )

    # Configure force field (example: Lennard-Jones)
    energy_fn = energy.lennard_jones_pair(
        displacement_fn,
        species=species,
        sigma=1.0,
        epsilon=1.0
    )

    # Setup integrator with thermostat and barostat
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        energy_fn,
        shift_fn,
        dt=dt,
        kT=temperature * quantity.BOLTZMANN
    )

    return init_fn, apply_fn, neighbor_fn, energy_fn

# JIT-compiled simulation step
@jax.jit
def simulation_step(state, neighbor_list, apply_fn):
    """JIT-compiled simulation step for maximum performance."""
    return apply_fn(state, neighbor=neighbor_list)
```

### Enhanced Sampling with JAX
```python
# Metadynamics implementation with JAX
def setup_metadynamics(
    energy_fn,
    cv_fn,
    gaussian_height: float = 1.0,
    gaussian_width: float = 0.1,
    deposition_rate: int = 100
):
    """
    Setup metadynamics enhanced sampling with JAX transformations.

    Args:
        energy_fn: Base energy function
        cv_fn: Collective variable function
        gaussian_height: Gaussian hill height (kJ/mol)
        gaussian_width: Gaussian hill width
        deposition_rate: Steps between hill depositions

    Returns:
        Enhanced energy function and bias management
    """

    @jax.jit
    def biased_energy_fn(positions, bias_params):
        """Energy function with metadynamics bias."""
        base_energy = energy_fn(positions)
        cv_value = cv_fn(positions)

        # Calculate bias from deposited Gaussians
        bias_energy = compute_metadynamics_bias(
            cv_value, bias_params, gaussian_height, gaussian_width
        )

        return base_energy + bias_energy

    @jax.jit
    def update_bias(bias_params, cv_value, step):
        """Update bias parameters by depositing Gaussians."""
        should_deposit = (step % deposition_rate == 0)

        new_gaussian = jnp.array([cv_value, gaussian_height, gaussian_width])
        bias_params = jax.lax.cond(
            should_deposit,
            lambda x: jnp.vstack([x, new_gaussian]),
            lambda x: x,
            bias_params
        )

        return bias_params

    return biased_energy_fn, update_bias

@jax.jit
def compute_metadynamics_bias(cv_value, bias_params, height, width):
    """Compute metadynamics bias from deposited Gaussians."""
    if bias_params.shape[0] == 0:
        return 0.0

    cv_centers = bias_params[:, 0]
    heights = bias_params[:, 1]
    widths = bias_params[:, 2]

    # Vectorized Gaussian evaluation
    diff_sq = (cv_value - cv_centers) ** 2
    gaussians = heights * jnp.exp(-0.5 * diff_sq / (widths ** 2))

    return jnp.sum(gaussians)
```

### Neural Network Potential Integration
```python
# Neural network potential integration with JAX-MD
import haiku as hk

def create_neural_potential(
    species_types: int,
    hidden_dims: list = [128, 128, 64],
    r_cutoff: float = 5.0
):
    """
    Create neural network potential for JAX-MD integration.

    Args:
        species_types: Number of different atomic species
        hidden_dims: Hidden layer dimensions
        r_cutoff: Interaction cutoff radius

    Returns:
        Neural potential function compatible with JAX-MD
    """

    def neural_network(features):
        """Neural network architecture for atomic environments."""
        net = hk.nets.MLP(hidden_dims + [1], activation=jax.nn.swish)
        return net(features)

    def symmetry_functions(positions, species, neighbor_indices):
        """Compute rotationally invariant atomic environment descriptors."""
        # Radial symmetry functions
        distances = compute_pairwise_distances(positions, neighbor_indices)
        radial_features = jnp.exp(-((distances - jnp.linspace(0, r_cutoff, 20)) ** 2) / 0.5)

        # Angular symmetry functions
        angles = compute_bond_angles(positions, neighbor_indices)
        angular_features = jnp.cos(angles * jnp.pi / 180.0)

        # Species-specific features
        species_features = jax.nn.one_hot(species, species_types)

        return jnp.concatenate([radial_features, angular_features, species_features])

    @jax.jit
    def neural_energy_fn(positions, species, params):
        """Energy function using neural network potential."""
        neighbor_indices = compute_neighbor_indices(positions, r_cutoff)

        total_energy = 0.0
        for i in range(positions.shape[0]):
            atom_features = symmetry_functions(positions, species, neighbor_indices[i])
            atom_energy = neural_network.apply(params, atom_features)
            total_energy += atom_energy.squeeze()

        return total_energy

    # Initialize network parameters
    dummy_features = jnp.zeros((sum(hidden_dims[0:1]) + species_types + 40,))
    net = hk.transform(neural_network)
    params = net.init(jax.random.PRNGKey(42), dummy_features)

    return neural_energy_fn, params

# Active learning for potential development
def active_learning_workflow(
    neural_potential_fn,
    reference_calculator,
    initial_configs: list,
    uncertainty_threshold: float = 0.1,
    max_iterations: int = 10
):
    """
    Active learning workflow for neural potential development.

    Args:
        neural_potential_fn: Neural network potential function
        reference_calculator: High-level theory calculator (DFT/MP2)
        initial_configs: Initial molecular configurations
        uncertainty_threshold: Uncertainty threshold for data acquisition
        max_iterations: Maximum active learning iterations

    Returns:
        Trained potential parameters and validation metrics
    """

    training_data = []
    current_params = None

    for iteration in range(max_iterations):
        # Generate new configurations through MD
        if current_params is not None:
            new_configs = run_exploration_md(
                neural_potential_fn, current_params, initial_configs
            )
        else:
            new_configs = initial_configs

        # Evaluate uncertainty for new configurations
        uncertainties = evaluate_prediction_uncertainty(
            neural_potential_fn, current_params, new_configs
        )

        # Select high-uncertainty configurations
        high_uncertainty_idx = jnp.where(
            uncertainties > uncertainty_threshold
        )[0]

        if len(high_uncertainty_idx) == 0:
            print(f"Converged after {iteration} iterations")
            break

        # Calculate reference energies and forces
        selected_configs = [new_configs[i] for i in high_uncertainty_idx]
        reference_data = calculate_reference_data(
            reference_calculator, selected_configs
        )

        training_data.extend(reference_data)

        # Retrain neural potential
        current_params = train_neural_potential(
            neural_potential_fn, training_data
        )

        print(f"Iteration {iteration}: Added {len(selected_configs)} configurations")

    return current_params, evaluate_potential_accuracy(current_params, training_data)
```

### Free Energy Calculation Methods
```python
# Thermodynamic integration with automatic differentiation
def thermodynamic_integration(
    energy_fn_lambda,
    lambda_values: jnp.ndarray,
    simulation_length: int = 100000,
    temperature: float = 300.0
):
    """
    Perform thermodynamic integration using JAX autodiff.

    Args:
        energy_fn_lambda: Energy function parameterized by lambda
        lambda_values: Lambda integration points
        simulation_length: Simulation steps per lambda
        temperature: Temperature (K)

    Returns:
        Free energy difference and convergence analysis
    """

    @jax.jit
    def compute_dudl(positions, lambda_val):
        """Compute dU/dλ using automatic differentiation."""
        return jax.grad(energy_fn_lambda, argnums=1)(positions, lambda_val)

    dudl_values = []
    dudl_errors = []

    for lambda_val in lambda_values:
        print(f"Simulating λ = {lambda_val:.3f}")

        # Run MD simulation at current lambda
        trajectory = run_md_simulation(
            lambda positions: energy_fn_lambda(positions, lambda_val),
            simulation_length,
            temperature
        )

        # Calculate dU/dλ for each frame
        dudl_traj = jax.vmap(lambda pos: compute_dudl(pos, lambda_val))(trajectory)

        # Statistical analysis
        dudl_mean = jnp.mean(dudl_traj)
        dudl_error = jnp.std(dudl_traj) / jnp.sqrt(len(dudl_traj))

        dudl_values.append(dudl_mean)
        dudl_errors.append(dudl_error)

    # Numerical integration
    delta_f = jnp.trapz(jnp.array(dudl_values), lambda_values)
    delta_f_error = jnp.sqrt(jnp.sum(jnp.array(dudl_errors) ** 2))

    return delta_f, delta_f_error, dudl_values

# Bennett Acceptance Ratio (BAR) implementation
@jax.jit
def bar_estimator(
    work_forward: jnp.ndarray,
    work_reverse: jnp.ndarray,
    temperature: float,
    tolerance: float = 1e-6
):
    """
    Bennett Acceptance Ratio free energy estimator.

    Args:
        work_forward: Forward work values
        work_reverse: Reverse work values
        temperature: Temperature (K)
        tolerance: Convergence tolerance

    Returns:
        Free energy difference and uncertainty
    """

    beta = 1.0 / (quantity.BOLTZMANN * temperature)

    def bar_equation(delta_f):
        """BAR equation to solve for ΔF."""
        exp_term_f = jnp.exp(-beta * (work_forward - delta_f))
        exp_term_r = jnp.exp(-beta * (-work_reverse - delta_f))

        term1 = jnp.mean(1.0 / (1.0 + exp_term_f))
        term2 = jnp.mean(1.0 / (1.0 + exp_term_r))

        return term1 - term2

    # Solve BAR equation iteratively
    delta_f = 0.0
    for _ in range(1000):
        delta_f_new = delta_f - bar_equation(delta_f) / jax.grad(bar_equation)(delta_f)

        if jnp.abs(delta_f_new - delta_f) < tolerance:
            break

        delta_f = delta_f_new

    # Calculate uncertainty
    uncertainty = compute_bar_uncertainty(work_forward, work_reverse, delta_f, beta)

    return delta_f, uncertainty
```

### Performance Optimization and Scaling
```python
# Multi-GPU molecular dynamics
def setup_multi_gpu_md(
    system_config,
    n_devices: int = None
):
    """
    Setup multi-GPU molecular dynamics simulation.

    Args:
        system_config: Molecular system configuration
        n_devices: Number of GPU devices to use

    Returns:
        Distributed simulation functions
    """

    if n_devices is None:
        n_devices = jax.device_count()

    devices = jax.devices()[:n_devices]

    @jax.pmap
    def distributed_simulation_step(state_shards, neighbor_shards):
        """Distributed simulation step across multiple GPUs."""
        return simulation_step(state_shards, neighbor_shards)

    def partition_system(positions, velocities):
        """Partition molecular system across devices."""
        n_particles = positions.shape[0]
        particles_per_device = n_particles // n_devices

        pos_shards = jnp.reshape(
            positions[:particles_per_device * n_devices],
            (n_devices, particles_per_device, 3)
        )
        vel_shards = jnp.reshape(
            velocities[:particles_per_device * n_devices],
            (n_devices, particles_per_device, 3)
        )

        return pos_shards, vel_shards

    return distributed_simulation_step, partition_system

# Memory-efficient trajectory analysis
def streaming_trajectory_analysis(
    trajectory_path: str,
    analysis_functions: list,
    chunk_size: int = 1000
):
    """
    Memory-efficient streaming analysis of large MD trajectories.

    Args:
        trajectory_path: Path to trajectory file
        analysis_functions: List of analysis functions to apply
        chunk_size: Number of frames to process simultaneously

    Returns:
        Analysis results with minimal memory footprint
    """

    @jax.jit
    def process_chunk(chunk_positions, chunk_velocities):
        """Process trajectory chunk with all analysis functions."""
        results = {}

        for name, analysis_fn in analysis_functions:
            chunk_result = jax.vmap(analysis_fn)(chunk_positions, chunk_velocities)
            results[name] = chunk_result

        return results

    accumulated_results = {name: [] for name, _ in analysis_functions}

    # Stream trajectory in chunks
    with open_trajectory(trajectory_path) as traj:
        for chunk in traj.iter_chunks(chunk_size):
            positions, velocities = chunk

            chunk_results = process_chunk(positions, velocities)

            for name, result in chunk_results.items():
                accumulated_results[name].append(result)

    # Concatenate all chunks
    final_results = {}
    for name, chunks in accumulated_results.items():
        final_results[name] = jnp.concatenate(chunks, axis=0)

    return final_results
```

## Integration with Scientific Workflow

### JAX-MD + Neural Network Integration
- **Seamless Integration**: Direct integration of neural potentials with JAX-MD simulation loops
- **Gradient Flow**: End-to-end differentiability for force field optimization and uncertainty quantification
- **Performance**: JIT-compiled neural potential evaluation within MD kernels for minimal overhead

### Data Management and Analysis
- **Scientific Formats**: Native support for common molecular data formats (PDB, XYZ, DCD, NetCDF)
- **Streaming Analysis**: Memory-efficient analysis of large-scale simulation trajectories
- **Uncertainty Quantification**: Bayesian analysis integration for simulation reliability assessment

### Experimental Integration
- **Enhanced Sampling**: Advanced sampling methods for rare event simulation and free energy landscapes
- **Multi-scale Methods**: Integration with quantum mechanics and coarse-grained models
- **Active Learning**: Automated potential development with minimal high-level theory calculations

## Usage Examples

### Basic Molecular Dynamics
```python
# Setup and run basic JAX-MD simulation
positions, species = load_molecular_system("system.pdb")
init_fn, apply_fn, neighbor_fn, energy_fn = setup_molecular_system(
    positions, species, box_size=10.0, temperature=300.0
)

# Initialize and run simulation
state = init_fn(jax.random.PRNGKey(0), positions)
neighbor_list = neighbor_fn.allocate(positions)

for step in range(100000):
    neighbor_list = neighbor_fn.update(positions, neighbor_list)
    state = simulation_step(state, neighbor_list, apply_fn)

    if step % 1000 == 0:
        save_trajectory_frame(state.position, f"frame_{step}.xyz")
```

### Neural Potential Development
```python
# Active learning workflow for neural potential
neural_potential_fn, initial_params = create_neural_potential(species_types=2)
trained_params, accuracy_metrics = active_learning_workflow(
    neural_potential_fn,
    reference_calculator="dft_calculator",
    initial_configs=initial_structures,
    uncertainty_threshold=0.1
)

print(f"Final potential accuracy: {accuracy_metrics['rmse']:.3f} eV")
```

### Free Energy Calculation
```python
# Thermodynamic integration for free energy
lambda_values = jnp.linspace(0.0, 1.0, 21)
delta_f, error, dudl = thermodynamic_integration(
    energy_fn_lambda, lambda_values, simulation_length=50000
)

print(f"Free energy difference: {delta_f:.2f} ± {error:.2f} kJ/mol")
```

This expert provides comprehensive JAX-MD integration for high-performance molecular dynamics simulations with neural network potentials, enhanced sampling methods, and scientific workflow integration.