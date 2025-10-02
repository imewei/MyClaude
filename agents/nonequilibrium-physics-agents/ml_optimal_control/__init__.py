"""Machine Learning for Optimal Control.

This module provides neural network architectures and training algorithms
for hybrid physics + ML approaches to optimal control problems.

Key Components (Week 5):
- Actor-Critic networks (Flax/JAX)
- PPO (Proximal Policy Optimization) for control
- PINN (Physics-Informed Neural Networks) for HJB equation
- RL environments for thermodynamic systems
- Neural network warm starts for PMP solvers

Advanced Components (Week 6):
- SAC, TD3, DDPG for continuous control
- Model-based RL (world models, MPC)
- Meta-learning (MAML, Reptile)
- Context-based adaptation

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

__version__ = "4.1.0-dev"

# Neural network architectures
try:
    from .networks import (
        ActorCriticNetwork,
        PINNNetwork,
        ValueNetwork,
        PolicyNetwork
    )
    NETWORKS_AVAILABLE = True
except ImportError:
    NETWORKS_AVAILABLE = False
    ActorCriticNetwork = None
    PINNNetwork = None
    ValueNetwork = None
    PolicyNetwork = None

# Training algorithms
try:
    from .training import (
        PPOTrainer,
        PINNTrainer,
        train_actor_critic,
        train_pinn
    )
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    PPOTrainer = None
    PINNTrainer = None
    train_actor_critic = None
    train_pinn = None

# RL environments
try:
    from .environments import (
        OptimalControlEnv,
        QuantumControlEnv,
        ThermodynamicEnv
    )
    ENVIRONMENTS_AVAILABLE = True
except ImportError:
    ENVIRONMENTS_AVAILABLE = False
    OptimalControlEnv = None
    QuantumControlEnv = None
    ThermodynamicEnv = None

# Advanced RL algorithms (Week 6)
try:
    from .advanced_rl import (
        DDPGTrainer,
        TD3Trainer,
        SACTrainer,
        ReplayBuffer,
        create_ddpg_trainer,
        create_td3_trainer,
        create_sac_trainer
    )
    ADVANCED_RL_AVAILABLE = True
except ImportError:
    ADVANCED_RL_AVAILABLE = False
    DDPGTrainer = None
    TD3Trainer = None
    SACTrainer = None
    ReplayBuffer = None
    create_ddpg_trainer = None
    create_td3_trainer = None
    create_sac_trainer = None

# Model-based RL (Week 6)
try:
    from .model_based_rl import (
        DynamicsModelTrainer,
        ModelPredictiveControl,
        DynaAgent,
        ModelBasedValueExpansion,
        create_dynamics_model,
        create_mpc_controller
    )
    MODEL_BASED_RL_AVAILABLE = True
except ImportError:
    MODEL_BASED_RL_AVAILABLE = False
    DynamicsModelTrainer = None
    ModelPredictiveControl = None
    DynaAgent = None
    ModelBasedValueExpansion = None
    create_dynamics_model = None
    create_mpc_controller = None

# Meta-learning (Week 6)
try:
    from .meta_learning import (
        Task,
        TaskDistribution,
        MAMLTrainer,
        ReptileTrainer,
        ContextBasedAdapter,
        create_task_distribution
    )
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False
    Task = None
    TaskDistribution = None
    MAMLTrainer = None
    ReptileTrainer = None
    ContextBasedAdapter = None
    create_task_distribution = None

__all__ = [
    # Networks
    'ActorCriticNetwork',
    'PINNNetwork',
    'ValueNetwork',
    'PolicyNetwork',
    'NETWORKS_AVAILABLE',

    # Training
    'PPOTrainer',
    'PINNTrainer',
    'train_actor_critic',
    'train_pinn',
    'TRAINING_AVAILABLE',

    # Environments
    'OptimalControlEnv',
    'QuantumControlEnv',
    'ThermodynamicEnv',
    'ENVIRONMENTS_AVAILABLE',

    # Advanced RL
    'DDPGTrainer',
    'TD3Trainer',
    'SACTrainer',
    'ReplayBuffer',
    'create_ddpg_trainer',
    'create_td3_trainer',
    'create_sac_trainer',
    'ADVANCED_RL_AVAILABLE',

    # Model-Based RL
    'DynamicsModelTrainer',
    'ModelPredictiveControl',
    'DynaAgent',
    'ModelBasedValueExpansion',
    'create_dynamics_model',
    'create_mpc_controller',
    'MODEL_BASED_RL_AVAILABLE',

    # Meta-Learning
    'Task',
    'TaskDistribution',
    'MAMLTrainer',
    'ReptileTrainer',
    'ContextBasedAdapter',
    'create_task_distribution',
    'META_LEARNING_AVAILABLE',
]
