---
name: julia-reinforcement-learning
description: Implement reinforcement learning in Julia with ReinforcementLearning.jl. Covers policy gradient methods (PPO, A2C), value-based methods (DQN, DDPG), custom environments, multi-agent RL, and integration with Lux.jl for policy networks. Use when building RL agents or custom environments in Julia.
---

# Julia Reinforcement Learning

## Expert Agent

For RL algorithm design and environment implementation in Julia, delegate to:

- **`julia-ml-hpc`** at `plugins/science-suite/agents/julia-ml-hpc.md`

## Environment Interface

### Built-in Environments

```julia
using ReinforcementLearning

env = CartPoleEnv()
env = MountainCarEnv()
env = PendulumEnv()
env = AtariEnv("Pong")
```

### Environment API

```julia
s = state(env)                     # Current observation
as = action_space(env)             # Available actions
r = reward(env)                    # Last reward
done = is_terminated(env)          # Terminal state?
reset!(env)                        # Reset to initial state
env(action)                        # Step environment
```

## Custom GridWorld Environment

Full `RLBase` interface implementation:

```julia
using ReinforcementLearning

mutable struct GridWorldEnv <: AbstractEnv
    pos::Tuple{Int,Int}
    goal::Tuple{Int,Int}
    size::Int
    reward::Float64
    done::Bool
end

GridWorldEnv(; size=5) = GridWorldEnv((1,1), (size,size), size, 0.0, false)

# Required RLBase interface
RLBase.state(env::GridWorldEnv) = [env.pos[1], env.pos[2]]
RLBase.state_space(env::GridWorldEnv) = Space([1..env.size, 1..env.size])
RLBase.action_space(env::GridWorldEnv) = Base.OneTo(4)  # up, down, left, right
RLBase.reward(env::GridWorldEnv) = env.reward
RLBase.is_terminated(env::GridWorldEnv) = env.done

function RLBase.reset!(env::GridWorldEnv)
    env.pos = (1, 1)
    env.reward = 0.0
    env.done = false
    nothing
end

function (env::GridWorldEnv)(action::Int)
    dx, dy = [(0,1), (0,-1), (-1,0), (1,0)][action]
    x = clamp(env.pos[1] + dx, 1, env.size)
    y = clamp(env.pos[2] + dy, 1, env.size)
    env.pos = (x, y)

    if env.pos == env.goal
        env.reward = 1.0
        env.done = true
    else
        env.reward = -0.01
    end
end
```

## DQN

Deep Q-Network with experience replay:

```julia
using ReinforcementLearning, Lux

env = CartPoleEnv()
ns = length(state(env))
na = length(action_space(env))

# Q-network
q_net = Chain(Dense(ns, 128, relu), Dense(128, 128, relu), Dense(128, na))

agent = Agent(
    policy = QBasedPolicy(
        learner = DQNLearner(
            approximator = NeuralNetworkApproximator(
                model = q_net,
                optimizer = Adam(1e-3)
            ),
            target_update_freq = 100,
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            epsilon_init = 1.0,
            epsilon_stable = 0.01,
            decay_steps = 1000
        )
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 10_000,
        state = Vector{Float32} => (ns,),
    )
)

# Train
stop_condition = StopAfterEpisode(500)
hook = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
run(agent, env, stop_condition, hook)
```

### Double DQN

```julia
learner = DQNLearner(
    approximator = NeuralNetworkApproximator(
        model = q_net,
        optimizer = Adam(1e-3)
    ),
    target_update_freq = 100,
    method = :double,             # Use Double DQN
    batch_size = 32
)
```

## PPO

Proximal Policy Optimization with GAE:

```julia
actor = Chain(Dense(ns, 64, tanh), Dense(64, 64, tanh), Dense(64, na))
critic = Chain(Dense(ns, 64, tanh), Dense(64, 64, tanh), Dense(64, 1))

agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = NeuralNetworkApproximator(model=actor, optimizer=Adam(3e-4)),
            critic = NeuralNetworkApproximator(model=critic, optimizer=Adam(1e-3))
        ),
        gamma = 0.99,
        lambda = 0.95,            # GAE lambda
        clip_range = 0.2,         # PPO clip range
        n_epochs = 4,
        n_microbatches = 4,
        update_freq = 2048
    ),
    trajectory = CircularArraySARTTrajectory(capacity = 2048)
)

run(agent, env, StopAfterEpisode(1000))
```

## A2C

Advantage Actor-Critic with entropy bonus:

```julia
agent = Agent(
    policy = A2CPolicy(
        approximator = ActorCritic(
            actor = NeuralNetworkApproximator(model=actor, optimizer=Adam(7e-4)),
            critic = NeuralNetworkApproximator(model=critic, optimizer=Adam(7e-4))
        ),
        gamma = 0.99,
        actor_loss_weight = 1.0,
        critic_loss_weight = 0.5,
        entropy_loss_weight = 0.01,  # Entropy bonus for exploration
        n_actors = 8                  # Parallel environments
    ),
    trajectory = CircularArraySARTTrajectory(capacity = 128)
)
```

## DDPG for Continuous Actions

Deep Deterministic Policy Gradient:

```julia
env = PendulumEnv()
ns = length(state(env))
na = 1  # Continuous action dimension

# Actor (deterministic policy)
behavior_actor = Chain(Dense(ns, 64, relu), Dense(64, 64, relu), Dense(64, na, tanh))
target_actor = deepcopy(behavior_actor)

# Critic Q(s, a)
behavior_critic = Chain(Dense(ns + na, 64, relu), Dense(64, 64, relu), Dense(64, 1))
target_critic = deepcopy(behavior_critic)

agent = Agent(
    policy = DDPGPolicy(
        behavior_actor = NeuralNetworkApproximator(model=behavior_actor, optimizer=Adam(1e-4)),
        behavior_critic = NeuralNetworkApproximator(model=behavior_critic, optimizer=Adam(1e-3)),
        target_actor = NeuralNetworkApproximator(model=target_actor),
        target_critic = NeuralNetworkApproximator(model=target_critic),
        gamma = 0.99,
        rho = 0.005,                 # Soft update coefficient (tau)
        batch_size = 64,
        act_noise = 0.1
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 100_000,
        state = Vector{Float32} => (ns,),
        action = Vector{Float32} => (na,)
    )
)
```

## CommonRLInterface.jl Wrapper

Use CommonRLInterface for cross-library compatibility:

```julia
using CommonRLInterface

struct MyEnv <: AbstractEnv
    # ...
end

CommonRLInterface.reset!(env::MyEnv) = reset_impl!(env)
CommonRLInterface.actions(env::MyEnv) = 1:4
CommonRLInterface.observe(env::MyEnv) = get_state(env)
CommonRLInterface.act!(env::MyEnv, a) = step_impl!(env, a)
CommonRLInterface.terminated(env::MyEnv) = env.done
```

## Training Hooks

Monitor and control training:

```julia
hook = ComposedHook(
    TotalRewardPerEpisode(),
    StepsPerEpisode(),
    DoEveryNEpisode(; n=100) do t, agent, env
        @info "Episode $(t)" reward=hook[1].rewards[end]
    end
)

# Custom hook for saving best model
mutable struct SaveBestModel <: AbstractHook
    best_reward::Float64
    path::String
end

function (h::SaveBestModel)(::PostEpisodeStage, agent, env)
    r = sum(agent.trajectory[:reward])
    if r > h.best_reward
        h.best_reward = r
        save_agent(h.path, agent)
    end
end
```

## Reward Shaping

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Sparse | Reward only at goal | Simple tasks, avoids bias |
| Dense | Continuous feedback | Speeds learning on hard tasks |
| Potential-based | Shaped = R + gamma * phi(s') - phi(s) | Preserves optimal policy |
| Curiosity-driven | Intrinsic reward for novel states | Sparse-reward exploration |

## Checklist

- [ ] Implement full `RLBase` interface for custom environments
- [ ] Start with DQN for discrete actions, DDPG for continuous
- [ ] Use PPO as the default policy gradient method
- [ ] Set up experience replay with sufficient capacity
- [ ] Add epsilon decay schedule for exploration
- [ ] Monitor with `TotalRewardPerEpisode` and custom hooks
- [ ] Use `CommonRLInterface.jl` for framework-agnostic environments
- [ ] Save best model checkpoints during training
