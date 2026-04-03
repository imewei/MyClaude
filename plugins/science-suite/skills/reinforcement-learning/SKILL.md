---
name: reinforcement-learning
description: "Build reinforcement learning agents with Gymnasium, Stable-Baselines3, and RLlib including DQN, PPO, SAC, multi-agent RL, reward shaping, and environment design. Use when training RL agents, designing reward functions, creating custom environments, or implementing policy optimization."
---

# Reinforcement Learning

Train agents, design environments, and implement policy optimization.

## Expert Agent

For deep learning architectures and training pipelines in RL, delegate to the expert agent:

- **`neural-network-master`**: Deep learning specialist for architecture design, training optimization, and model deployment.
  - *Location*: `plugins/science-suite/agents/neural-network-master.md`
  - *Capabilities*: Network architecture, loss functions, distributed training, debugging training failures.

## Gymnasium Environment API

```python
import gymnasium as gym
import numpy as np

# Standard environment usage
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)

total_reward = 0
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # Replace with policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

env.close()
```

## Custom Environment

```python
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """Custom grid world environment following Gymnasium API."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size: int = 5, render_mode: str = None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
        })
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2)
        self._target_location = self.np_random.integers(0, self.size, size=2)
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)
        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else -0.01
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location)}
```

## Stable-Baselines3

```python
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

vec_env = make_vec_env("CartPole-v1", n_envs=4, seed=42)

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, clip_range=0.2, verbose=1, seed=42,
)

# Callbacks for evaluation and checkpointing
eval_callback = EvalCallback(
    eval_env=gym.make("CartPole-v1"),
    n_eval_episodes=10,
    eval_freq=5000,
    best_model_save_path="./best_model/",
)
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path="./checkpoints/"
)

model.learn(total_timesteps=100_000, callback=[eval_callback, checkpoint_callback])

# Evaluate
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

## Reward Shaping

```python
class ShapedRewardWrapper(gym.Wrapper):
    """Add potential-based reward shaping (preserves optimal policy)."""

    def __init__(self, env, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma
        self._prev_potential = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = self._potential(obs)
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential
        return obs, reward + shaping, terminated, truncated, info

    def _potential(self, obs) -> float:
        return -np.linalg.norm(obs["agent"] - obs["target"])
```

## Hyperparameter Tuning

```python
# Key hyperparameters by algorithm
TUNING_GUIDE = {
    "PPO": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [512, 1024, 2048],
        "batch_size": [32, 64, 128],
        "clip_range": [0.1, 0.2, 0.3],
        "n_epochs": [3, 5, 10],
    },
    "SAC": {
        "learning_rate": [1e-4, 3e-4],
        "batch_size": [128, 256],
        "buffer_size": [100_000, 1_000_000],
        "tau": [0.005, 0.01],
        "learning_starts": [1000, 10_000],
    },
}
```

## RL Development Checklist

- [ ] Define clear reward signal aligned with desired behavior
- [ ] Normalize observations (zero mean, unit variance)
- [ ] Start with PPO for discrete, SAC for continuous action spaces
- [ ] Use vectorized environments (n_envs >= 4) for training
- [ ] Monitor episode return, episode length, and value loss
- [ ] Test with random policy baseline before training
- [ ] Check for reward hacking and unintended behaviors
- [ ] Set explicit random seeds for reproducibility
- [ ] Save and version checkpoints during training
- [ ] Evaluate on held-out episodes (not training episodes)
