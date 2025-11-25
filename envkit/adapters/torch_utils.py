"""
PyTorch utilities for RL training with EnvKit environments.

Provides helpers for:
- Converting observations/actions between numpy and torch
- Batched rollout collection
- Vectorized advantage computation
- Experience buffer management

Usage:
    >>> from envkit.adapters.torch_utils import TorchRolloutCollector
    >>>
    >>> collector = TorchRolloutCollector(env, policy, device="cuda")
    >>> batch = collector.collect_rollouts(num_steps=2048)
    >>> # batch contains {obs, actions, rewards, dones, values, log_probs}
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal, Categorical
except ImportError:
    raise ImportError("PyTorch not installed. Install with: pip install torch")


# ----------------------------------------------------------------------
# Rollout batch
# ----------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """
    Batched rollout data for RL training.

    All tensors have shape (num_steps, num_envs, *).
    """
    observations: torch.Tensor  # (T, B, obs_dim)
    actions: torch.Tensor  # (T, B, action_dim)
    rewards: torch.Tensor  # (T, B)
    dones: torch.Tensor  # (T, B) - terminated or truncated
    values: torch.Tensor  # (T, B) - value predictions
    log_probs: torch.Tensor  # (T, B) - action log probabilities

    # Computed fields
    advantages: Optional[torch.Tensor] = None  # (T, B)
    returns: Optional[torch.Tensor] = None  # (T, B)

    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ) -> None:
        """
        Compute advantages using GAE.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize: If True, normalize advantages to mean=0, std=1
        """
        T, B = self.rewards.shape
        device = self.rewards.device

        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = torch.zeros(B, device=device)

        # Compute GAE backwards through time
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=device)
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.advantages = advantages
        self.returns = advantages + self.values

        # Normalize advantages
        if normalize:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_flat_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get flattened batch for minibatch training.

        Returns:
            Dict with all tensors flattened to (T*B, *)
        """
        T, B = self.observations.shape[:2]

        return {
            "observations": self.observations.reshape(T * B, -1),
            "actions": self.actions.reshape(T * B, -1),
            "rewards": self.rewards.reshape(T * B),
            "dones": self.dones.reshape(T * B),
            "values": self.values.reshape(T * B),
            "log_probs": self.log_probs.reshape(T * B),
            "advantages": self.advantages.reshape(T * B) if self.advantages is not None else None,
            "returns": self.returns.reshape(T * B) if self.returns is not None else None,
        }

    def to(self, device: torch.device) -> "RolloutBatch":
        """Move all tensors to device."""
        return RolloutBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            values=self.values.to(device),
            log_probs=self.log_probs.to(device),
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
        )


# ----------------------------------------------------------------------
# Rollout collector
# ----------------------------------------------------------------------

class TorchRolloutCollector:
    """
    Collects rollouts from vectorized environments for RL training.

    Works with any Gymnasium-compatible environment (including EnvKit adapters).
    """

    def __init__(
        self,
        env: Any,  # Gymnasium vectorized env
        policy: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize rollout collector.

        Args:
            env: Vectorized Gymnasium environment
            policy: Policy network (should have .get_action_and_value method)
            device: Device to run policy on
        """
        self.env = env
        self.policy = policy
        self.device = torch.device(device)

        self.num_envs = getattr(env, 'num_envs', 1)

        # Move policy to device
        self.policy.to(self.device)

    @torch.no_grad()
    def collect_rollouts(
        self,
        num_steps: int,
        deterministic: bool = False,
    ) -> RolloutBatch:
        """
        Collect rollouts from environment.

        Args:
            num_steps: Number of steps to collect
            deterministic: If True, use deterministic policy

        Returns:
            RolloutBatch with collected data
        """
        # Storage
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        # Get current observation
        if not hasattr(self, '_current_obs'):
            self._current_obs, _ = self.env.reset()

        obs = self._current_obs

        for step in range(num_steps):
            # Convert obs to torch
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            # Get action from policy
            action, log_prob, value = self._get_action_and_value(obs_tensor, deterministic)

            # Convert action to numpy
            action_np = action.cpu().numpy()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated | truncated

            # Store
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
            dones.append(torch.as_tensor(done, dtype=torch.float32, device=self.device))
            values.append(value)
            log_probs.append(log_prob)

            # Update obs
            obs = next_obs

        # Store current obs for next collection
        self._current_obs = obs

        # Stack tensors
        batch = RolloutBatch(
            observations=torch.stack(observations),  # (T, B, obs_dim)
            actions=torch.stack(actions),  # (T, B, action_dim)
            rewards=torch.stack(rewards),  # (T, B)
            dones=torch.stack(dones),  # (T, B)
            values=torch.stack(values).squeeze(-1),  # (T, B)
            log_probs=torch.stack(log_probs),  # (T, B)
        )

        return batch

    def _get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, and value from policy.

        Args:
            obs: Observations (B, obs_dim)
            deterministic: If True, use mean action

        Returns:
            (action, log_prob, value)
        """
        if hasattr(self.policy, 'get_action_and_value'):
            # Policy has our expected interface
            return self.policy.get_action_and_value(obs, deterministic)
        else:
            # Fallback: assume policy returns distribution
            dist = self.policy(obs)

            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            # Dummy value
            value = torch.zeros(obs.shape[0], 1, device=self.device)

            return action, log_prob, value


# ----------------------------------------------------------------------
# Simple actor-critic policy
# ----------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Simple actor-critic network for continuous control.

    Can be used with TorchRolloutCollector for PPO/A2C training.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [64, 64],
        activation: nn.Module = nn.Tanh,
    ):
        """
        Initialize actor-critic.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super().__init__()

        # Shared feature extractor
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation(),
            ])
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor_mean = nn.Linear(prev_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value)
        self.critic = nn.Linear(prev_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Get action distribution.

        Args:
            obs: Observations (B, obs_dim)

        Returns:
            Normal distribution over actions
        """
        features = self.features(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value prediction.

        Args:
            obs: Observations (B, obs_dim)

        Returns:
            Values (B, 1)
        """
        features = self.features(obs)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, and value.

        Args:
            obs: Observations (B, obs_dim)
            deterministic: If True, use mean action

        Returns:
            (action, log_prob, value)
        """
        dist = self.forward(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.get_value(obs)

        return action, log_prob, value


# ----------------------------------------------------------------------
# Experience replay buffer (for off-policy algorithms)
# ----------------------------------------------------------------------

class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL algorithms.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cpu",
    ):
        """
        Initialize replay buffer.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            max_size: Maximum buffer size
            device: Device to store tensors on
        """
        self.max_size = max_size
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        # Preallocate storage
        self.observations = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(max_size, dtype=torch.float32, device=self.device)
        self.next_observations = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(max_size, dtype=torch.float32, device=self.device)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------

def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Compute PPO clipped policy loss.

    Args:
        log_probs: New log probabilities
        old_log_probs: Old log probabilities
        advantages: Advantage estimates
        clip_range: PPO clipping range

    Returns:
        Policy loss (scalar)
    """
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    return policy_loss


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute value loss.

    Args:
        values: Predicted values
        returns: Target returns
        clip_range: Optional value clipping range

    Returns:
        Value loss (scalar)
    """
    if clip_range is not None:
        values_clipped = torch.clamp(values, returns - clip_range, returns + clip_range)
        loss_clipped = (values_clipped - returns).pow(2)
        loss_unclipped = (values - returns).pow(2)
        value_loss = 0.5 * torch.max(loss_clipped, loss_unclipped).mean()
    else:
        value_loss = 0.5 * (values - returns).pow(2).mean()

    return value_loss
