
"""
PettingZoo adapter for EnvKit multi-agent environments.

Provides the standard PettingZoo ParallelEnv API for multi-agent RL.

Usage:
    >>> from envkit.adapters.pettingzoo_adapter import make_pettingzoo_env
    >>>
    >>> env = make_pettingzoo_env(
    ...     ir_path="gridball.yaml",
    ...     backend="torch",
    ...     num_envs=64,
    ... )
    >>> observations, infos = env.reset()
    >>> while env.agents:
    ...     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    ...     observations, rewards, terminations, truncations, infos = env.step(actions)
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np
import yaml # Import yaml

try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "PettingZoo not installed. Install with: pip install pettingzoo gymnasium"
    )

from envkit.ir.schema import IR
from envkit.core.compiler import compile_ir
from envkit.core.runtime import EnvRunner
from envkit.core.extractors import extract_view


# ----------------------------------------------------------------------
# PettingZoo adapter
# ----------------------------------------------------------------------

class PettingZooEnvKitAdapter(ParallelEnv):
    """
    PettingZoo ParallelEnv adapter for EnvKit environments.

    This adapter:
    - Exposes all roles and agents as separate PettingZoo agents
    - Builds observations from role sensors (per agent)
    - Maps actions to role actuators
    - Handles vectorized environments (num_envs parallel instances)
    """

    metadata = {"render_modes": [], "name": "envkit_v0"}

    def __init__(
        self,
        layout: Any,  # CompiledLayout
        seed: Optional[int] = None,
    ):
        """
        Initialize PettingZoo adapter.

        Args:
            layout: Compiled EnvKit layout
            seed: Random seed
        """
        super().__init__()

        self.layout = layout
        self.seed_val = seed or 0
        self.num_envs = layout.B

        # Build agent list: "role_id_agent_idx"
        self.possible_agents = []
        self.role_agent_mapping = {}  # agent_name -> (role_id, agent_idx)

        for role_id, role in layout.roles.items():
            for agent_idx in range(role.max_agents):
                agent_name = f"{role_id}_{agent_idx}"
                self.possible_agents.append(agent_name)
                self.role_agent_mapping[agent_name] = (role_id, agent_idx)

        self.agents = self.possible_agents.copy()

        # Build observation and action spaces per agent
        self._observation_spaces = {}
        self._action_spaces = {}

        for agent_name in self.possible_agents:
            role_id, agent_idx = self.role_agent_mapping[agent_name]
            role = layout.roles[role_id]

            self._observation_spaces[agent_name] = self._build_observation_space(role)
            self._action_spaces[agent_name] = self._build_action_space(role)

        # Initialize runner
        self.runner = EnvRunner(layout, seed=self.seed_val)

        # Track episode stats per environment
        self.episode_returns = {
            agent: np.zeros(self.num_envs) for agent in self.possible_agents
        }
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Get observation spaces for all agents."""
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Get action spaces for all agents."""
        return self._action_spaces

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for specific agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for specific agent."""
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset environment.

        Returns:
            observations: Dict[agent_name, (num_envs, obs_dim)]
            infos: Dict[agent_name, info_dict]
        """
        if seed is not None:
            self.seed_val = seed

        # Reset EnvKit environment
        core_state, rng_state = self.runner.reset(seed=self.seed_val, options=options)

        # Reset active agents
        self.agents = self.possible_agents.copy()

        # Build observations for all agents
        observations = {}
        infos = {}

        for agent_name in self.agents:
            role_id, agent_idx = self.role_agent_mapping[agent_name]
            obs = self._build_agent_observation(core_state, role_id, agent_idx)
            observations[agent_name] = obs
            infos[agent_name] = {"num_envs": self.num_envs}

        # Reset episode stats
        for agent in self.possible_agents:
            self.episode_returns[agent] = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, np.ndarray],  # rewards
        Dict[str, np.ndarray],  # terminations
        Dict[str, np.ndarray],  # truncations
        Dict[str, Dict[str, Any]],  # infos
    ]:
        """
        Execute one step for all agents.

        Args:
            actions: Dict[agent_name, (num_envs, action_dim)]

        Returns:
            observations: Dict[agent_name, (num_envs, obs_dim)]
            rewards: Dict[agent_name, (num_envs,)]
            terminations: Dict[agent_name, (num_envs,)]
            truncations: Dict[agent_name, (num_envs,)]
            infos: Dict[agent_name, info_dict]
        """
        # Convert actions to EnvKit format
        be = self.layout.backend
        envkit_actions = self._actions_to_backend(actions)

        # Execute step
        result = self.runner.step(actions=envkit_actions)

        # Extract outputs for each agent
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Get global termination flags
        global_term = be.to_numpy(result.terminated)  # (B,)
        global_trunc = be.to_numpy(result.truncated)  # (B,)

        for agent_name in self.agents:
            role_id, agent_idx = self.role_agent_mapping[agent_name]

            # Observation
            obs = self._build_agent_observation(result.core_state, role_id, agent_idx)
            observations[agent_name] = obs

            # Reward
            role_rewards = result.role_rewards[role_id]  # (B, N_role)
            agent_reward = be.to_numpy(role_rewards[:, agent_idx])  # (B,)
            rewards[agent_name] = agent_reward

            # Termination/truncation (same for all agents)
            terminations[agent_name] = global_term.copy()
            truncations[agent_name] = global_trunc.copy()

            # Info
            infos[agent_name] = {}

            # Update episode stats
            self.episode_returns[agent_name] += agent_reward

        self.episode_lengths += 1

        # Add episode stats for completed episodes
        done_mask = global_term | global_trunc
        if np.any(done_mask):
            for agent_name in self.agents:
                infos[agent_name]["episode"] = {
                    "r": self.episode_returns[agent_name][done_mask],
                    "l": self.episode_lengths[done_mask],
                }
                # Reset stats for done episodes
                self.episode_returns[agent_name][done_mask] = 0

            self.episode_lengths[done_mask] = 0

        return observations, rewards, terminations, truncations, infos

    def close(self):
        """Close environment."""
        pass

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------

    def _build_observation_space(self, role: Any) -> spaces.Space:
        """Build observation space for a role."""
        total_obs_dim = 0

        for sensor in role.sensors:
            sensor_id = str(sensor.id)
            sensor_shape = self.layout.shape_service.sensor_shape(sensor_id)
            sensor_dim = int(np.prod(sensor_shape))
            total_obs_dim += sensor_dim

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, total_obs_dim),
            dtype=np.float32,
        )

    def _build_action_space(self, role: Any) -> spaces.Space:
        """Build action space for a role."""
        if not role.actuators or not role.actuators.writes:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_envs, 1),
                dtype=np.float32,
            )

        total_action_dim = 0
        low_bounds = []
        high_bounds = []

        for field_id in role.actuators.writes:
            field_spec = self.layout.field_specs[str(field_id)]

            # Field shape is typically (N_role, *feature_shape);
            # we want per-agent feature dims only.
            if len(field_spec.shape) > 1:
                feature_shape = field_spec.shape[1:]
            else:
                feature_shape = []

            field_dim = int(np.prod(feature_shape)) if feature_shape else 1

            if (getattr(role.actuators, "constraints", None)
                    and str(field_id) in role.actuators.constraints.bounds):
                bounds = role.actuators.constraints.bounds[str(field_id)]
                low = bounds.min
                high = bounds.max
            else:
                low = -1.0
                high = 1.0

            low_bounds.extend([low] * field_dim)
            high_bounds.extend([high] * field_dim)
            total_action_dim += field_dim

        return spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            shape=(self.num_envs, total_action_dim),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation and action conversion
    # ------------------------------------------------------------------

    def _build_agent_observation(
        self,
        core_state: Any,
        role_id: str,
        agent_idx: int,
    ) -> np.ndarray:
        """Build observation for a specific agent."""
        be = self.layout.backend
        role = self.layout.roles[role_id]
        obs_parts = []

        for sensor in role.sensors:
            sensor_id = str(sensor.id)

            if hasattr(sensor, 'from_fields'):  # View sensor
                fields = [str(f) for f in sensor.from_fields]
                sensor_obs = extract_view(self.layout, core_state, fields)
            else:
                sensor_obs = be.zeros(
                    (self.num_envs, int(np.prod(self.layout.shape_service.sensor_shape(sensor_id)))),
                    dtype=be.float_dtype
                )

            sensor_obs_flat = be.reshape(sensor_obs, (self.num_envs, -1))
            obs_parts.append(sensor_obs_flat)

        if obs_parts:
            obs = be.concat(obs_parts, axis=1)
        else:
            obs = be.zeros((self.num_envs, 0), dtype=be.float_dtype)

        return be.to_numpy(obs)

    def _actions_to_backend(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Convert PettingZoo actions to EnvKit format."""
        be = self.layout.backend

        # Group actions by role
        role_actions = {}

        if not role.actuators or not role.actuators.writes:
                action_dim = 1
            else:
                action_dim = 0
                for f in role.actuators.writes:
                    field_spec = self.layout.field_specs[str(f)]
                    if len(field_spec.shape) > 1:
                        feature_shape = field_spec.shape[1:]
                    else:
                        feature_shape = []
                    field_dim = int(np.prod(feature_shape)) if feature_shape else 1
                    action_dim += field_dim

            # Initialize role action array (B, N_role, action_dim)
            role_action = be.zeros(
                (self.num_envs, N_role, action_dim),
                dtype=be.float_dtype
            )

            # Fill in actions from each agent
            for agent_idx in range(N_role):
                agent_name = f"{role_id}_{agent_idx}"
                if agent_name in actions:
                    agent_action = be.asarray(actions[agent_name], dtype=be.float_dtype)
                    role_action[:, agent_idx, :] = agent_action

            role_actions[role_id] = role_action

        return role_actions


# ----------------------------------------------------------------------
# Convenience factory function
# ----------------------------------------------------------------------

def make_pettingzoo_env(
    ir_path: str,
    backend: str = "numpy",
    num_envs: int = 1,
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> PettingZooEnvKitAdapter:
    """
    Create a PettingZoo environment from an EnvKit IR specification.

    Args:
        ir_path: Path to IR YAML file
        backend: Backend to use ('numpy', 'torch', 'jax')
        num_envs: Number of parallel environments
        task_id: Which task to use (default: first task)
        seed: Random seed

    Returns:
        PettingZoo ParallelEnv

    Example:
        >>> env = make_pettingzoo_env(
        ...     "gridball.yaml",
        ...     backend="torch",
        ...     num_envs=64,
        ... )
        >>> observations, infos = env.reset(seed=42)
        >>>
        >>> for _ in range(1000):
        ...     actions = {
        ...         agent: env.action_space(agent).sample()
        ...         for agent in env.agents
        ...     }
        ...     obs, rewards, terms, truncs, infos = env.step(actions)
    """
    # Load IR from YAML
    with open(ir_path, 'r') as f:
        ir_data = yaml.safe_load(f)
    ir = IR.model_validate(ir_data)

    # Select backend
    if backend == "numpy":
        from envkit.backends.numpy_backend import NumpyBackend
        be = NumpyBackend()
    elif backend == "torch":
        from envkit.backends.torch_backend import TorchBackend
        be = TorchBackend()
    elif backend == "jax":
        from envkit.backends.jax_backend import JaxBackend
        be = JaxBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Compile
    layout = compile_ir(ir, batch_size=num_envs, backend=be, task_id=task_id)

    # Create adapter
    return PettingZooEnvKitAdapter(layout, seed)
