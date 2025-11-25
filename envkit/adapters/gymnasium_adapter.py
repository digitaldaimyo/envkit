
"""
Gymnasium adapter for EnvKit environments.

Wraps EnvKit environments in the standard Gymnasium API for single-agent RL,
controlling one agent inside a chosen agent group.

Usage:
    >>> from envkit.adapters.gymnasium_adapter import make_gymnasium_env
    >>>
    >>> env = make_gymnasium_env(
    ...     yaml_path="envkit/packs/gridball/gridball_ir.yaml",
    ...     group_id="team_red",  # Which group to control
    ...     agent_idx=0,          # Which agent within that group
    ... )
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "Gymnasium not installed. Install with: pip install gymnasium"
    )

from envkit.ir.schema import IR
    # IR is the top-level schema model
from envkit.core.compiler import compile_ir
from envkit.core.runtime import EnvRunner
from envkit.core.extractors import extract_view, extract_topk
from envkit.engines.expr_eval import evaluate_expression


# ----------------------------------------------------------------------
# Main adapter
# ----------------------------------------------------------------------

class GymnasiumEnvKitAdapter(gym.Env):
    """
    Gymnasium adapter for EnvKit environments with vectorization support.

    This adapter:
    - Handles vectorized (num_envs) environments
    - Selects a single agent within a given group to control
    - Builds observations from that group's sensors
    - Maps actions into that group's actuator fields
    - Returns batched (num_envs, ...) arrays

    Note: Spaces describe per-env shape. Actual observations/actions
    have shape (num_envs, *space.shape).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: Any,  # CompiledLayout
        group_id: str,
        agent_idx: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Gymnasium adapter.

        Args:
            layout: Compiled EnvKit layout
            group_id: Which agent group to control
            agent_idx: Which agent index in that group (default: 0)
            seed: Random seed
        """
        super().__init__()

        self.layout = layout
        self.group_id = group_id
        self.agent_idx = agent_idx
        self.seed_val = seed or 0

        # Validate group exists
        if group_id not in layout.groups:
            available = ", ".join(layout.groups.keys())
            raise ValueError(
                f"Group '{group_id}' not found. Available groups: {available}"
            )

        self.group = layout.groups[group_id]
        self.num_envs = layout.B

        # Safety: ensure agent_idx is in range
        group_count = int(getattr(self.group, "count", 1))
        if not (0 <= self.agent_idx < group_count):
            raise ValueError(
                f"agent_idx={self.agent_idx} out of range for group '{group_id}' "
                f"with count={group_count}"
            )

        # Build observation / action spaces (per-env shapes)
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Initialize runner
        self.runner = EnvRunner(layout, seed=self.seed_val)

        # Track episode stats
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.

        Returns:
            obs: (num_envs, *obs_space.shape)
            info: Dict with num_envs key, etc.
        """
        if seed is not None:
            self.seed_val = seed

        # Reset EnvKit environment
        core_state, _ = self.runner.reset(seed=self.seed_val, options=options)

        # Build observation
        obs = self._build_observation(core_state)

        # Reset episode stats
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)

        info = {"num_envs": self.num_envs}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action array (num_envs, *action_space.shape)

        Returns:
            obs: Observation (num_envs, *obs_space.shape)
            reward: Rewards (num_envs,)
            terminated: Termination flags (num_envs,)
            truncated: Truncation flags (num_envs,)
            info: Info dict
        """
        # Validate action shape
        expected_shape = (self.num_envs,) + self.action_space.shape
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected action shape {expected_shape}, got {action.shape}"
            )

        # Convert action to backend format
        action_dict = self._action_to_backend(action)

        # Execute step
        result = self.runner.step(actions=action_dict)

        # Extract observation
        obs = self._build_observation(result.core_state)

        be = self.layout.backend

        # Extract reward for this group and agent
        group_rewards = result.group_rewards[self.group_id]  # (B, N_group)
        reward = be.to_numpy(group_rewards[:, self.agent_idx])  # (B,)

        # Extract termination flags
        terminated = be.to_numpy(result.terminated)  # (B,)
        truncated = be.to_numpy(result.truncated)  # (B,)

        # Update episode stats
        self.episode_returns += reward
        self.episode_lengths += 1

        # Build info dict
        info: Dict[str, Any] = {}

        # Add episode stats for completed episodes
        done_mask = terminated | truncated
        if np.any(done_mask):
            info["episode"] = {
                "r": self.episode_returns[done_mask].copy(),
                "l": self.episode_lengths[done_mask].copy(),
            }
            # Reset stats for done episodes
            self.episode_returns[done_mask] = 0.0
            self.episode_lengths[done_mask] = 0

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close environment."""
        # Nothing to clean up yet
        pass

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Space:
        """
        Build Gymnasium observation space from group sensors.

        Returns space for a SINGLE environment.
        Actual observations will be (num_envs, *obs_space.shape).
        """
        total_obs_dim = 0

        # Sum up all sensor output dimensions
        for sensor in self.group.sensors:
            sensor_id = str(sensor.id)
            sensor_shape = self.layout.shape_service.sensor_shape(sensor_id)

            # Flatten sensor shape
            sensor_dim = int(np.prod(sensor_shape))
            total_obs_dim += sensor_dim

        # Per-env observation space (without batch dimension!)
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),  # per-env shape only
            dtype=np.float32,
        )

    def _build_action_space(self) -> spaces.Space:
        """
        Build Gymnasium action space from group actuators.

        Returns space for a SINGLE environment.
        Actual actions will be (num_envs, *action_space.shape).
        """
        if not self.group.actuators or not self.group.actuators.writes:
            raise ValueError(f"Group '{self.group_id}' has no actuators defined")

        total_action_dim = 0
        low_bounds: List[float] = []
        high_bounds: List[float] = []

        for field_id in self.group.actuators.writes:
            fid_str = str(field_id)
            field_spec = self.layout.field_specs[fid_str]

            # Field shape is typically (N_group, *feature_shape)
            # We want the per-agent feature shape.
            if len(field_spec.shape) > 1:
                feature_shape = field_spec.shape[1:]
            else:
                # Scalar per agent
                feature_shape = []

            field_dim = int(np.prod(feature_shape)) if feature_shape else 1

            # Bounds, if defined
            constraints = self.group.actuators.constraints
            if constraints and fid_str in constraints.bounds:
                bounds = constraints.bounds[fid_str]
                low = float(bounds.min)
                high = float(bounds.max)
            else:
                # Default bounds
                low = -1.0
                high = 1.0

            low_bounds.extend([low] * field_dim)
            high_bounds.extend([high] * field_dim)
            total_action_dim += field_dim

        low_arr = np.array(low_bounds, dtype=np.float32)
        high_arr = np.array(high_bounds, dtype=np.float32)

        return spaces.Box(
            low=low_arr,
            high=high_arr,
            shape=(total_action_dim,),  # per-env shape only
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation and action conversion
    # ------------------------------------------------------------------

    def _build_observation(self, core_state: Any) -> np.ndarray:
        """
        Build observation from current state using group sensors.

        Returns:
            (num_envs, obs_dim) float32 array
        """
        be = self.layout.backend
        obs_parts = []

        for sensor in self.group.sensors:
            sensor_id = str(sensor.id)

            # View sensor (flatten fields)
            if hasattr(sensor, "from_fields"):
                fields = [str(f) for f in sensor.from_fields]
                sensor_obs = extract_view(
                    self.layout,
                    core_state,
                    fields,
                )
            # TopK sensor
            elif hasattr(sensor, "pos_field"):
                sensor_obs = extract_topk(
                    self.layout,
                    core_state,
                    sensor.pos_field,
                    sensor.extra_fields,
                    sensor.k,
                    sensor.center_expr,
                    evaluate_expression,
                    sensor.normalize,
                )
            else:
                # Fallback for unsupported sensor types: zeros of correct shape
                sensor_shape = self.layout.shape_service.sensor_shape(sensor_id)
                sensor_obs = be.zeros(
                    (self.num_envs, int(np.prod(sensor_shape))),
                    dtype=be.float_dtype,
                )

            # Flatten sensor output to (num_envs, sensor_dim)
            sensor_obs_flat = be.reshape(sensor_obs, (self.num_envs, -1))
            obs_parts.append(sensor_obs_flat)

        # Concatenate all sensors
        if obs_parts:
            obs_be = be.concat(obs_parts, axis=1)
        else:
            obs_be = be.zeros((self.num_envs, 0), dtype=be.float_dtype)

        # Convert to numpy
        return be.to_numpy(obs_be)

    def _action_to_backend(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Convert Gymnasium action to EnvKit action dict.

        Args:
            action: (num_envs, action_dim) numpy array

        Returns:
            Dict[group_id, backend_array] with shape (B, N_group, action_dim)
            where only the selected agent_idx is non-zero.
        """
        be = self.layout.backend

        # (B, action_dim)
        action_tensor = be.asarray(action, dtype=be.float_dtype)

        # Group size = count in the IR
        N_group = int(getattr(self.group, "count", 1))
        action_dim = action.shape[1]

        # One-hot mask over agents: shape (N_group,)
        # (using backend.xp for numpy/torch/jax interop)
        agent_indices = be.xp.arange(N_group)
        mask_1d = (agent_indices == self.agent_idx)
        agent_mask = be.asarray(mask_1d, dtype=be.float_dtype)  # (N_group,)

        # Broadcast to (1, N_group, 1)
        agent_mask = be.expand_dims(agent_mask, axis=0)  # (1, N_group)
        agent_mask = be.expand_dims(agent_mask, axis=2)  # (1, N_group, 1)

        # Broadcast action_tensor (B, 1, D) * mask (1, N_group, 1) -> (B, N_group, D)
        group_action = action_tensor[:, None, :] * agent_mask

        return {self.group_id: group_action}


# ----------------------------------------------------------------------
# Convenience factory function
# ----------------------------------------------------------------------

def make_gymnasium_env(
    yaml_path: str,
    group_id: str,
    agent_idx: int = 0,
    num_envs: int = 1,
    backend: str = "numpy",
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> GymnasiumEnvKitAdapter:
    """
    Create a Gymnasium environment from an EnvKit IR specification.

    Args:
        yaml_path: Path to IR YAML file
        group_id: Which agent group to control
        agent_idx: Which agent index within that group (default: 0)
        num_envs: Number of parallel environments (batch size)
        backend: Backend to use ('numpy', 'torch', 'jax')
        task_id: Which task to use (default: IR default task)
        seed: Random seed

    Returns:
        Gymnasium environment that handles vectorized envs
    """
    # Load IR from YAML
    with open(yaml_path, "r") as f:
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
        raise ValueError(f"Unknown backend: {backend!r}")

    # Compile IR to a layout
    layout = compile_ir(
        ir,
        batch_size=num_envs,
        backend=be,
        task_id=task_id,
    )

    # Wrap in Gymnasium adapter
    return GymnasiumEnvKitAdapter(layout, group_id, agent_idx, seed)
