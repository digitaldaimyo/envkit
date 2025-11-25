# envkit/core/runtime.py
from __future__ import annotations

from typing import Tuple, Dict, List, Any, Optional
import time

from envkit.core.compiled_layout import (
    CompiledLayout,
    CoreState,
    RNGState,
    Events,
    Array,
    create_initial_core_state,
    create_initial_rng_state,
    reset_scratch_fields,
)


# ----------------------------------------------------------------------
# Type aliases for return values
# ----------------------------------------------------------------------

GroupRewards = Dict[str, Array]  # group_id -> (B, N_group)
TermVec = Array  # (B,) bool
TruncVec = Array  # (B,) bool


# ----------------------------------------------------------------------
# Step result
# ----------------------------------------------------------------------

class StepResult:
    """Result of a single environment step."""

    def __init__(
        self,
        core_state: CoreState,
        rng_state: RNGState,
        group_rewards: GroupRewards,
        terminated: TermVec,
        truncated: TruncVec,
        events: Events,
        step_time: float = 0.0,
        phase_times: Optional[Dict[str, float]] = None,
    ):
        self.core_state = core_state
        self.rng_state = rng_state
        self.group_rewards = group_rewards
        self.terminated = terminated
        self.truncated = truncated
        self.events = events
        self.step_time = step_time
        self.phase_times = phase_times or {}

    def __repr__(self) -> str:
        return (
            f"StepResult(rewards={len(self.group_rewards)} groups, "
            f"events={len(self.events)} channels, "
            f"time={self.step_time:.4f}s)"
        )



def _jit_compile_systems(
    layout: CompiledLayout,
    enable_jit: bool = True,
) -> None:
    """
    JIT-compile systems that are marked as JIT-compatible.

    Modifies layout.systems in-place by replacing impl_fn with compiled version.

    Args:
        layout: Compiled layout with systems to JIT
        enable_jit: Whether to actually compile (False for debugging)
    """
    if not enable_jit:
        return

    be = layout.backend

    # Check if backend supports JIT
    if not hasattr(be, 'jit_compile'):
        return

    for sys in layout.systems:
        if not sys.jit_compatible:
            continue

        try:
            # Wrap system function to match JIT signature
            original_fn = sys.impl_fn

            # JIT compile the function
            compiled_fn = be.jit_compile(original_fn)

            # Replace impl_fn with compiled version
            sys.impl_fn = compiled_fn

        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to JIT-compile system '{sys.id}': {e}\n"
                f"Falling back to eager execution.",
                RuntimeWarning
            )

# ----------------------------------------------------------------------
# Core step function
# ----------------------------------------------------------------------

def core_step(
    layout: CompiledLayout,
    core_state: CoreState,
    rng_state: RNGState,
    t: int,
    actions: Optional[Dict[str, Array]] = None,
    collect_timings: bool = False,
) -> StepResult:
    """
    Execute one environment step (v2.1 Section 7 - Runtime).

    This:
    1. Resets scratch fields
    2. Applies actions to state
    3. Executes all phases in order
    4. Computes rewards
    5. Evaluates termination conditions
    6. Returns updated state and outcomes

    Note: JIT compilation must be done before calling this function.
    Use runtime.jit_compile_layout(layout) to enable JIT.
    """
    step_start = time.perf_counter() if collect_timings else 0.0
    phase_times: Optional[Dict[str, float]] = {} if collect_timings else None

    # Step 0: Validate inputs (if validation enabled)
    if layout.validation_mode != "none":
        _validate_step_inputs(layout, core_state, rng_state, actions)

    # Step 1: Reset scratch fields
    core_state = reset_scratch_fields(core_state, layout)

    # Step 2: Apply actions to state (write to actuator fields)
    if actions is not None:
        core_state = _apply_actions(layout, core_state, actions)

    # Step 3: Execute all phases
    active_phases = _get_active_phases(layout)
    events: Events = {}

    for phase_id in active_phases:
        phase_start = time.perf_counter() if collect_timings else 0.0

        core_state, rng_state, phase_events = _execute_phase(
            layout, core_state, rng_state, phase_id
        )

        # Merge phase events into global events:
        #   - Same event_id, same field_name → sum arrays
        #   - New field → insert
        for event_id, chan in phase_events.items():
            if event_id not in events:
                events[event_id] = dict(chan)
            else:
                existing = events[event_id]
                for field_name, arr in chan.items():
                    if field_name in existing:
                        existing[field_name] = existing[field_name] + arr
                    else:
                        existing[field_name] = arr

        if collect_timings and phase_times is not None:
            phase_times[phase_id] = time.perf_counter() - phase_start

    # Step 4: Compute rewards (reward engine)
    group_rewards = _compute_rewards(layout, core_state, events)

    # Step 5: Compute termination (termination engine)
    terminated, truncated = _compute_termination(layout, core_state, events, t)

    # Step 6: Build result
    step_time = time.perf_counter() - step_start if collect_timings else 0.0

    return StepResult(
        core_state=core_state,
        rng_state=rng_state,
        group_rewards=group_rewards,
        terminated=terminated,
        truncated=truncated,
        events=events,
        step_time=step_time,
        phase_times=phase_times,
    )


# ----------------------------------------------------------------------
# Core reset function
# ----------------------------------------------------------------------

def core_reset(
    layout: CompiledLayout,
    seed: Optional[int] = None,
    task_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[CoreState, RNGState]:
    """
    Reset environment to initial state.
    """
    # Step 1: Switch task if requested
    if task_id is not None and task_id != layout.active_task_id:
        layout.switch_task(task_id)

    # Step 2: Initialize RNG state
    if seed is None:
        seed = 0
    rng_state = create_initial_rng_state(layout, seed=seed)

    # Step 3: Initialize core state from defaults
    core_state = create_initial_core_state(layout)

    # Step 4: Apply curriculum knobs (if provided)
    if options and "curriculum_knobs" in options:
        core_state = _apply_curriculum_knobs(
            layout, core_state, options["curriculum_knobs"]
        )

    # Step 5: Execute reset phase
    core_state, rng_state, _ = _execute_phase(
        layout, core_state, rng_state, "reset"
    )

    # Step 6: Validate result (if validation enabled)
    if layout.validation_mode != "none":
        layout.validate_core_state(core_state)

    return core_state, rng_state


# ----------------------------------------------------------------------
# Phase execution
# ----------------------------------------------------------------------

def _execute_phase(
    layout: CompiledLayout,
    core_state: CoreState,
    rng_state: RNGState,
    phase_id: str,
) -> Tuple[CoreState, RNGState, Events]:
    """Execute all systems in a phase."""
    try:
        phase_idx = layout.phases.index(phase_id)
    except ValueError:
        raise RuntimeError(f"Phase '{phase_id}' not in layout.phases")

    start, end = layout.phase_system_ranges[phase_idx]

    if start == end:
        return core_state, rng_state, {}

    be = layout.backend
    events: Events = {}

    for sys in layout.systems[start:end]:
        try:
            core_state, rng_state, sys_events = sys.impl_fn(
                core_state,
                rng_state,
                be,
                layout,
                sys.params,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error executing system '{sys.id}' in phase '{phase_id}': {e}"
            ) from e

        # Validate events in debug mode
        if layout.validation_mode == "debug":
            _validate_events(sys_events, layout, sys.id)
            _validate_system_output(layout, sys, core_state)

        # Merge system events into phase events tensor-wise
        for event_id, chan in sys_events.items():
            if event_id not in events:
                events[event_id] = dict(chan)
            else:
                existing = events[event_id]
                for field_name, arr in chan.items():
                    if field_name in existing:
                        # Sum arrays for same field (merge semantic)
                        existing[field_name] = existing[field_name] + arr
                    else:
                        existing[field_name] = arr

    return core_state, rng_state, events

# ----------------------------------------------------------------------
# Action application
# ----------------------------------------------------------------------

def _apply_actions(
    layout: CompiledLayout,
    core_state: CoreState,
    actions: Dict[str, Array],
) -> CoreState:
    """
    Apply agent actions to actuator fields.

    For each group, writes action array to corresponding actuator fields.
    """
    be = layout.backend
    cs = list(core_state)

    for group_id, action_array in actions.items():
        if group_id not in layout.groups:
            raise ValueError(f"Unknown agent group: '{group_id}'")

        group = layout.groups[group_id]

        # Get actuator spec
        if group.actuators is None:
            raise ValueError(f"Group '{group_id}' has no actuators defined")

        actuator_fields = group.actuators.writes
        if not actuator_fields:
            continue

        constraints = group.actuators.constraints

        # NOTE: This is still the simple "single field" / shared actions version.
        # A richer action layout (splitting along dim) can be added later.
        for field_id in actuator_fields:
            field_id_str = str(field_id)
            field_idx = layout.field_index[field_id_str]

            # Apply constraints if defined
            if field_id_str in constraints.bounds:
                bounds = constraints.bounds[field_id_str]
                action_slice = be.clip(action_array, bounds.min, bounds.max)
            elif field_id_str in constraints.discrete:
                action_slice = action_array
            else:
                action_slice = action_array

            cs[field_idx] = action_slice

    return tuple(cs)


# ----------------------------------------------------------------------
# Reward computation
# ----------------------------------------------------------------------

def _compute_rewards(
    layout: CompiledLayout,
    core_state: CoreState,
    events: Events,
) -> GroupRewards:
    """
    Compute rewards for all agent groups.

    Delegates to RewardEngine; falls back to zeros if unavailable.
    """
    try:
        from envkit.engines.reward_engine import RewardEngine

        engine = RewardEngine(layout)
        return engine.compute(core_state, events, layout, layout.B)
    except ImportError:
        be = layout.backend
        group_rewards: GroupRewards = {}
        for group_id, group in layout.groups.items():
            group_rewards[group_id] = be.zeros(
                (layout.B, group.count),
                dtype=be.float_dtype,
            )
        return group_rewards


# ----------------------------------------------------------------------
# Termination computation
# ----------------------------------------------------------------------

def _compute_termination(
    layout: CompiledLayout,
    core_state: CoreState,
    events: Events,
    t: int,
) -> Tuple[TermVec, TruncVec]:
    """
    Compute termination and truncation masks.

    Delegates to TerminationEngine; falls back to False.
    """
    try:
        from envkit.engines.termination_engine import TerminationEngine

        engine = TerminationEngine(layout)
        return engine.compute(core_state, events, layout, layout.B, t)
    except ImportError:
        be = layout.backend
        terminated = be.zeros((layout.B,), dtype=be.bool_dtype)
        truncated = be.zeros((layout.B,), dtype=be.bool_dtype)
        return terminated, truncated


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _get_active_phases(layout: CompiledLayout) -> List[str]:
    """Get phase list for active task."""
    task = layout.get_active_task()
    if task.phases:
        return [str(p) for p in task.phases]
    else:
        return layout.phases


def _apply_curriculum_knobs(
    layout: CompiledLayout,
    core_state: CoreState,
    knobs: Dict[str, float],
) -> CoreState:
    """
    Apply curriculum knob values to bound state fields.
    """
    task = layout.get_active_task()
    if not hasattr(task, "curriculum") or not task.curriculum:
        return core_state

    be = layout.backend
    cs = list(core_state)

    for binding in task.curriculum.bindings:
        knob_id = str(binding.knob)
        field_id = str(binding.state_field)

        if knob_id not in knobs:
            continue

        value = knobs[knob_id]

        if field_id in layout.field_index:
            idx = layout.field_index[field_id]
            shape = (layout.B,) + layout.field_specs[field_id].shape

            value_arr = be.asarray(value, dtype=be.float_dtype)
            value_arr = be.broadcast_to(value_arr, shape)

            cs[idx] = value_arr

    return tuple(cs)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_step_inputs(
    layout: CompiledLayout,
    core_state: CoreState,
    rng_state: RNGState,
    actions: Optional[Dict[str, Array]],
) -> None:
    """Validate inputs to core_step."""
    # Validate core_state
    layout.validate_core_state(core_state)

    # Validate RNG state
    for stream_id in layout.rng_streams:
        if stream_id not in rng_state.counters:
            raise ValueError(f"RNG state missing stream: '{stream_id}'")

    # Validate actions
    if actions is not None:
        for group_id, action_array in actions.items():
            if group_id not in layout.groups:
                raise ValueError(
                    f"Action provided for unknown agent group: '{group_id}'"
                )

            group = layout.groups[group_id]

            # Basic consistency checks on batch & agent dims
            if action_array.shape[0] != layout.B:
                raise ValueError(
                    f"Action for group '{group_id}' has wrong batch size: "
                    f"{action_array.shape[0]} vs {layout.B}"
                )

            if action_array.shape[1] != group.count:
                raise ValueError(
                    f"Action for group '{group_id}' has wrong agent count: "
                    f"{action_array.shape[1]} vs {group.count}"
                )


def _validate_system_output(
    layout: CompiledLayout,
    sys: Any,
    core_state: CoreState,
) -> None:
    """Validate system output shapes and types."""
    for field_id in sys.writes:
        idx = layout.field_index[field_id]
        arr = core_state[idx]

        expected_shape = (layout.B,) + layout.field_specs[field_id].shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"System '{sys.id}' produced wrong shape for field '{field_id}': "
                f"{arr.shape} vs expected {expected_shape}"
            )


def _validate_events(
    events: Events,
    layout: CompiledLayout,
    system_id: str,
) -> None:
    """
    Validate emitted events match schema (debug mode only).

    Checks:
    - Event IDs are declared
    - Field names match EventChannelSpec
    - Arrays have correct batch dimension
    """
    be = layout.backend
    B = layout.B

    for event_id, fields in events.items():
        # Check event exists
        if event_id not in layout.event_specs:
            raise RuntimeError(
                f"System '{system_id}' emitted unknown event '{event_id}'\n"
                f"Available events: {', '.join(sorted(layout.event_specs.keys()))}"
            )

        event_spec = layout.event_specs[event_id]
        expected_fields = {str(f.id) for f in event_spec.fields}

        # Check fields match schema
        actual_fields = set(fields.keys())

        missing = expected_fields - actual_fields
        if missing:
            raise RuntimeError(
                f"System '{system_id}' event '{event_id}' missing fields: {missing}\n"
                f"Expected fields: {expected_fields}"
            )

        extra = actual_fields - expected_fields
        if extra:
            raise RuntimeError(
                f"System '{system_id}' event '{event_id}' has unexpected fields: {extra}\n"
                f"Expected fields: {expected_fields}"
            )

        # Check array shapes have correct batch dimension
        for field_name, arr in fields.items():
            arr = be.asarray(arr)
            if arr.shape[0] != B:
                raise RuntimeError(
                    f"System '{system_id}' event '{event_id}' field '{field_name}' "
                    f"has wrong batch size: {arr.shape[0]} (expected {B})"
                )

# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

class EnvRunner:
    """
    Convenience wrapper for running an environment.

    Manages state and provides a simple step/reset interface.
    """

    def __init__(
        self,
        layout: CompiledLayout,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        enable_jit: bool = False,
        jit_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize environment runner.

        Args:
            layout: Compiled layout
            seed: Random seed
            task_id: Initial task ID
            enable_jit: Whether to JIT-compile systems
            jit_kwargs: Backend-specific JIT options
        """
        self.layout = layout
        self.initial_seed = seed or 0
        self.t = 0

        # JIT compile if requested
        if enable_jit:
            jit_compile_layout(layout, enable_jit=True, jit_kwargs=jit_kwargs)

        self.core_state, self.rng_state = core_reset(
            layout, seed=self.initial_seed, task_id=task_id
        )

    def step(
        self,
        actions: Optional[Dict[str, Array]] = None,
        collect_timings: bool = False,
    ) -> StepResult:
        """Execute one step."""
        result = core_step(
            self.layout,
            self.core_state,
            self.rng_state,
            self.t,
            actions=actions,
            collect_timings=collect_timings,
        )

        self.core_state = result.core_state
        self.rng_state = result.rng_state
        self.t += 1

        return result

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[CoreState, RNGState]:
        """Reset environment."""
        if seed is None:
            seed = self.initial_seed

        self.core_state, self.rng_state = core_reset(
            self.layout, seed=seed, task_id=task_id, options=options
        )
        self.t = 0

        return self.core_state, self.rng_state

    def get_field(self, field_id: str) -> Array:
        """Get current value of a field."""
        return self.layout.get_field_array(self.core_state, field_id)

    def set_field(self, field_id: str, value: Array) -> None:
        """Set value of a field (for testing/debugging)."""
        if field_id not in self.layout.field_index:
            raise KeyError(f"Unknown field: '{field_id}'")

        cs = list(self.core_state)
        cs[self.layout.field_index[field_id]] = value
        self.core_state = tuple(cs)


def jit_compile_layout(
    layout: CompiledLayout,
    enable_jit: bool = True,
    jit_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    JIT-compile all compatible systems in a layout.

    This should be called once after compilation, before stepping.
    Modifies the layout in-place.

    Args:
        layout: Compiled layout to JIT
        enable_jit: Whether to enable JIT (False for debugging)
        jit_kwargs: Backend-specific JIT options
            - PyTorch: mode='default'|'reduce-overhead'|'max-autotune'
            - JAX: static_argnums, donate_argnums

    Example:
        >>> layout = compile_ir(ir, batch_size=64, backend=torch_backend)
        >>> jit_compile_layout(layout, jit_kwargs={'mode': 'reduce-overhead'})
        >>> runner = EnvRunner(layout)
        >>> # Systems are now JIT-compiled
    """
    if jit_kwargs is None:
        jit_kwargs = {}

    _jit_compile_systems(layout, enable_jit)

    if enable_jit and layout.backend_name in ['torch', 'jax']:
        print(f"JIT-compiled {sum(1 for s in layout.systems if s.jit_compatible)} "
              f"/ {len(layout.systems)} systems using {layout.backend_name}")
