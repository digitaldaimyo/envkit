# envkit/core/compiled_layout.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------

Array = Any  # Backend-specific array type
CoreState = Tuple[Array, ...]  # Tuple of backend arrays, one per field

# Tensor events:
#   Events[event_id][field_name] = backend Array
#   First dimension is always batch (B); remaining dims are free.
#
#   Merge semantics within a phase:
#     - When multiple systems emit the same event/field, arrays are summed
#     - Allows accumulation of event signals across systems
#     - Systems should emit zeros where event didn't occur
Events = Dict[str, Dict[str, Array]]

# Forward declarations
Backend = Any        # envkit.backend.base.Backend
ShapeService = Any   # envkit.ir.shapes.ShapeService
GroupSpec = Any      # envkit.ir.schema.AgentGroupSpec
TaskConfig = Any     # envkit.ir.schema.TaskConfig
RNGStreamSpec = Any  # envkit.ir.schema.RNGStreamSpec
EventChannelSpec = Any  # envkit.ir.schema.EventChannelSpec

# System function type
SystemFn = Callable[
    [CoreState, "RNGState", Backend, "CompiledLayout", Dict[str, Any]],
    Tuple[CoreState, "RNGState", Events],
]

# ----------------------------------------------------------------------
# RNG State
# ----------------------------------------------------------------------

@dataclass
class RNGState:
    """
    Stateless RNG state using integer counters (v2.1 Section 2).

    Each stream has a counter that increments with each call.
    The actual seed for a call is: hash_combine(base_seed, counter).

    This design is:
    - Fully deterministic and reproducible
    - JIT-friendly (no stateful generators)
    - Backend-agnostic
    - Checkpoint-friendly (just save counters)
    - Parallel-safe (each call gets unique seed)
    """
    counters: Dict[str, int]
    """Per-stream counter values (stream_id -> counter)"""

    base_seed: int
    """Global seed for reproducibility"""

    def get_stream_seed(self, stream_id: str) -> int:
        """
        Compute stateless seed for this stream call.

        This uses a deterministic hash combining base_seed and counter.
        """
        # Simple hash combination (production should use better hash)
        return hash((self.base_seed, stream_id, self.counters[stream_id])) & 0x7FFFFFFF

    def increment(self, stream_id: str) -> None:
        """Increment counter for stream after use."""
        self.counters[stream_id] += 1


# ----------------------------------------------------------------------
# Field Specification
# ----------------------------------------------------------------------

@dataclass
class FieldSpec:
    """
    Resolved field specification at compile time.

    All shapes are per-environment (no batch dimension).
    At runtime, actual arrays have shape (B, *shape).
    """
    dtype: str
    """Data type: 'float32', 'float64', 'int32', 'int64', 'bool'"""

    shape: Tuple[int, ...]
    """Resolved per-env shape (no batch dim)"""

    persistence: str
    """'permanent' or 'scratch'"""

    default: Optional[Any] = None
    """Default value for initialization"""

    bounds: Optional[Dict[str, Any]] = None
    """Optional min/max bounds"""

    enum: Optional[List[Any]] = None
    """Optional discrete valid values"""


# ----------------------------------------------------------------------
# Compiled System
# ----------------------------------------------------------------------

@dataclass
class CompiledSystem:
    """
    Metadata for a compiled system (v2.1 Section 7.3).

    Systems are pure functions that transform CoreState.
    """
    id: str
    """Unique system identifier"""

    phase: str
    """Phase this system executes in"""

    impl_fn: SystemFn
    """Actual callable system function"""

    reads: List[str]
    """Field IDs this system reads"""

    writes: List[str]
    """Field IDs this system writes"""

    uses_events: List[str]
    """Event IDs this system may emit"""

    rng_stream: Optional[str]
    """RNG stream this system uses (if any)"""

    params: Dict[str, Any]
    """System-specific parameters"""

    rank: int
    """Execution priority in serial phases (lower = earlier)"""

    jit_compatible: bool
    """Whether this system is JIT-compilable"""

    requires_grad: bool
    """Whether this system needs gradient tracking"""

    doc: Optional[str] = None
    """Human-readable description"""

# ----------------------------------------------------------------------
# Compiled Layout
# ----------------------------------------------------------------------

@dataclass
class CompiledLayout:
    """
    Fully resolved, IR-derived structure for runtime execution (v2.1 Section 13.1).

    This is the primary data structure passed to all runtime functions.
    It contains all information needed for stepping, rewards, termination.
    """

    # ---- State layout ----
    B: int
    """Batch size"""

    field_ids: List[str]
    """Ordered list of field IDs (matches CoreState tuple order)"""

    field_index: Dict[str, int]
    """field_id -> index in CoreState tuple"""

    field_specs: Dict[str, FieldSpec]
    """field_id -> resolved field specification"""

    # ---- Agent group metadata ----
    groups: Dict[str, GroupSpec]
    """group_id -> agent group specification"""

    # ---- Phase execution ----
    phases: List[str]
    """Ordered phase IDs for execution"""

    phase_system_ranges: List[Tuple[int, int]]
    """[start, end) indices in systems array for each phase"""

    systems: List[CompiledSystem]
    """All systems in execution order"""

    # ---- Backend ----
    backend: Backend
    """Backend operations provider (numpy/torch/jax)"""

    backend_name: str
    """Backend name: 'numpy', 'torch', 'jax'"""

    # ---- Task configuration ----
    task_configs: Dict[str, TaskConfig]
    """task_id -> task configuration (for runtime switching)"""

    active_task_id: str
    """Currently active task ID"""

    # ---- Validation ----
    validation_mode: str
    """Validation level: 'debug', 'production', 'none'"""

    # ---- RNG ----
    rng_streams: Dict[str, RNGStreamSpec]
    """stream_id -> RNG stream specification"""

    # ---- Shape service ----
    shape_service: ShapeService
    """Shape resolution and query service"""

    # ---- Events ----
    event_specs: Dict[str, EventChannelSpec] = field(default_factory=dict)
    """event_id -> event channel specification"""

    # ---- Scratch field tracking ----
    scratch_field_ids: List[str] = field(default_factory=list)
    """Field IDs with persistence='scratch' (reset each step)"""

    # ---- IR version ----
    ir_version: str = "2.1"
    """Version of IR spec this layout was compiled from"""

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------

    def get_active_task(self) -> TaskConfig:
        """Get the currently active task configuration."""
        return self.task_configs[self.active_task_id]

    def switch_task(self, task_id: str) -> None:
        """
        Switch to a different task at reset boundary.

        Raises:
            ValueError: If task is not switchable or doesn't exist
        """
        if task_id not in self.task_configs:
            raise ValueError(f"Unknown task: '{task_id}'")

        task = self.task_configs[task_id]
        if not getattr(task, "switchable", False):
            raise ValueError(f"Task '{task_id}' is not switchable")

        self.active_task_id = task_id

    # ------------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------------

    def get_field_array(self, core_state: CoreState, field_id: str) -> Array:
        """
        Helper to read a field array from CoreState.

        Returns:
            Backend array with shape (B, *field_shape)
        """
        if field_id not in self.field_index:
            raise KeyError(f"Unknown field: '{field_id}'")
        return core_state[self.field_index[field_id]]

    def get_scratch_indices(self) -> List[int]:
        """Get CoreState indices for all scratch fields."""
        return [self.field_index[fid] for fid in self.scratch_field_ids]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_core_state(self, core_state: CoreState) -> None:
        """
        Validate that CoreState structure matches compiled layout.

        Checks:
        - Correct number of arrays
        - Correct shapes (if validation_mode != 'none')
        - Correct dtypes (if validation_mode == 'debug')

        Only runs if validation_mode != 'none'.
        """
        if self.validation_mode == "none":
            return

        # Check length
        if len(core_state) != len(self.field_ids):
            raise ValueError(
                f"CoreState has {len(core_state)} arrays, "
                f"expected {len(self.field_ids)}"
            )

        if self.validation_mode == "debug":
            # Full validation
            be = self.backend
            for field_id, idx in self.field_index.items():
                arr = core_state[idx]
                spec = self.field_specs[field_id]

                expected_shape = (self.B,) + spec.shape
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"Field '{field_id}' has shape {arr.shape}, "
                        f"expected {expected_shape}"
                    )

                # TODO: dtype checking (backend-specific)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def create_initial_core_state(
    layout: CompiledLayout,
) -> CoreState:
    """
    Create initial CoreState from field defaults.

    Args:
        layout: Compiled layout with field specs

    Returns:
        CoreState tuple with all fields initialized to defaults
    """
    be = layout.backend
    B = layout.B

    arrays = []
    for field_id in layout.field_ids:
        spec = layout.field_specs[field_id]
        shape = (B,) + spec.shape

        # Get backend dtype
        dtype_name = spec.dtype + "_dtype"
        dtype = getattr(be, dtype_name, be.float_dtype)

        # Initialize from default
        if spec.default is not None:
            arr = be.asarray(spec.default, dtype=dtype)
            # Broadcast to batch shape
            if arr.shape != shape:
                arr = be.broadcast_to(arr, shape)
        else:
            # Zero initialization
            arr = be.zeros(shape, dtype=dtype)

        arrays.append(arr)

    return tuple(arrays)


def create_initial_rng_state(
    layout: CompiledLayout,
    seed: int = 0,
) -> RNGState:
    """
    Create initial RNG state with all counters at zero.

    Args:
        layout: Compiled layout with RNG stream specs
        seed: Base seed for reproducibility

    Returns:
        RNGState with counters initialized to 0
    """
    return RNGState(
        counters={stream_id: 0 for stream_id in layout.rng_streams},
        base_seed=seed,
    )


def reset_scratch_fields(
    core_state: CoreState,
    layout: CompiledLayout,
) -> CoreState:
    """
    Reset all scratch fields to their default values.

    This is called at the start of each step (or in a dedicated phase).

    Args:
        core_state: Current core state
        layout: Compiled layout

    Returns:
        CoreState with scratch fields reset
    """
    if not layout.scratch_field_ids:
        return core_state

    be = layout.backend
    B = layout.B
    cs = list(core_state)

    for field_id in layout.scratch_field_ids:
        idx = layout.field_index[field_id]
        spec = layout.field_specs[field_id]
        shape = (B,) + spec.shape

        # Get backend dtype
        dtype_name = spec.dtype + "_dtype"
        dtype = getattr(be, dtype_name, be.float_dtype)

        # Reset to default
        if spec.default is not None:
            arr = be.asarray(spec.default, dtype=dtype)
            arr = be.broadcast_to(arr, shape)
        else:
            arr = be.zeros(shape, dtype=dtype)

        cs[idx] = arr

    return tuple(cs)
