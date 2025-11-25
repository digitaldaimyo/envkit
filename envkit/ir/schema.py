# envkit/ir/schema.py
from __future__ import annotations

from typing import Any, Annotated, Dict, List, Optional, Tuple, Literal, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .string_types import SnakeId, ScratchId, SymbolId, FieldId


# ---------------------------------------------------------------------------
# ID type aliases
# ---------------------------------------------------------------------------

EventId = SnakeId          # could be NameId if you ever need looser rules
GroupId = SnakeId
AggregatorId = SnakeId
RewardChannelId = SnakeId
PredicateId = SnakeId
BundleId = SnakeId
TagId = SnakeId


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class IRBaseModel(BaseModel):
    """
    Base class for all IR models.

    - Forbids extra fields (catches typos early).
    """
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class ShapeLike(IRBaseModel):
    """
    Shape alias: reference another field's shape, optionally selecting dims.

    Example:
      shape: [ShapeLike(like="grid", dims=[0, 1])]  # copy first 2 dims of 'grid'
    """
    like: FieldId
    dims: Optional[List[int]] = None


# ShapeDim: int literal, symbol name, or ShapeLike alias
ShapeDim = Union[int, SymbolId, ShapeLike]


# ---------------------------------------------------------------------------
# Meta & Backends
# ---------------------------------------------------------------------------

class BackendPrefs(IRBaseModel):
    """
    Backend preferences and numeric conventions for a pack.
    """

    preferred: List[str] = Field(
        default_factory=list,
        description="Ordered list of backend names (e.g. ['numpy', 'torch', 'jax']).",
    )
    precision: Literal["float32", "float64"] = Field(
        default="float32",
        description="Default float precision for this pack.",
    )
    vectorized: bool = Field(
        default=True,
        description="Whether systems are written assuming vectorized stepping.",
    )


class SuppressionConfig(IRBaseModel):
    """Explicit error suppressions."""
    linter: List[str] = Field(default_factory=list)
    runtime: List[str] = Field(default_factory=list)


class ValidationConfig(IRBaseModel):
    """Validation and linting configuration."""
    linter_severity: Literal["strict", "moderate", "permissive"] = "strict"
    runtime_validation: Literal["debug", "production", "none"] = "debug"

    suppress: SuppressionConfig = Field(default_factory=SuppressionConfig)
    custom_rules: Dict[str, Any] = Field(default_factory=dict)


class Meta(IRBaseModel):
    """
    Top-level pack metadata.

    Purely descriptive; does not affect runtime semantics except for
    impl_modules and backend preferences.
    """
    env_id: SnakeId
    display_name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    impl_modules: List[str] = Field(
        default_factory=list,
        description="Modules to import so systems/impls are registered.",
    )

    backends: Optional[BackendPrefs] = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

class RNGStreamSpec(IRBaseModel):
    """
    Named RNG stream declaration.

    - id:          symbolic stream name
    - counter_bits: logical counter width (32/64/128 typical)
    """
    id: SnakeId
    counter_bits: int = 64

    @field_validator("counter_bits")
    @classmethod
    def _check_bits(cls, v: int) -> int:
        if v not in (32, 64, 128):
            raise ValueError("counter_bits should typically be 32, 64, or 128")
        return v


class RNGSpec(IRBaseModel):
    """
    RNG configuration for the pack.

    The runtime uses these to materialize backend-native RNG streams.
    """
    streams: List[RNGStreamSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class StateField(IRBaseModel):
    """
    Single state field declaration.

    - id:         snake/scratch id
    - dtype:      one of: int8, int32, uint32, float32, float64, bool
    - shape:      per-env shape expressed via ints, symbols, or ShapeLike
    - persistence:
        * 'permanent' – persists across steps
        * 'scratch'   – reset each step
    """
    kind: Literal["field"] = "field"

    id: FieldId
    dtype: str
    shape: List[ShapeDim] = Field(default_factory=list)
    persistence: Literal["permanent", "scratch"] = "permanent"

    default: Optional[Any] = None
    doc: Optional[str] = None

    bounds: Optional[Dict[str, Any]] = None
    enum: Optional[List[Any]] = None
    tags: List[TagId] = Field(default_factory=list)

    # Mix-in bundles (root_type='state_field')
    bundles: List[BundleId] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventFieldSpec(IRBaseModel):
    """
    Single field inside an event payload.
    """
    id: EventId
    type: str  # "float", "int", "bool", etc.
    doc: Optional[str] = None


class EventChannelSpec(IRBaseModel):
    """
    Structured event channel emitted by systems.

    Events are tensor-based with shape (B, ...) where B is batch size.

    Merge semantics:
    - When multiple systems emit the same event in a phase, field arrays are summed
    - This allows accumulation of event signals (e.g., multiple goals in same step)
    - Systems should emit zeros for envs where event didn't occur

    Example: GoalScored{team: int, value: float}
    """
    id: EventId
    fields: List[EventFieldSpec] = Field(default_factory=list)
    doc: Optional[str] = None
    tags: List[TagId] = Field(default_factory=list)

class EventsSpec(IRBaseModel):
    """
    All event channels for the environment.
    """
    channels: List[EventChannelSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sensors / Observations / Actuators
# ---------------------------------------------------------------------------

# ---- Sensors ----

class SensorView(IRBaseModel):
    """
    'view' sensor: concatenate views of state fields into a 1D vector.

    Output shape (per-env, no batch): (F_total,)
    """
    kind: Literal["view"] = "view"
    id: SnakeId
    from_fields: List[FieldId] = Field(alias="from")
    normalize: bool = False


class SensorExpr(IRBaseModel):
    """
    'expr' sensor: safe expression over state → fixed shape.

    The shape must be fully explicit (no symbols here).
    """
    kind: Literal["expr"] = "expr"
    id: SnakeId
    expr: str
    shape: List[int]          # per-env shape, no batch dim
    normalize: bool = False


class SensorTopK(IRBaseModel):
    """
    'topk' sensor: K nearest neighbors around a center.

    Output shape (per-env): (K, feat_dim)
    """
    kind: Literal["topk"] = "topk"
    id: SnakeId
    pos_field: FieldId
    extra_fields: List[FieldId] = Field(default_factory=list)
    k: int
    center_expr: str
    mask_empty: bool = True
    normalize: bool = True


class SensorMapBins(IRBaseModel):
    """
    'map_bins' sensor: stack per-bin scalar values from multiple fields.

    Output shape (per-env): (K, F)
    """
    kind: Literal["map_bins"] = "map_bins"
    id: SnakeId
    fields: List[FieldId]
    do_normalize: bool = True


class SensorImpl(IRBaseModel):
    """
    'impl' sensor: custom implementation.

    - impl_ref: registry key for Python function
    - shape:    explicit output shape (per-env)
    """
    kind: Literal["impl"] = "impl"
    id: SnakeId
    impl_ref: str
    shape: List[int]
    params: Dict[str, Any] = Field(default_factory=dict)
    normalize: bool = False


SensorSpec = Annotated[
    Union[SensorView, SensorExpr, SensorTopK, SensorMapBins, SensorImpl],
    Field(discriminator="kind"),
]


class ObservationSpec(IRBaseModel):
    """
    Observation mapping for an agent group.

    - id:     observation id
    - sensor: sensor id within the same group
    """
    id: SnakeId
    sensor: SnakeId


# ---- Actuators ----

class DiscreteActuatorSpec(IRBaseModel):
    """
    Discrete actuator constraint: allowable integer values.
    """
    values: List[int]


class BoundsSpec(IRBaseModel):
    """
    Continuous actuator bounds.
    """
    min: float
    max: float


class ActuatorConstraints(IRBaseModel):
    """
    Collection of actuator constraints for a group, keyed by state field id.
    """
    bounds: Dict[FieldId, BoundsSpec] = Field(default_factory=dict)
    discrete: Dict[FieldId, DiscreteActuatorSpec] = Field(default_factory=dict)


class GroupActuators(IRBaseModel):
    """
    Actuator interface for an agent group.

    - writes:      fields the group is allowed to write via actions
    - constraints: optional bounds/discrete constraints
    """
    writes: List[FieldId] = Field(default_factory=list)
    constraints: ActuatorConstraints = Field(default_factory=ActuatorConstraints)


# ---- Agent Groups ----

class AgentAxisSpec(IRBaseModel):
    """
    Bind a state field axis to this group's agent dimension.

    Example:
      field: "red_pos"
      axis:  0      # (B, N_red, 2) → axis 0 after batch is the agent axis
    """
    field: FieldId
    axis: int


class AgentGroupSpec(IRBaseModel):
    """
    Agent group definition: count, sensors, observations, actuators, etc.

    A group represents a homogeneous set of agents that share:
      - the same state-indexing pattern (bind_axes)
      - the same observation/sensor definitions
      - the same actuator interface
    """
    kind: Literal["group"] = "group"

    id: GroupId
    count: int

    # Selection policy for which concrete agents are "active" (future use).
    selection: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "fixed"},
        description="Group selection policy (currently opaque to runtime).",
    )

    bind_axes: List[AgentAxisSpec] = Field(default_factory=list)
    sensors: List[SensorSpec] = Field(default_factory=list)
    observations: List[ObservationSpec] = Field(default_factory=list)
    actuators: Optional[GroupActuators] = None

    # Mix-in bundles (root_type='agent_group')
    bundles: List[BundleId] = Field(default_factory=list)

    tags: List[TagId] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

class BundleInstance(IRBaseModel):
    """
    Inline bundle reference with overrides.

    Used in places like state_schema.fields and agents.groups to
    instantiate a concrete object from a named bundle.
    """
    kind: Literal["bundle"] = "bundle"
    bundle: BundleId
    overrides: Dict[str, Any] = Field(default_factory=dict)


class Bundle(IRBaseModel):
    """
    Generic bundle template.

    - id:        name of the bundle
    - root_type: semantic target type (e.g. 'state_field', 'agent_group')
    - values:    partial config for that type (validated at compile time)
    """
    id: BundleId
    root_type: str
    values: Dict[str, Any] = Field(default_factory=dict)
    tags: List[TagId] = Field(default_factory=list)


# Backwards-compatible alias for any code that referenced BundleSpec
BundleSpec = Bundle


# Type aliases for lists that accept either a concrete spec or a bundle instance
FieldLike = Annotated[
    Union["StateField", "BundleInstance"],
    Field(discriminator="kind"),
]

GroupLike = Annotated[
    Union["AgentGroupSpec", "BundleInstance"],
    Field(discriminator="kind"),
]


class AgentsSpec(IRBaseModel):
    """
    All agent groups participating in the environment.
    """
    groups: List[GroupLike] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# StateSchema (uses FieldLike)
# ---------------------------------------------------------------------------

class StateSchema(IRBaseModel):
    """
    All state fields + shape symbols for an environment.
    """
    symbols: Dict[SymbolId, int] = Field(
        default_factory=dict,
        description="Compile-time integer symbols used in shapes.",
    )
    fields: List[FieldLike] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phases & Systems
# ---------------------------------------------------------------------------

class PhaseSpec(IRBaseModel):
    """
    Execution phase description.

    - schedule:
        * serial   – systems run in deterministic order (rank, then id)
        * parallel – order unspecified; WAW/RAW conflicts disallowed by lints
    - turn:
        * simultaneous / group_order / agent_order / tag_order (future use)
    """
    id: SnakeId
    schedule: Literal["serial", "parallel"] = "parallel"
    turn: Literal["simultaneous", "group_order", "agent_order", "tag_order"] = (
        "simultaneous"
    )
    order: List[SnakeId] = Field(
        default_factory=list,
        description="Group/agent/tag order for non-simultaneous turns.",
    )
    tags: List[TagId] = Field(default_factory=list)


class SystemSpec(IRBaseModel):
    """
    System declaration.

    - phase:      phase id this system belongs to
    - impl_ref:   registry reference 'module.path:function'
    - reads:      fields this system may read
    - writes:     fields this system may write
    - uses_events: events it may emit
    - rng_stream: named RNG stream or None
    """
    id: SnakeId
    phase: SnakeId
    impl_ref: str

    reads: List[FieldId] = Field(default_factory=list)
    writes: List[FieldId] = Field(default_factory=list)
    uses_events: List[SnakeId] = Field(default_factory=list)
    rng_stream: Optional[SnakeId] = None

    params: Dict[str, Any] = Field(default_factory=dict)
    tags: List[TagId] = Field(default_factory=list)
    rank: Optional[int] = Field(
        default=None,
        description="Ordering hint inside phase (0=front, -1=back).",
    )


class SchedulingSpec(IRBaseModel):
    """
    Global phase ordering for the environment.

    Invariants:
      - At least one phase.
      - Phases[0].id == 'reset'
      - Phases[-1].id == 'reward'
    """
    phases: List[PhaseSpec] = Field(default_factory=list)

    @field_validator("phases")
    @classmethod
    def _must_have_reset_and_reward(cls, phases: List[PhaseSpec]) -> List[PhaseSpec]:
        ids = [p.id for p in phases]
        if not ids:
            raise ValueError("At least one phase is required")
        if ids[0] != "reset":
            raise ValueError("Phase list must start with 'reset'")
        if ids[-1] != "reward":
            raise ValueError("Phase list must end with 'reward'")
        return phases


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class RewardTargetMode(str, Enum):
    """
    How a reward channel maps into per-agent rewards.
    """
    PER_ENV = "per_env"      # one scalar per env, broadcast to group agents
    PER_GROUP = "per_group"  # reserved for future; currently same as PER_ENV
    PER_AGENT = "per_agent"  # per-agent rewards (B, count) for that group


class RewardSourceBase(IRBaseModel):
    """
    Base class for reward sources.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)


class RewardSourceField(RewardSourceBase):
    """
    Reward source from a state field.

    Semantics:
      - field is reduced to (B,) by the engine (mean over non-batch dims)
    """
    kind: Literal["field"] = "field"
    field: FieldId


class RewardSourceEvent(RewardSourceBase):
    """
    Reward source from the last payload of an event channel.

    - event: event channel id
    - field: optional key if payload is a dict
    """
    kind: Literal["event"] = "event"
    event: EventId
    field: Optional[str] = None


class RewardSourceExpr(RewardSourceBase):
    """
    Reward source from a safe expression over state+events.
    """
    kind: Literal["expr"] = "expr"
    expr: str


class RewardSourceImpl(RewardSourceBase):
    """
    Custom reward source implemented in Python.

    Registry key:
      REGISTRY.get_reward_impl(impl_ref)
    """
    kind: Literal["impl"] = "impl"
    impl_ref: str
    params: Dict[str, Any] = Field(default_factory=dict)


RewardSource = Annotated[
    Union[
        RewardSourceField,
        RewardSourceEvent,
        RewardSourceExpr,
        RewardSourceImpl,
    ],
    Field(discriminator="kind"),
]


class RewardTarget(IRBaseModel):
    """
    Where to apply a channel within a group.

    Modes:
    - per_env:   scalar per env, broadcast to all agents for that group
    - per_group: (currently same as per_env, reserved for future)
    - per_agent: expects or produces (B, count) shape
    """
    mode: RewardTargetMode = RewardTargetMode.PER_ENV


class RewardChannelSpec(IRBaseModel):
    """
    A single reward component for a given agent group.

    Engine:
      1. Evaluates 'source' to produce array
      2. Normalizes to (B, N_group) based on target.mode
      3. Multiplies by weight
      4. Passes to aggregator
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: RewardChannelId
    group: GroupId

    source: RewardSource
    target: RewardTarget = Field(default_factory=RewardTarget)

    weight: float = 1.0
    tags: List[TagId] = Field(default_factory=list)


class BuiltinAggStrategy(str, Enum):
    """
    Built-in reward aggregation strategies.
    """
    SUM = "sum"
    MEAN = "mean"
    DOT_WEIGHT = "dot_weight"  # weighted sum with weights baked into channels


class AggregatorOutputSpec(IRBaseModel):
    """Direct output targeting for aggregators."""
    target: Literal["global", "per_group", "per_agent"] = "per_group"
    groups: List[GroupId] = Field(
        default_factory=list,
        description="Which groups receive this aggregated reward. Empty = all groups for 'global' target.",
    )


class BuiltinAggregatorSpec(IRBaseModel):
    kind: Literal["builtin"] = "builtin"
    id: AggregatorId
    strategy: BuiltinAggStrategy
    channels: List[RewardChannelId]
    output: AggregatorOutputSpec = Field(default_factory=AggregatorOutputSpec)
    doc: Optional[str] = None


class ImplAggregatorSpec(IRBaseModel):
    kind: Literal["impl"] = "impl"
    id: AggregatorId
    impl_ref: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output: AggregatorOutputSpec = Field(default_factory=AggregatorOutputSpec)
    doc: Optional[str] = None


RewardAggregator = Annotated[
    Union[BuiltinAggregatorSpec, ImplAggregatorSpec],
    Field(discriminator="kind"),
]


class RewardConfig(IRBaseModel):
    """
    Reward configuration for a task.

    - channels:     declared reward components (per-group)
    - aggregators:  how channels are combined
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    channels: List[RewardChannelSpec] = Field(default_factory=list)
    aggregators: List[RewardAggregator] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Termination & Episode config
# ---------------------------------------------------------------------------

class EpisodeConfig(IRBaseModel):
    """
    Per-episode configuration.

    - max_steps:       hard time limit (0 => no time limit)
    - when_time_limit: truncate vs terminate semantics
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = 0
    when_time_limit: Literal["truncate", "terminate"] = "truncate"


class TerminationMode(str, Enum):
    """
    How multiple termination predicates are combined.
    """
    ANY = "any"   # episode ends if any predicate is True (per env)
    ALL = "all"   # episode ends only if all predicates are True (per env)


class TerminationSourceBase(IRBaseModel):
    """
    Base class for termination sources.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)


class TerminationSourceField(TerminationSourceBase):
    """
    Termination predicate from a state field.

    Semantics:
      - reduced to (B,) and treated as bool or numeric != 0.
    """
    kind: Literal["field"] = "field"
    field: FieldId


class TerminationSourceEvent(TerminationSourceBase):
    """
    Termination predicate from an event payload.
    """
    kind: Literal["event"] = "event"
    event: EventId
    field: Optional[str] = None


class TerminationSourceExpr(TerminationSourceBase):
    """
    Termination predicate from a safe expression over state+events.
    """
    kind: Literal["expr"] = "expr"
    expr: str


class TerminationSourceImpl(TerminationSourceBase):
    """
    Custom termination predicate implemented in Python.

    Registry key:
      REGISTRY.get_termination_impl(impl_ref)
    """
    kind: Literal["impl"] = "impl"
    impl_ref: str
    params: Dict[str, Any] = Field(default_factory=dict)


TerminationSource = Annotated[
    Union[
        TerminationSourceField,
        TerminationSourceEvent,
        TerminationSourceExpr,
        TerminationSourceImpl,
    ],
    Field(discriminator="kind"),
]


class TerminationPredicate(IRBaseModel):
    """
    Single termination condition.

    Engine evaluates source → (B,) bool; TerminationConfig.mode
    defines how multiple predicates are combined.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: PredicateId
    source: TerminationSource
    tags: List[TagId] = Field(default_factory=list)


class TerminationConfig(IRBaseModel):
    """
    Termination configuration for a task.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: TerminationMode = TerminationMode.ANY
    predicates: List[TerminationPredicate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

PredicateSourceSpec = TerminationSource


class CurriculumKnob(IRBaseModel):
    """
    Continuous or discrete curriculum knob.

    - range:   [lo, hi] for uniform sampling
    - choices: explicit list of allowed values
    """
    id: SnakeId
    range: Optional[Tuple[float, float]] = None
    choices: Optional[List[float]] = None
    doc: Optional[str] = None


class CurriculumBinding(IRBaseModel):
    """
    Bind a curriculum knob to a state field.
    """
    knob: SnakeId
    state_field: FieldId


class CurriculumStage(IRBaseModel):
    """
    Optional staged curriculum layer.

    - advance_when: termination-like predicate source
    - overrides:    arbitrary IR overrides for the stage (future use)
    """
    id: SnakeId
    advance_when: Optional[PredicateSourceSpec] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    doc: Optional[str] = None


class CurriculumConfig(IRBaseModel):
    """
    Curriculum configuration for a task.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    knobs: List[CurriculumKnob] = Field(default_factory=list)
    bindings: List[CurriculumBinding] = Field(default_factory=list)
    stages: List[CurriculumStage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskConfig(IRBaseModel):
    """
    A single task definition over the same IR/world.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SnakeId
    switchable: bool = False
    phases: List[SnakeId] = Field(default_factory=list)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class TasksConfig(IRBaseModel):
    """
    All tasks for a pack + default task selection.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: List[TaskConfig] = Field(default_factory=list)
    default_task_id: Optional[SnakeId] = None


# ---------------------------------------------------------------------------
# Logging & Replay
# ---------------------------------------------------------------------------

class LoggingMetricSourceExpr(IRBaseModel):
    """
    Logging metric from an expression over state+events.
    """
    type: Literal["expr"] = "expr"
    expr: str


class LoggingMetricSourceImpl(IRBaseModel):
    """
    Logging metric from a custom implementation.
    """
    type: Literal["impl"] = "impl"
    impl_ref: str
    params: Dict[str, Any] = Field(default_factory=dict)


LoggingMetricSource = Annotated[
    Union[LoggingMetricSourceExpr, LoggingMetricSourceImpl],
    Field(discriminator="type"),
]


class LoggingMetricSpec(IRBaseModel):
    """
    Single logging metric declaration.
    """
    id: SnakeId
    source: LoggingMetricSource


class LoggingSpec(IRBaseModel):
    """
    Logging configuration for the environment.
    """
    events: List[SnakeId] = Field(default_factory=list)
    fields: List[FieldId] = Field(default_factory=list)
    metrics: List[LoggingMetricSpec] = Field(default_factory=list)


class ReplaySpec(IRBaseModel):
    """
    Replay configuration.
    """
    include: List[SnakeId] = Field(default_factory=list)
    ring_buffer_steps: int = 0


# ---------------------------------------------------------------------------
# Top-level IR
# ---------------------------------------------------------------------------

class IR(IRBaseModel):
    """
    Top-level IR document.
    """
    ir_version: str

    meta: Meta
    rng: RNGSpec = Field(default_factory=RNGSpec)
    state_schema: StateSchema
    events: EventsSpec = Field(default_factory=EventsSpec)
    agents: AgentsSpec = Field(default_factory=AgentsSpec)

    # Generic bundles
    bundles: List[BundleSpec] = Field(default_factory=list)

    scheduling: SchedulingSpec
    systems: List[SystemSpec] = Field(default_factory=list)
    tasks: TasksConfig = Field(default_factory=TasksConfig)

    logging: LoggingSpec = Field(default_factory=LoggingSpec)
    replay: ReplaySpec = Field(default_factory=ReplaySpec)

    @field_validator("systems")
    @classmethod
    def _check_system_phases(cls, systems: List[SystemSpec], info):
        """
        Ensure every system references a known phase.
        """
        sched: SchedulingSpec = info.data.get("scheduling")  # type: ignore
        if sched is None:
            return systems
        phase_ids = {p.id for p in sched.phases}
        for s in systems:
            if s.phase not in phase_ids:
                raise ValueError(
                    f"System '{s.id}' references unknown phase '{s.phase}'"
                )
        return systems

    @field_validator("tasks")
    @classmethod
    def _check_task_phases(cls, tasks_spec: TasksConfig, info):
        """
        Ensure each task's phase list is compatible with the global SchedulingSpec.
        """
        sched: SchedulingSpec = info.data.get("scheduling")  # type: ignore
        if sched is None:
            return tasks_spec
        phase_ids = {p.id for p in sched.phases}
        for t in tasks_spec.tasks:
            for pid in t.phases:
                if pid not in phase_ids:
                    raise ValueError(
                        f"Task '{t.id}' references unknown phase '{pid}'"
                    )
            if t.phases and (t.phases[0] != "reset" or t.phases[-1] != "reward"):
                raise ValueError(
                    f"Task '{t.id}' phases must start with 'reset' and end with 'reward'"
                )
        return tasks_spec

    # Convenience hook so callers can do `ir.build_shape_service()`
    def build_shape_service(self) -> "ShapeService":
        from .shapes import ShapeService
        return ShapeService(self)
