# envkit/core/compiler.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import importlib

from envkit.ir.schema import (
    IR,
    TaskConfig,
    StateField,
    AgentGroupSpec,
    BundleInstance,
)
from envkit.ir.shapes import ShapeService
from envkit.ir.linter import Linter, LintReport
from envkit.core.registry import REGISTRY
from envkit.core.compiled_layout import (
    CompiledLayout,
    CompiledSystem,
    FieldSpec,
)


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class CompilationError(Exception):
    """Raised when IR compilation fails."""
    pass


# ----------------------------------------------------------------------
# Compiler
# ----------------------------------------------------------------------

def compile_ir(
    ir: IR,
    batch_size: int,
    backend: Any,
    task_id: Optional[str] = None,
    validation_mode: str = "debug",
    lint_severity: Optional[str] = None,
    skip_linting: bool = False,
    skip_imports: bool = False,
    enable_jit: bool = False,
    jit_kwargs: Optional[Dict[str, Any]] = None,
) -> CompiledLayout:
    """
    Compile IR to executable CompiledLayout.

    Args:
        ir: IR specification
        batch_size: Number of parallel environments
        backend: Backend instance (numpy/torch/jax)
        task_id: Active task ID
        validation_mode: 'debug', 'production', or 'none'
        lint_severity: 'strict', 'moderate', or 'permissive'
        skip_linting: Skip linter checks
        skip_imports: Skip importing impl_modules
        enable_jit: JIT-compile compatible systems
        jit_kwargs: Backend-specific JIT options

    Returns:
        Compiled layout ready for execution
    """

    # Step 1: Expand bundles (prototype + mixins) for fields & groups
    _expand_bundles(ir)

    # Step 2: Linting
    if not skip_linting:
        lint_report = _run_linter(ir, lint_severity)
        if lint_report.has_errors():
            raise CompilationError(
                f"Linting failed:\n{lint_report}\n\n"
                f"To see all issues, check the lint report. "
                f"To suppress specific errors, add them to meta.validation.suppress.linter"
            )

    # Step 3: Import implementations
    if not skip_imports:
        _import_implementations(ir)

    # Step 4: Resolve shapes
    try:
        shape_service = ShapeService(ir)
    except Exception as e:
        raise CompilationError(f"Shape resolution failed: {e}") from e

    # Step 5: Build field index and specs
    field_ids, field_index, field_specs = _build_field_metadata(ir, shape_service)

    # Step 6: Compile systems
    compiled_systems = _compile_systems(ir)

   # Step 6.5: Validate system event usage
    _validate_system_events(ir, compiled_systems)

    # Step 7: Build phase execution order and globally ordered systems
    phases, phase_system_ranges, compiled_systems = _build_phase_order(
        ir, compiled_systems
    )

    # Step 8: Select and validate task
    task_configs, active_task_id = _select_task(ir, task_id)

    # Step 9: Collect scratch fields
    scratch_field_ids = [
        str(f.id)
        for f in ir.state_schema.fields
        if isinstance(f, StateField) and f.persistence == "scratch"
    ]

    # Step 10: Build compiled layout
    layout = CompiledLayout(
        # State layout
        B=batch_size,
        field_ids=field_ids,
        field_index=field_index,
        field_specs=field_specs,

        # Agent group metadata
        groups={str(g.id): g for g in ir.agents.groups},  # all are AgentGroupSpec now

        # Phase execution
        phases=phases,
        phase_system_ranges=phase_system_ranges,
        systems=compiled_systems,

        # Backend
        backend=backend,
        backend_name=backend.__class__.__name__,

        # Task configuration
        task_configs=task_configs,
        active_task_id=active_task_id,

        # Validation
        validation_mode=validation_mode,

        # RNG
        rng_streams={str(s.id): s for s in ir.rng.streams},

        # Shape service
        shape_service=shape_service,

        # Events
        event_specs={str(e.id): e for e in ir.events.channels},

        # Scratch fields
        scratch_field_ids=scratch_field_ids,

        # IR version
        ir_version=getattr(ir, "ir_version", "2.1"),
    )

    # JIT compile if requested
    if enable_jit:
        from envkit.core.runtime import jit_compile_layout
        jit_compile_layout(layout, enable_jit=True, jit_kwargs=jit_kwargs)

    return layout


# ----------------------------------------------------------------------
# Step 2: Linting
# ----------------------------------------------------------------------

def _run_linter(ir: IR, lint_severity: Optional[str]) -> LintReport:
    """
    Run linter and apply suppressions from IR.
    """
    if lint_severity is None:
        if ir.meta.validation:
            lint_severity = ir.meta.validation.linter_severity
        else:
            lint_severity = "strict"

    linter = Linter()
    report = linter.lint(ir, severity=lint_severity)

    if ir.meta.validation and ir.meta.validation.suppress:
        for code in ir.meta.validation.suppress.linter:
            report.suppress(code)

    return report


# ----------------------------------------------------------------------
# Step 3: Import implementations
# ----------------------------------------------------------------------

def _import_implementations(ir: IR) -> None:
    """
    Import all impl_modules to register systems/rewards/etc.
    """
    impl_modules = getattr(ir.meta, "impl_modules", []) or []

    for mod_name in impl_modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            raise CompilationError(
                f"Failed to import implementation module '{mod_name}': {e}\n"
                f"Make sure the module is installed and on the Python path."
            ) from e
        except Exception as e:
            raise CompilationError(
                f"Error while importing module '{mod_name}': {e}"
            ) from e


# ----------------------------------------------------------------------
# Step 3.5: Bundle expansion
# ----------------------------------------------------------------------

def _expand_bundles(ir: IR) -> None:
    """
    Expand bundle prototypes + mixins for fields and groups, in-place.

    After this runs:
      - ir.state_schema.fields -> List[StateField]
      - ir.agents.groups       -> List[AgentGroupSpec]
      - all field/group bundles have been applied
    """
    bundle_map: Dict[str, Any] = {str(b.id): b for b in ir.bundles}

    # ---- Fields: prototypes + mixins ----
    expanded_fields: List[StateField] = []
    for item in ir.state_schema.fields:
        # Prototype from bundle
        if isinstance(item, BundleInstance):
            bundle_id = str(item.bundle)
            if bundle_id not in bundle_map:
                raise CompilationError(
                    f"Field bundle '{bundle_id}' not found in ir.bundles"
                )
            bundle = bundle_map[bundle_id]
            if bundle.root_type not in ("state_field", "field"):
                raise CompilationError(
                    f"Bundle '{bundle_id}' has root_type='{bundle.root_type}' "
                    f"but is used in state_schema.fields"
                )

            base_data = dict(bundle.values)
            base_data.update(item.overrides)

            try:
                field = StateField.model_validate(base_data)
            except Exception as e:
                raise CompilationError(
                    f"Failed to instantiate StateField from bundle '{bundle_id}': {e}"
                ) from e
        else:
            # Already a concrete StateField
            field = item

        # Mix-in bundles for fields
        if getattr(field, "bundles", None):
            field = _apply_field_bundles(field, bundle_map)

        expanded_fields.append(field)

    ir.state_schema.fields = expanded_fields  # type: ignore

    # ---- Groups: prototypes + mixins ----
    expanded_groups: List[AgentGroupSpec] = []
    for item in ir.agents.groups:
        # Prototype from bundle
        if isinstance(item, BundleInstance):
            bundle_id = str(item.bundle)
            if bundle_id not in bundle_map:
                raise CompilationError(
                    f"Group bundle '{bundle_id}' not found in ir.bundles"
                )
            bundle = bundle_map[bundle_id]
            if bundle.root_type not in ("agent_group", "group"):
                raise CompilationError(
                    f"Bundle '{bundle_id}' has root_type='{bundle.root_type}' "
                    f"but is used in agents.groups"
                )

            base_data = dict(bundle.values)
            base_data.update(item.overrides)

            try:
                group = AgentGroupSpec.model_validate(base_data)
            except Exception as e:
                raise CompilationError(
                    f"Failed to instantiate AgentGroupSpec from bundle '{bundle_id}': {e}"
                ) from e
        else:
            # Already a concrete AgentGroupSpec
            group = item

        # Mix-in bundles for groups
        if getattr(group, "bundles", None):
            group = _apply_group_bundles(group, bundle_map)

        expanded_groups.append(group)

    ir.agents.groups = expanded_groups  # type: ignore


def _apply_field_bundles(
    field: StateField,
    bundle_map: Dict[str, Any],
) -> StateField:
    """
    Apply mix-in bundles (root_type='state_field') to a StateField.
    """
    result = field.model_copy(deep=True)

    for bundle_id in field.bundles:
        bid = str(bundle_id)
        if bid not in bundle_map:
            raise CompilationError(
                f"State field '{field.id}' references unknown bundle '{bid}'"
            )

        bundle = bundle_map[bid]
        if bundle.root_type not in ("state_field", "field"):
            raise CompilationError(
                f"Bundle '{bid}' has root_type='{bundle.root_type}' "
                f"but is used as a field mixin"
            )

        bundle_field = StateField.model_validate(bundle.values)
        result = _merge_state_field(result, bundle_field)

    # Clear bundles to avoid double-application downstream
    result.bundles = []
    return result


def _apply_group_bundles(
    group: AgentGroupSpec,
    bundle_map: Dict[str, Any],
) -> AgentGroupSpec:
    """
    Apply mix-in bundles (root_type='agent_group') to an AgentGroupSpec.
    """
    result = group.model_copy(deep=True)

    for bundle_id in group.bundles:
        bid = str(bundle_id)
        if bid not in bundle_map:
            raise CompilationError(
                f"Agent group '{group.id}' references unknown bundle '{bid}'"
            )

        bundle = bundle_map[bid]
        if bundle.root_type not in ("agent_group", "group"):
            raise CompilationError(
                f"Bundle '{bid}' has root_type='{bundle.root_type}' "
                f"but is used as a group mixin"
            )

        bundle_group = AgentGroupSpec.model_validate(bundle.values)
        result = _merge_agent_group(result, bundle_group)

    result.bundles = []
    return result


def _merge_state_field(base: StateField, addon: StateField) -> StateField:
    """
    Merge addon (from bundle) into base field.

    Rules:
      - Scalars: base wins; addon only fills None/doc if missing.
      - bounds:  per-key union, base overrides bundle on conflicts.
      - enum:    base.enum wins if present; else use bundle.enum.
      - tags:    set union (bundle tags + base tags).
    """
    merged = base.model_copy(deep=True)

    # doc: only fill if missing
    if merged.doc is None and addon.doc is not None:
        merged.doc = addon.doc

    # bounds: bundle provides defaults; field overrides keys
    if addon.bounds:
        merged_bounds: Dict[str, Any] = {}
        merged_bounds.update(addon.bounds)
        if merged.bounds:
            merged_bounds.update(merged.bounds)
        merged.bounds = merged_bounds

    # enum: only if field didn't set one
    if merged.enum is None and addon.enum is not None:
        merged.enum = list(addon.enum)

    # tags: union, keep order
    merged.tags = list(dict.fromkeys((addon.tags or []) + (merged.tags or [])))

    return merged


def _merge_agent_group(base: AgentGroupSpec, addon: AgentGroupSpec) -> AgentGroupSpec:
    """
    Merge addon (from bundle) into base group.

    Rules:
      - id/count: base wins (we treat them as "identity"/shape).
      - selection: base wins if explicitly set; else bundle's selection.
      - bind_axes/sensors/observations: concatenate (bundle first).
      - actuators:
          * if base.actuators is None -> adopt bundle's actuators
          * else union writes, merge constraints (bundle default, base override)
      - tags: union (bundle tags + base tags).
    """
    merged = base.model_copy(deep=True)

    # selection: fill from bundle if base uses default
    if merged.selection == {"type": "fixed"} and addon.selection != {"type": "fixed"}:
        merged.selection = addon.selection

    # bind_axes / sensors / observations: prepend bundle entries
    merged.bind_axes = list(addon.bind_axes) + list(merged.bind_axes)
    merged.sensors = list(addon.sensors) + list(merged.sensors)
    merged.observations = list(addon.observations) + list(merged.observations)

    # actuators
    if addon.actuators is not None:
        if merged.actuators is None:
            merged.actuators = addon.actuators
        else:
            merged.actuators = _merge_actuators(merged.actuators, addon.actuators)

    # tags: union
    merged.tags = list(dict.fromkeys((addon.tags or []) + (merged.tags or [])))

    return merged


def _merge_actuators(base: Any, addon: Any) -> Any:
    """
    Merge addon GroupActuators into base GroupActuators.

    Rules:
      - writes: union, preserving order (addon first).
      - bounds/discrete: bundle provides defaults, base overrides per field.
    """
    merged = base.model_copy(deep=True)

    # writes: union with bundle first
    merged.writes = list(
        dict.fromkeys(list(addon.writes) + list(merged.writes))
    )

    # bounds
    b_bounds = merged.constraints.bounds or {}
    a_bounds = addon.constraints.bounds or {}
    merged_bounds: Dict[Any, Any] = {}
    merged_bounds.update(a_bounds)
    merged_bounds.update(b_bounds)
    merged.constraints.bounds = merged_bounds

    # discrete
    b_disc = merged.constraints.discrete or {}
    a_disc = addon.constraints.discrete or {}
    merged_disc: Dict[Any, Any] = {}
    merged_disc.update(a_disc)
    merged_disc.update(b_disc)
    merged.constraints.discrete = merged_disc

    return merged


# ----------------------------------------------------------------------
# Step 5: Build field metadata
# ----------------------------------------------------------------------

def _build_field_metadata(
    ir: IR,
    shape_service: ShapeService,
) -> Tuple[List[str], Dict[str, int], Dict[str, FieldSpec]]:
    """
    Build field index and specifications.
    """
    field_ids = [str(f.id) for f in ir.state_schema.fields]  # all StateField now
    field_index = {fid: idx for idx, fid in enumerate(field_ids)}

    field_specs: Dict[str, FieldSpec] = {}
    for f in ir.state_schema.fields:
        fid = str(f.id)
        shape = shape_service.field_shape(fid)

        field_specs[fid] = FieldSpec(
            dtype=f.dtype,
            shape=shape,
            persistence=f.persistence,
            default=f.default,
            bounds=f.bounds,
            enum=f.enum,
        )

    return field_ids, field_index, field_specs


# ----------------------------------------------------------------------
# Step 6: Compile systems
# ----------------------------------------------------------------------

def _compile_systems(ir: IR) -> List[CompiledSystem]:
    """
    Compile all systems by fetching implementations from registry.
    """
    compiled_systems: List[CompiledSystem] = []

    for sys_spec in ir.systems:
        impl_ref = sys_spec.impl_ref
        if not REGISTRY.has_system(impl_ref):
            raise CompilationError(
                f"System implementation not found: '{impl_ref}'\n"
                f"Make sure the implementation module has been imported and "
                f"the function is decorated with @register_system"
            )

        impl_fn = REGISTRY.get_system(impl_ref)

        # Contract
        contract = REGISTRY.system_contracts.get(impl_ref)
        jit_compatible = contract.jit_compatible if contract else True
        requires_grad = contract.requires_grad if contract else False

        if contract:
            _validate_system_contract(sys_spec, contract)

        compiled_systems.append(
            CompiledSystem(
                id=str(sys_spec.id),
                phase=str(sys_spec.phase),
                impl_fn=impl_fn,
                reads=[str(f) for f in sys_spec.reads],
                writes=[str(f) for f in sys_spec.writes],
                uses_events=[str(e) for e in sys_spec.uses_events],
                rng_stream=str(sys_spec.rng_stream) if sys_spec.rng_stream else None,
                params=dict(sys_spec.params),
                rank=sys_spec.rank if sys_spec.rank is not None else 0,
                jit_compatible=jit_compatible,
                requires_grad=requires_grad,
                doc=getattr(sys_spec, "doc", None),
            )
        )

    return compiled_systems

# ----------------------------------------------------------------------
# Step 6.5: Validate system events and contracts
# ----------------------------------------------------------------------

def _validate_system_contract(sys_spec: Any, contract: Any) -> None:
    """
    Validate that system spec matches its contract.
    """
    spec_reads = set(str(f) for f in sys_spec.reads)
    contract_reads = set(contract.reads)

    if contract_reads and not contract_reads.issubset(spec_reads):
        missing = contract_reads - spec_reads
        raise CompilationError(
            f"System '{sys_spec.id}' contract declares reads {sorted(contract.reads)}, "
            f"but IR spec is missing: {sorted(missing)}\n"
            f"Add missing fields to 'reads' in the IR or update the contract."
        )

    spec_writes = set(str(f) for f in sys_spec.writes)
    contract_writes = set(contract.writes)

    if contract_writes and not contract_writes.issubset(spec_writes):
        missing = contract_writes - spec_writes
        raise CompilationError(
            f"System '{sys_spec.id}' contract declares writes {sorted(contract.writes)}, "
            f"but IR spec is missing: {sorted(missing)}\n"
            f"Add missing fields to 'writes' in the IR or update the contract."
        )

def _validate_system_events(ir: IR, compiled_systems: List[CompiledSystem]) -> None:
    """
    Validate that systems' event usage matches declared event schemas.

    Checks:
    - Events in uses_events exist in ir.events.channels
    - (Future: could validate emitted event fields match schema)
    """
    event_specs = {str(e.id): e for e in ir.events.channels}

    for sys in compiled_systems:
        for event_id in sys.uses_events:
            if event_id not in event_specs:
                raise CompilationError(
                    f"System '{sys.id}' uses unknown event '{event_id}'\n"
                    f"Available events: {', '.join(sorted(event_specs.keys()))}"
                )


# ----------------------------------------------------------------------
# Step 7: Build phase execution order
# ----------------------------------------------------------------------

def _build_phase_order(
    ir: IR,
    compiled_systems: List[CompiledSystem],
) -> Tuple[List[str], List[Tuple[int, int]], List[CompiledSystem]]:
    """
    Build phase execution order and system ranges.

    We rebuild a *global* ordered systems list:

      - phases appear in ir.scheduling.phases order
      - within each phase, systems are sorted by (rank, id)
      - phase_system_ranges give [start, end) slices into this list
    """
    phases = [str(p.id) for p in ir.scheduling.phases]

    # Group systems by phase id (using the compiled system's phase string)
    systems_by_phase: Dict[str, List[CompiledSystem]] = {}
    for sys in compiled_systems:
        systems_by_phase.setdefault(sys.phase, []).append(sys)

    ordered_systems: List[CompiledSystem] = []
    phase_system_ranges: List[Tuple[int, int]] = []

    for phase_id in phases:
        systems_in_phase = systems_by_phase.get(phase_id, [])

        # Sort by (rank, id) for deterministic order
        systems_in_phase.sort(key=lambda s: (s.rank, s.id))

        start_idx = len(ordered_systems)
        ordered_systems.extend(systems_in_phase)
        end_idx = len(ordered_systems)

        phase_system_ranges.append((start_idx, end_idx))

    return phases, phase_system_ranges, ordered_systems


# ----------------------------------------------------------------------
# Step 8: Task selection
# ----------------------------------------------------------------------

def _select_task(
    ir: IR,
    task_id: Optional[str],
) -> Tuple[Dict[str, TaskConfig], str]:
    """
    Select active task and validate.
    """
    if not ir.tasks.tasks:
        raise CompilationError(
            "No tasks defined in IR. At least one task is required."
        )

    task_configs = {str(t.id): t for t in ir.tasks.tasks}

    if task_id is None:
        if ir.tasks.default_task_id:
            active_task_id = str(ir.tasks.default_task_id)
        else:
            active_task_id = str(ir.tasks.tasks[0].id)
    else:
        active_task_id = task_id

    if active_task_id not in task_configs:
        available = ", ".join(sorted(task_configs.keys()))
        raise CompilationError(
            f"Task '{active_task_id}' not found. Available tasks: {available}"
        )

    return task_configs, active_task_id


# ----------------------------------------------------------------------
# Utility: Compile multiple tasks at once
# ----------------------------------------------------------------------

def compile_ir_multi_task(
    ir: IR,
    batch_size: int,
    backend: Any,
    task_ids: List[str],
    validation_mode: str = "debug",
    lint_severity: Optional[str] = None,
) -> CompiledLayout:
    """
    Compile IR with multiple tasks for runtime switching.
    """
    task_configs = {str(t.id): t for t in ir.tasks.tasks}

    for tid in task_ids:
        if tid not in task_configs:
            available = ", ".join(sorted(task_configs.keys()))
            raise CompilationError(
                f"Task '{tid}' not found. Available: {available}"
            )

        task = task_configs[tid]
        if not getattr(task, "switchable", False):
            raise CompilationError(
                f"Task '{tid}' is not marked as switchable. "
                f"Add 'switchable: true' to the task definition."
            )

    first_phases = set(str(p) for p in task_configs[task_ids[0]].phases)
    for tid in task_ids[1:]:
        task_phases = set(str(p) for p in task_configs[tid].phases)
        if task_phases != first_phases:
            raise CompilationError(
                f"Switchable tasks must use the same phase set. "
                f"Task '{task_ids[0]}' uses {sorted(first_phases)}, "
                f"but task '{tid}' uses {sorted(task_phases)}"
            )

    layout = compile_ir(
        ir,
        batch_size=batch_size,
        backend=backend,
        task_id=task_ids[0],
        validation_mode=validation_mode,
        lint_severity=lint_severity,
    )

    return layout


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def validate_compiled_layout(layout: CompiledLayout) -> List[str]:
    """
    Run post-compilation validation checks.
    """
    warnings: List[str] = []

    # Empty phases
    for phase_id, (start, end) in zip(layout.phases, layout.phase_system_ranges):
        if start == end:
            warnings.append(
                f"Phase '{phase_id}' has no systems. "
                f"Consider removing it or adding systems."
            )

    # Systems with no effect
    for sys in layout.systems:
        if not sys.reads and not sys.writes:
            warnings.append(
                f"System '{sys.id}' has no reads or writes. "
                f"This system has no effect on state."
            )

    if layout.B <= 0:
        warnings.append(
            f"Batch size B={layout.B} is invalid. Must be > 0."
        )

    return warnings
