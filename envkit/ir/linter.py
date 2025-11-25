
# envkit/ir/linter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any
from enum import Enum

from .schema import (
    IR,
    RewardSourceEvent,
    RewardSourceField,
    TerminationSourceEvent,
    TerminationSourceField,
)
from .shapes import ShapeService, ShapeResolutionError


# ----------------------------------------------------------------------
# Error severity and codes
# ----------------------------------------------------------------------

class ErrorSeverity(Enum):
    """Linter issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintIssue:
    """A single linter issue (error, warning, or info)."""
    code: str
    severity: ErrorSeverity
    message: str
    location: str
    suggestion: Optional[str] = None
    suppressed: bool = False

    def __str__(self) -> str:
        prefix = (
            "❌"
            if self.severity == ErrorSeverity.ERROR
            else "⚠️"
            if self.severity == ErrorSeverity.WARNING
            else "ℹ️"
        )
        s = f"{prefix} [{self.code}] {self.message}"
        if self.location:
            s += f"\n  Location: {self.location}"
        if self.suggestion:
            s += f"\n  Fix: {self.suggestion}"
        if self.suppressed:
            s += "\n  (suppressed)"
        return s


# ----------------------------------------------------------------------
# Linter Report
# ----------------------------------------------------------------------

@dataclass
class LintReport:
    """Collection of linter issues with filtering and reporting."""
    issues: List[LintIssue] = field(default_factory=list)
    severity: str = "strict"  # strict | moderate | permissive

    def suppress(self, code: str) -> None:
        """Mark all issues with this code as suppressed."""
        for issue in self.issues:
            if issue.code == code:
                issue.suppressed = True

    def has_errors(self, include_suppressed: bool = False) -> bool:
        """Check if report has any errors."""
        for issue in self.issues:
            if issue.suppressed and not include_suppressed:
                continue
            if self._is_error(issue):
                return True
        return False

    def has_warnings(self, include_suppressed: bool = False) -> bool:
        """Check if report has any warnings."""
        for issue in self.issues:
            if issue.suppressed and not include_suppressed:
                continue
            if issue.severity == ErrorSeverity.WARNING:
                return True
        return False

    def _is_error(self, issue: LintIssue) -> bool:
        """Determine if issue counts as error given severity level."""
        if self.severity == "strict":
            # E and W are errors
            return issue.severity in (ErrorSeverity.ERROR, ErrorSeverity.WARNING)
        elif self.severity == "moderate":
            # Only E are errors
            return issue.severity == ErrorSeverity.ERROR
        else:  # permissive
            # Only critical E codes are errors
            return issue.severity == ErrorSeverity.ERROR and issue.code.startswith("E0")

    @property
    def errors(self) -> List[LintIssue]:
        """Get all error-level issues (not suppressed)."""
        return [i for i in self.issues if self._is_error(i) and not i.suppressed]

    @property
    def warnings(self) -> List[LintIssue]:
        """Get all warning-level issues (not suppressed)."""
        return [i for i in self.issues if i.severity == ErrorSeverity.WARNING and not i.suppressed]

    @property
    def infos(self) -> List[LintIssue]:
        """Get all info-level issues (not suppressed)."""
        return [i for i in self.issues if i.severity == ErrorSeverity.INFO and not i.suppressed]

    def summary(self) -> str:
        """Generate summary string."""
        total_errors = len([i for i in self.issues if self._is_error(i)])
        suppressed_errors = len(
            [i for i in self.issues if self._is_error(i) and i.suppressed]
        )

        total_warnings = len(
            [i for i in self.issues if i.severity == ErrorSeverity.WARNING]
        )
        suppressed_warnings = len(
            [i for i in self.issues if i.severity == ErrorSeverity.WARNING and i.suppressed]
        )

        total_infos = len([i for i in self.issues if i.severity == ErrorSeverity.INFO])

        lines = [
            f"Linting completed with severity: {self.severity}",
            f"  Errors: {total_errors - suppressed_errors} ({suppressed_errors} suppressed)",
            f"  Warnings: {total_warnings - suppressed_warnings} ({suppressed_warnings} suppressed)",
            f"  Info: {total_infos}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = [self.summary(), ""]

        for issue in self.errors:
            lines.append(str(issue))
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for issue in self.warnings:
                lines.append(str(issue))
                lines.append("")

        return "\n".join(lines)


# ----------------------------------------------------------------------
# Linter
# ----------------------------------------------------------------------

class Linter:
    """
    Static analysis and validation for IR specifications.

    Checks:
    - Undefined references (fields, groups, events, systems, phases, bundles)
    - Shape compatibility
    - WAW/RAW conflicts in parallel phases
    - Required phases (reset first, reward last)
    - Type mismatches
    - Suspicious patterns and unused declarations
    """

    def __init__(self):
        self.issues: List[LintIssue] = []

    def lint(self, ir: IR, severity: str = "strict") -> LintReport:
        """
        Run all linter checks on IR.

        NOTE: The compiler runs bundle expansion before linting, so the IR
        here should already have concrete StateField / AgentGroupSpec entries
        (no BundleInstance in fields/groups).
        """
        self.issues = []

        # Check 1: Core structure
        self._check_phases(ir)

        # Check 2: Build reference sets
        field_ids = self._get_field_ids(ir)
        group_ids = self._get_group_ids(ir)
        event_ids = self._get_event_ids(ir)
        phase_ids = self._get_phase_ids(ir)
        rng_stream_ids = self._get_rng_stream_ids(ir)
        bundle_ids = self._get_bundle_ids(ir)

        # Check 2.5: Event schema validation
        self._check_event_schemas(ir)

        # Check 3: Undefined references
        self._check_system_references(ir, field_ids, event_ids, phase_ids, rng_stream_ids)
        self._check_group_references(ir, field_ids)
        self._check_group_bundle_references(ir, bundle_ids)
        self._check_task_references(ir, phase_ids)
        self._check_reward_references(ir, field_ids, event_ids, group_ids)
        self._check_termination_references(ir, field_ids, event_ids)
        self._check_logging_references(ir, field_ids, event_ids)
        self._check_task_episode_sanity(ir)

        # Check 4: Shape resolution (and group/axis sanity)
        self._check_shapes(ir)

        # Check 5: Conflicts in parallel phases
        self._check_conflicts(ir)

        # Check 6: Unused declarations (fields, events, groups, bundles)
        self._check_unused(ir, field_ids, event_ids, group_ids, bundle_ids)

        # Check 7: Performance warnings
        self._check_performance(ir)

        return LintReport(issues=self.issues, severity=severity)

    # ------------------------------------------------------------------
    # Phase validation
    # ------------------------------------------------------------------

    def _check_phases(self, ir: IR) -> None:
        """Validate phase structure."""
        if not ir.scheduling.phases:
            self.issues.append(
                LintIssue(
                    code="E200",
                    severity=ErrorSeverity.ERROR,
                    message="At least one phase is required",
                    location="scheduling.phases",
                    suggestion="Add phase definitions to scheduling",
                )
            )
            return

        phase_ids = [p.id for p in ir.scheduling.phases]

        # Check first phase is reset
        if phase_ids[0] != "reset":
            self.issues.append(
                LintIssue(
                    code="E200",
                    severity=ErrorSeverity.ERROR,
                    message=f"First phase must be 'reset', got '{phase_ids[0]}'",
                    location="scheduling.phases[0]",
                    suggestion="Change first phase to 'reset' or reorder phases",
                )
            )

        # Check last phase is reward
        if phase_ids[-1] != "reward":
            self.issues.append(
                LintIssue(
                    code="E201",
                    severity=ErrorSeverity.ERROR,
                    message=f"Last phase must be 'reward', got '{phase_ids[-1]}'",
                    location=f"scheduling.phases[{len(phase_ids)-1}]",
                    suggestion="Change last phase to 'reward' or reorder phases",
                )
            )

        # Check for duplicate phase IDs
        seen = set()
        for i, phase_id in enumerate(phase_ids):
            if phase_id in seen:
                self.issues.append(
                    LintIssue(
                        code="E202",
                        severity=ErrorSeverity.ERROR,
                        message=f"Duplicate phase ID: '{phase_id}'",
                        location=f"scheduling.phases[{i}]",
                        suggestion="Use unique phase IDs",
                    )
                )
            seen.add(phase_id)

    # ------------------------------------------------------------------
    # Reference collection
    # ------------------------------------------------------------------

    def _get_field_ids(self, ir: IR) -> Set[str]:
        """Get all declared field IDs."""
        return {str(f.id) for f in ir.state_schema.fields}

    def _get_group_ids(self, ir: IR) -> Set[str]:
        """Get all declared agent group IDs."""
        return {str(g.id) for g in ir.agents.groups}

    def _get_event_ids(self, ir: IR) -> Set[str]:
        """Get all declared event IDs."""
        return {str(e.id) for e in ir.events.channels}

    def _get_phase_ids(self, ir: IR) -> Set[str]:
        """Get all declared phase IDs."""
        return {str(p.id) for p in ir.scheduling.phases}

    def _get_rng_stream_ids(self, ir: IR) -> Set[str]:
        """Get all declared RNG stream IDs."""
        return {str(s.id) for s in ir.rng.streams}

    def _get_bundle_ids(self, ir: IR) -> Set[str]:
        """Get all declared bundle IDs."""
        return {str(b.id) for b in getattr(ir, "bundles", [])}

    # ------------------------------------------------------------------
    # Reference validation
    # ------------------------------------------------------------------

    def _check_system_references(
        self,
        ir: IR,
        field_ids: Set[str],
        event_ids: Set[str],
        phase_ids: Set[str],
        rng_stream_ids: Set[str],
    ) -> None:
        """Check system references are valid."""
        for sys in ir.systems:
            # Check phase reference
            if str(sys.phase) not in phase_ids:
                self.issues.append(
                    LintIssue(
                        code="E003",
                        severity=ErrorSeverity.ERROR,
                        message=f"System '{sys.id}' references unknown phase '{sys.phase}'",
                        location=f"systems.{sys.id}.phase",
                        suggestion=f"Use one of: {', '.join(sorted(phase_ids))}",
                    )
                )

            # Check field reads
            for field_id in sys.reads:
                if str(field_id) not in field_ids:
                    self.issues.append(
                        LintIssue(
                            code="E001",
                            severity=ErrorSeverity.ERROR,
                            message=f"System '{sys.id}' reads unknown field '{field_id}'",
                            location=f"systems.{sys.id}.reads",
                            suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                        )
                    )

            # Check field writes
            for field_id in sys.writes:
                if str(field_id) not in field_ids:
                    self.issues.append(
                        LintIssue(
                            code="E001",
                            severity=ErrorSeverity.ERROR,
                            message=f"System '{sys.id}' writes unknown field '{field_id}'",
                            location=f"systems.{sys.id}.writes",
                            suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                        )
                    )

            # Check event references
            for event_id in sys.uses_events:
                if str(event_id) not in event_ids:
                    self.issues.append(
                        LintIssue(
                            code="E004",
                            severity=ErrorSeverity.ERROR,
                            message=f"System '{sys.id}' emits unknown event '{event_id}'",
                            location=f"systems.{sys.id}.uses_events",
                            suggestion=f"Use one of: {', '.join(sorted(event_ids))}",
                        )
                    )

            # Check RNG stream reference
            if sys.rng_stream is not None and str(sys.rng_stream) not in rng_stream_ids:
                self.issues.append(
                    LintIssue(
                        code="E005",
                        severity=ErrorSeverity.ERROR,
                        message=f"System '{sys.id}' uses unknown RNG stream '{sys.rng_stream}'",
                        location=f"systems.{sys.id}.rng_stream",
                        suggestion=f"Use one of: {', '.join(sorted(rng_stream_ids))}",
                    )
                )


    def _check_event_schemas(self, ir: IR) -> None:
        """
        Validate event channel schemas.

        Checks:
        - Event field types are valid
        - No duplicate field names
        - Event channels have unique IDs
        """
        seen_events = set()

        for event_spec in ir.events.channels:
            event_id = str(event_spec.id)

            # Check duplicate event IDs
            if event_id in seen_events:
                self.issues.append(
                    LintIssue(
                        code="E204",
                        severity=ErrorSeverity.ERROR,
                        message=f"Duplicate event channel ID: '{event_id}'",
                        location=f"events.channels.{event_id}",
                        suggestion="Use unique event IDs",
                    )
                )
            seen_events.add(event_id)

            # Check for duplicate field names
            field_names = [str(f.id) for f in event_spec.fields]
            duplicates = [name for name in field_names if field_names.count(name) > 1]
            if duplicates:
                self.issues.append(
                    LintIssue(
                        code="E205",
                        severity=ErrorSeverity.ERROR,
                        message=f"Event '{event_id}' has duplicate field names: {set(duplicates)}",
                        location=f"events.channels.{event_id}.fields",
                        suggestion="Use unique field names within an event",
                    )
                )

            # Validate field types
            valid_types = {"float", "int", "bool", "float32", "float64", "int32", "int64"}
            for field in event_spec.fields:
                if field.type not in valid_types:
                    self.issues.append(
                        LintIssue(
                            code="W211",
                            severity=ErrorSeverity.WARNING,
                            message=f"Event '{event_id}' field '{field.id}' has unusual type '{field.type}'",
                            location=f"events.channels.{event_id}.fields.{field.id}",
                            suggestion=f"Use one of: {', '.join(sorted(valid_types))}",
                        )
                    )

    def _check_group_references(self, ir: IR, field_ids: Set[str]) -> None:
        """Check agent group references to fields are valid."""
        for group in ir.agents.groups:
            # bind_axes references
            for binding in group.bind_axes:
                if str(binding.field) not in field_ids:
                    self.issues.append(
                        LintIssue(
                            code="E001",
                            severity=ErrorSeverity.ERROR,
                            message=(
                                f"Group '{group.id}' bind_axes references unknown field "
                                f"'{binding.field}'"
                            ),
                            location=f"agents.groups.{group.id}.bind_axes",
                            suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                        )
                    )

            # actuator writes & constraints
            if group.actuators:
                # writes
                for field_id in group.actuators.writes:
                    if str(field_id) not in field_ids:
                        self.issues.append(
                            LintIssue(
                                code="E001",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Group '{group.id}' actuators write unknown field "
                                    f"'{field_id}'"
                                ),
                                location=f"agents.groups.{group.id}.actuators.writes",
                                suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                            )
                        )

                # constraints reference known fields
                constraints = group.actuators.constraints
                for fid in constraints.bounds.keys():
                    if fid not in field_ids:
                        self.issues.append(
                            LintIssue(
                                code="E001",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Group '{group.id}' actuator bounds reference unknown "
                                    f"field '{fid}'"
                                ),
                                location=f"agents.groups.{group.id}.actuators.constraints.bounds",
                                suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                            )
                        )
                    elif fid not in {str(w) for w in group.actuators.writes}:
                        self.issues.append(
                            LintIssue(
                                code="W210",
                                severity=ErrorSeverity.WARNING,
                                message=(
                                    f"Group '{group.id}' has bounds for field '{fid}' "
                                    f"which is not in actuators.writes"
                                ),
                                location=f"agents.groups.{group.id}.actuators.constraints.bounds",
                                suggestion="Add field to actuators.writes or remove bounds",
                            )
                        )

                for fid in constraints.discrete.keys():
                    if fid not in field_ids:
                        self.issues.append(
                            LintIssue(
                                code="E001",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Group '{group.id}' actuator discrete set references "
                                    f"unknown field '{fid}'"
                                ),
                                location=f"agents.groups.{group.id}.actuators.constraints.discrete",
                                suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                            )
                        )
                    elif fid not in {str(w) for w in group.actuators.writes}:
                        self.issues.append(
                            LintIssue(
                                code="W210",
                                severity=ErrorSeverity.WARNING,
                                message=(
                                    f"Group '{group.id}' has discrete constraints for field '{fid}' "
                                    f"which is not in actuators.writes"
                                ),
                                location=f"agents.groups.{group.id}.actuators.constraints.discrete",
                                suggestion="Add field to actuators.writes or remove constraints",
                            )
                        )

    def _check_group_bundle_references(self, ir: IR, bundle_ids: Set[str]) -> None:
        """Check that groups reference existing bundles and track usage."""
        for group in ir.agents.groups:
            for bid in getattr(group, "bundles", []):
                if str(bid) not in bundle_ids:
                    self.issues.append(
                        LintIssue(
                            code="E007",
                            severity=ErrorSeverity.ERROR,
                            message=(
                                f"Group '{group.id}' references unknown bundle '{bid}'"
                            ),
                            location=f"agents.groups.{group.id}.bundles",
                            suggestion=f"Use one of: {', '.join(sorted(bundle_ids))}"
                            if bundle_ids
                            else "Define the bundle in top-level 'bundles'",
                        )
                    )

    def _check_task_references(self, ir: IR, phase_ids: Set[str]) -> None:
        """Check task phase references are valid."""
        for task in ir.tasks.tasks:
            for phase_id in task.phases:
                if str(phase_id) not in phase_ids:
                    self.issues.append(
                        LintIssue(
                            code="E003",
                            severity=ErrorSeverity.ERROR,
                            message=f"Task '{task.id}' references unknown phase '{phase_id}'",
                            location=f"tasks.{task.id}.phases",
                            suggestion=f"Use one of: {', '.join(sorted(phase_ids))}",
                        )
                    )

    def _check_reward_references(
        self,
        ir: IR,
        field_ids: Set[str],
        event_ids: Set[str],
        group_ids: Set[str],
    ) -> None:
        """Check reward channel and aggregator references."""
        for task in ir.tasks.tasks:
            # Check reward channels
            channel_ids = set()
            for channel in task.reward.channels:
                channel_ids.add(str(channel.id))

                # Check group reference
                if str(channel.group) not in group_ids:
                    self.issues.append(
                        LintIssue(
                            code="E002",
                            severity=ErrorSeverity.ERROR,
                            message=(
                                f"Reward channel '{channel.id}' references unknown group "
                                f"'{channel.group}'"
                            ),
                            location=f"tasks.{task.id}.reward.channels.{channel.id}",
                            suggestion=f"Use one of: {', '.join(sorted(group_ids))}",
                        )
                    )

                # Check source references – distinguish event vs field
                src = channel.source
                if isinstance(src, RewardSourceEvent):
                    # Validate event id
                    if str(src.event) not in event_ids:
                        self.issues.append(
                            LintIssue(
                                code="E004",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Reward channel '{channel.id}' event source references "
                                    f"unknown event '{src.event}'"
                                ),
                                location=f"tasks.{task.id}.reward.channels.{channel.id}.source",
                                suggestion=f"Use one of: {', '.join(sorted(event_ids))}",
                            )
                        )
                    # Validate payload field name if provided
                    if src.field is not None:
                        event_channel = next(
                            (e for e in ir.events.channels if str(e.id) == str(src.event)),
                            None,
                        )
                        if event_channel:
                            event_field_ids = {str(f.id) for f in event_channel.fields}
                            if src.field not in event_field_ids:
                                self.issues.append(
                                    LintIssue(
                                        code="E001",
                                        severity=ErrorSeverity.ERROR,
                                        message=(
                                            f"Reward channel '{channel.id}' event source "
                                            f"references unknown event field '{src.field}'"
                                        ),
                                        location=f"tasks.{task.id}.reward.channels.{channel.id}.source",
                                        suggestion=(
                                            f"Event '{src.event}' has fields: "
                                            f"{', '.join(sorted(event_field_ids))}"
                                        ),
                                    )
                                )
                elif isinstance(src, RewardSourceField):
                    if str(src.field) not in field_ids:
                        self.issues.append(
                            LintIssue(
                                code="E001",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Reward channel '{channel.id}' source references unknown "
                                    f"state field '{src.field}'"
                                ),
                                location=f"tasks.{task.id}.reward.channels.{channel.id}.source",
                                suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                            )
                        )

            # Check aggregators
            for agg in task.reward.aggregators:
                # Check channel references
                for ch_id in agg.channels:
                    if str(ch_id) not in channel_ids:
                        self.issues.append(
                            LintIssue(
                                code="E006",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Aggregator '{agg.id}' references unknown channel '{ch_id}'"
                                ),
                                location=f"tasks.{task.id}.reward.aggregators.{agg.id}",
                                suggestion=f"Use one of: {', '.join(sorted(channel_ids))}",
                            )
                        )

                # Check output group references
                if hasattr(agg, "output") and hasattr(agg.output, "groups"):
                    for group_id in agg.output.groups:
                        if str(group_id) not in group_ids:
                            self.issues.append(
                                LintIssue(
                                    code="E002",
                                    severity=ErrorSeverity.ERROR,
                                    message=(
                                        f"Aggregator '{agg.id}' output references unknown group "
                                        f"'{group_id}'"
                                    ),
                                    location=f"tasks.{task.id}.reward.aggregators.{agg.id}.output",
                                    suggestion=f"Use one of: {', '.join(sorted(group_ids))}",
                                )
                            )

    def _check_termination_references(
        self,
        ir: IR,
        field_ids: Set[str],
        event_ids: Set[str],
    ) -> None:
        """Check termination predicate references."""
        for task in ir.tasks.tasks:
            for pred in task.termination.predicates:
                src = pred.source

                # State field predicate
                if isinstance(src, TerminationSourceField):
                    if str(src.field) not in field_ids:
                        self.issues.append(
                            LintIssue(
                                code="E001",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Termination predicate '{pred.id}' references unknown "
                                    f"state field '{src.field}'"
                                ),
                                location=f"tasks.{task.id}.termination.predicates.{pred.id}",
                                suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                            )
                        )

                # Event predicate
                if isinstance(src, TerminationSourceEvent):
                    if str(src.event) not in event_ids:
                        self.issues.append(
                            LintIssue(
                                code="E004",
                                severity=ErrorSeverity.ERROR,
                                message=(
                                    f"Termination predicate '{pred.id}' references unknown "
                                    f"event '{src.event}'"
                                ),
                                location=f"tasks.{task.id}.termination.predicates.{pred.id}",
                                suggestion=f"Use one of: {', '.join(sorted(event_ids))}",
                            )
                        )

                    # Payload field check if provided
                    if src.field is not None:
                        event_channel = next(
                            (e for e in ir.events.channels if str(e.id) == str(src.event)),
                            None,
                        )
                        if event_channel:
                            event_field_ids = {str(f.id) for f in event_channel.fields}
                            if src.field not in event_field_ids:
                                self.issues.append(
                                    LintIssue(
                                        code="E001",
                                        severity=ErrorSeverity.ERROR,
                                        message=(
                                            f"Termination predicate '{pred.id}' event source "
                                            f"references unknown event field '{src.field}'"
                                        ),
                                        location=f"tasks.{task.id}.termination.predicates.{pred.id}",
                                        suggestion=(
                                            f"Event '{src.event}' has fields: "
                                            f"{', '.join(sorted(event_field_ids))}"
                                        ),
                                    )
                                )

    def _check_logging_references(
        self,
        ir: IR,
        field_ids: Set[str],
        event_ids: Set[str],
    ) -> None:
        """Check logging spec references."""
        log = ir.logging

        for eid in log.events:
            if str(eid) not in event_ids:
                self.issues.append(
                    LintIssue(
                        code="E004",
                        severity=ErrorSeverity.ERROR,
                        message=f"Logging references unknown event '{eid}'",
                        location="logging.events",
                        suggestion=f"Use one of: {', '.join(sorted(event_ids))}",
                    )
                )

        for fid in log.fields:
            if str(fid) not in field_ids:
                self.issues.append(
                    LintIssue(
                        code="E001",
                        severity=ErrorSeverity.ERROR,
                        message=f"Logging references unknown field '{fid}'",
                        location="logging.fields",
                        suggestion=f"Use one of: {', '.join(sorted(field_ids))}",
                    )
                )

        # Duplicate metric IDs
        seen = set()
        for metric in log.metrics:
            mid = str(metric.id)
            if mid in seen:
                self.issues.append(
                    LintIssue(
                        code="E203",
                        severity=ErrorSeverity.ERROR,
                        message=f"Duplicate logging metric id '{mid}'",
                        location=f"logging.metrics.{mid}",
                        suggestion="Use unique metric IDs",
                    )
                )
            seen.add(mid)

    def _check_task_episode_sanity(self, ir: IR) -> None:
        """Warn about tasks that can run forever with no termination."""
        for task in ir.tasks.tasks:
            max_steps = getattr(task.episode, "max_steps", 0)
            has_preds = bool(task.termination.predicates)
            if max_steps == 0 and not has_preds:
                self.issues.append(
                    LintIssue(
                        code="W300",
                        severity=ErrorSeverity.WARNING,
                        message=(
                            f"Task '{task.id}' has no time limit and no termination predicates "
                            f"(episodes may be infinite)"
                        ),
                        location=f"tasks.{task.id}",
                        suggestion="Set episode.max_steps or add termination predicates",
                    )
                )

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    def _check_shapes(self, ir: IR) -> None:
        """Validate shape resolution and group/axis consistency."""
        shape_service: Optional[ShapeService] = None
        try:
            shape_service = ir.build_shape_service()
        except ShapeResolutionError as e:
            self.issues.append(
                LintIssue(
                    code="E300",
                    severity=ErrorSeverity.ERROR,
                    message=f"Shape resolution failed: {e}",
                    location="state_schema.fields",
                    suggestion="Check symbol definitions and ShapeLike references",
                )
            )
            # If shapes don't resolve, skip group/axis shape checks
            return

        # Group axis sanity: check axis indexing and count vs axis size
        for group in ir.agents.groups:
            for binding in group.bind_axes:
                field_id = str(binding.field)
                try:
                    shape = shape_service.field_shape(field_id)
                except Exception:
                    # Already covered by shape resolution / field checks
                    continue

                axis = binding.axis
                if axis < 0 or axis >= len(shape):
                    self.issues.append(
                        LintIssue(
                            code="E301",
                            severity=ErrorSeverity.ERROR,
                            message=(
                                f"Group '{group.id}' bind_axes uses invalid axis {axis} "
                                f"for field '{field_id}' with shape {shape}"
                            ),
                            location=f"agents.groups.{group.id}.bind_axes",
                            suggestion="Use an axis index in range "
                                       f"[0, {len(shape) - 1}] for this field",
                        )
                    )
                    continue

                dim_size = shape[axis]
                if dim_size > 0 and dim_size != group.count:
                    self.issues.append(
                        LintIssue(
                            code="W301",
                            severity=ErrorSeverity.WARNING,
                            message=(
                                f"Group '{group.id}' count={group.count} does not match "
                                f"bound axis {axis} size ({dim_size}) for field '{field_id}'"
                            ),
                            location=f"agents.groups.{group.id}.bind_axes",
                            suggestion=(
                                "Set group.count to the axis size or adjust bind_axes "
                                "if agents do not bijectively map to that axis"
                            ),
                        )
                    )

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _check_conflicts(self, ir: IR) -> None:
        """Check for WAW/RAW conflicts in parallel phases."""
        for phase in ir.scheduling.phases:
            if phase.schedule != "parallel":
                continue

            # Get systems in this phase
            systems = [s for s in ir.systems if s.phase == phase.id]

            # Build read/write sets
            reads_by_sys = {s.id: {str(f) for f in s.reads} for s in systems}
            writes_by_sys = {s.id: {str(f) for f in s.writes} for s in systems}

            # Check for conflicts
            for i, sys1 in enumerate(systems):
                for sys2 in systems[i + 1 :]:
                    # WAW: both write to same field
                    waw = writes_by_sys[sys1.id] & writes_by_sys[sys2.id]
                    if waw:
                        self.issues.append(
                            LintIssue(
                                code="W100",
                                severity=ErrorSeverity.WARNING,
                                message=(
                                    f"Write-after-write conflict in phase '{phase.id}': "
                                    f"systems '{sys1.id}' and '{sys2.id}' both write to {waw}"
                                ),
                                location=f"systems.{sys1.id}",
                                suggestion="Use serial scheduling or separate phases",
                            )
                        )

                    # RAW: sys1 reads what sys2 writes (or vice versa)
                    raw1 = reads_by_sys[sys1.id] & writes_by_sys[sys2.id]
                    raw2 = reads_by_sys[sys2.id] & writes_by_sys[sys1.id]
                    if raw1 or raw2:
                        fields = raw1 | raw2
                        self.issues.append(
                            LintIssue(
                                code="W101",
                                severity=ErrorSeverity.WARNING,
                                message=(
                                    f"Read-after-write conflict in phase '{phase.id}': "
                                    f"'{sys1.id}' and '{sys2.id}' have RAW on {fields}"
                                ),
                                location=f"systems.{sys1.id}",
                                suggestion="Use serial scheduling or separate phases",
                            )
                        )

    # ------------------------------------------------------------------
    # Unused detection
    # ------------------------------------------------------------------

    def _check_unused(
        self,
        ir: IR,
        field_ids: Set[str],
        event_ids: Set[str],
        group_ids: Set[str],
        bundle_ids: Set[str],
    ) -> None:
        """Check for unused declarations."""
        # --- Fields: read/write usage by systems ---
        field_reads: Set[str] = set()
        field_writes: Set[str] = set()
        for sys in ir.systems:
            field_reads.update(str(f) for f in sys.reads)
            field_writes.update(str(f) for f in sys.writes)

        for f in ir.state_schema.fields:
            fid = str(f.id)
            if fid not in field_reads and fid not in field_writes:
                self.issues.append(
                    LintIssue(
                        code="I001",
                        severity=ErrorSeverity.INFO,
                        message=f"Field '{fid}' is never read or written",
                        location=f"state_schema.fields.{fid}",
                        suggestion="Remove if unused or add a system that uses it",
                    )
                )

        # --- Events: used by systems, rewards, termination, logging ---
        events_used: Set[str] = set()
        for sys in ir.systems:
            events_used.update(str(eid) for eid in sys.uses_events)

        for task in ir.tasks.tasks:
            for ch in task.reward.channels:
                if isinstance(ch.source, RewardSourceEvent):
                    events_used.add(str(ch.source.event))
            for pred in task.termination.predicates:
                src = pred.source
                if isinstance(src, TerminationSourceEvent):
                    events_used.add(str(src.event))

        for eid in ir.logging.events:
            events_used.add(str(eid))

        for e in ir.events.channels:
            eid = str(e.id)
            if eid not in events_used:
                self.issues.append(
                    LintIssue(
                        code="I003",
                        severity=ErrorSeverity.INFO,
                        message=f"Event '{eid}' is never emitted or consumed",
                        location=f"events.channels.{eid}",
                        suggestion="Remove if unused or wire into systems/rewards/logging",
                    )
                )

        # --- Groups: used by reward channels/aggregators or actuators ---
        groups_with_rewards: Set[str] = set()
        groups_with_agg_outputs: Set[str] = set()
        for task in ir.tasks.tasks:
            for ch in task.reward.channels:
                groups_with_rewards.add(str(ch.group))
            for agg in task.reward.aggregators:
                if hasattr(agg, "output") and hasattr(agg.output, "groups"):
                    groups_with_agg_outputs.update(str(g) for g in agg.output.groups)

        groups_with_actuators: Set[str] = set(
            str(g.id) for g in ir.agents.groups if g.actuators is not None
        )

        groups_used = groups_with_rewards | groups_with_agg_outputs | groups_with_actuators

        for gid in group_ids:
            if gid not in groups_used:
                self.issues.append(
                    LintIssue(
                        code="I004",
                        severity=ErrorSeverity.INFO,
                        message=f"Agent group '{gid}' is never rewarded and has no actuators",
                        location=f"agents.groups.{gid}",
                        suggestion="Remove if unused or connect it to rewards/actuators",
                    )
                )

        # --- Bundles: referenced by any group? ---
        used_bundle_ids: Set[str] = set()
        for group in ir.agents.groups:
            for bid in getattr(group, "bundles", []):
                used_bundle_ids.add(str(bid))

        for b in getattr(ir, "bundles", []):
            bid = str(b.id)
            if bid not in used_bundle_ids:
                self.issues.append(
                    LintIssue(
                        code="I002",
                        severity=ErrorSeverity.INFO,
                        message=f"Bundle '{bid}' is never referenced by any group",
                        location=f"bundles.{bid}",
                        suggestion="Remove if unused or reference it from agents.groups[*].bundles",
                    )
                )

    # ------------------------------------------------------------------
    # Performance warnings
    # ------------------------------------------------------------------

    def _check_performance(self, ir: IR) -> None:
        """Check for potential performance issues."""
        # Check for large field counts
        if len(ir.state_schema.fields) > 100:
            self.issues.append(
                LintIssue(
                    code="W200",
                    severity=ErrorSeverity.WARNING,
                    message=(
                        f"Large number of state fields ({len(ir.state_schema.fields)})"
                    ),
                    location="state_schema.fields",
                    suggestion=(
                        "Consider grouping related fields or using nested structures"
                    ),
                )
            )

        # Check for very deep nesting
        for f in ir.state_schema.fields:
            if len(f.shape) > 5:
                self.issues.append(
                    LintIssue(
                        code="W201",
                        severity=ErrorSeverity.WARNING,
                        message=(
                            f"Field '{f.id}' has deep nesting (rank {len(f.shape)})"
                        ),
                        location=f"state_schema.fields.{f.id}",
                        suggestion="Consider flattening shape or restructuring data",
                    )
                )
