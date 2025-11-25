
from __future__ import annotations

from typing import Tuple, Dict, Any

from envkit.core.compiled_layout import CompiledLayout, CoreState, Events, Array
from envkit.core.registry import REGISTRY
from envkit.ir.schema import (
    TerminationPredicate,
    TerminationSourceField,
    TerminationSourceEvent,
    TerminationSourceExpr,
    TerminationSourceImpl,
    TerminationMode,
)


# ----------------------------------------------------------------------
# Termination Engine
# ----------------------------------------------------------------------

class TerminationEngine:
    """
    Termination computation engine (v2.1 Section 9).

    Evaluates termination predicates and handles time limits:
    1. Evaluate all predicates to (B,) bool arrays
    2. Combine via mode (any/all)
    3. Check time limits
    4. Return (terminated, truncated)
    """

    def __init__(self, layout: CompiledLayout):
        self.layout = layout
        self.task = layout.get_active_task()
        self.termination_config = self.task.termination
        self.episode_config = self.task.episode

    def compute(
        self,
        core_state: CoreState,
        events: Events,
        layout: CompiledLayout,
        B: int,
        t: int,
    ) -> Tuple[Array, Array]:
        """
        Compute termination and truncation masks.
        """
        be = layout.backend

        # Step 1: Evaluate predicates
        terminated = self._evaluate_predicates(core_state, events, be, B)

        # Step 2: Check time limits
        truncated = self._check_time_limit(t, be, B)

        # Step 3: Apply time limit behavior
        if self.episode_config.when_time_limit == "terminate":
            # Time limit causes termination, not truncation
            terminated = terminated | truncated
            truncated = be.zeros((B,), dtype=be.bool_dtype)

        return terminated, truncated

    # ------------------------------------------------------------------
    # Predicate evaluation
    # ------------------------------------------------------------------

    def _evaluate_predicates(
        self,
        core_state: CoreState,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """
        Evaluate all termination predicates.

        Returns:
            (B,) bool array
        """
        if not self.termination_config.predicates:
            # No predicates, never terminate
            return be.zeros((B,), dtype=be.bool_dtype)

        # Evaluate each predicate
        pred_results = []
        for pred in self.termination_config.predicates:
            try:
                result = self._evaluate_predicate(pred, core_state, events, be, B)
                pred_results.append(result)
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating termination predicate '{pred.id}': {e}"
                ) from e

        # Combine via mode
        if self.termination_config.mode == TerminationMode.ANY:
            combined = pred_results[0]
            for pred_result in pred_results[1:]:
                combined = combined | pred_result
            return combined

        elif self.termination_config.mode == TerminationMode.ALL:
            combined = pred_results[0]
            for pred_result in pred_results[1:]:
                combined = combined & pred_result
            return combined

        else:
            raise ValueError(f"Unknown termination mode: {self.termination_config.mode}")

    def _evaluate_predicate(
        self,
        pred: TerminationPredicate,
        core_state: CoreState,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """
        Evaluate a single termination predicate.

        Returns:
            (B,) bool array
        """
        if isinstance(pred.source, TerminationSourceField):
            return self._eval_field_source(pred.source, core_state, be, B)
        elif isinstance(pred.source, TerminationSourceEvent):
            return self._eval_event_source(pred.source, events, be, B)
        elif isinstance(pred.source, TerminationSourceExpr):
            return self._eval_expr_source(pred.source, core_state, events, be)
        elif isinstance(pred.source, TerminationSourceImpl):
            return self._eval_impl_source(pred.source, core_state, events, be, B)
        else:
            raise ValueError(f"Unknown predicate source type: {type(pred.source)}")

    def _eval_field_source(
        self,
        source: TerminationSourceField,
        core_state: CoreState,
        be: Any,
        B: int,
    ) -> Array:
        """Evaluate field source."""
        field_id = str(source.field)
        if field_id not in self.layout.field_index:
            raise KeyError(f"Unknown state field in termination source: '{field_id}'")

        idx = self.layout.field_index[field_id]
        arr = core_state[idx]  # (B, *shape)

        # Reduce to (B,) bool
        if arr.shape == (B,):
            result = arr
        else:
            flat = be.reshape(arr, (B, -1))
            result = be.asarray(flat, dtype=be.bool_dtype)
            xp = be.xp
            if hasattr(xp, "any"):
                result = xp.any(result, axis=1)
            else:
                result = be.sum(result, axis=1) > 0

        return result  # (B,) bool

    def _eval_event_source(
        self,
        source: TerminationSourceEvent,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """
        Evaluate event-based termination from tensor events.

        Semantics:
          - events[event_id] is a dict of field_name -> Array
          - We consider the event "active" for env b if ANY field has a
            non-zero / True value for that env (over all other dims).
          - Returns (B,) bool.
        """
        event_id = str(source.event)

        if event_id not in events:
            return be.zeros((B,), dtype=be.bool_dtype)

        chan = events[event_id]
        if not isinstance(chan, dict) or not chan:
            return be.zeros((B,), dtype=be.bool_dtype)

        be_backend = be
        xp = be_backend.xp

        combined = None

        for arr in chan.values():
            a = be_backend.asarray(arr)
            if a.shape[0] != B:
                raise ValueError(
                    f"Event '{event_id}' field has shape {a.shape}, "
                    f"but batch size is B={B}"
                )

            if xp.__name__ == "torch":
                a_bool = a != 0
            else:
                a_bool = a != 0

            if a_bool.ndim > 1:
                axes = tuple(range(1, a_bool.ndim))
                if hasattr(xp, "any"):
                    reduced = xp.any(a_bool, axis=axes)
                else:
                    reduced = be_backend.sum(a_bool, axis=axes) > 0
            else:
                reduced = a_bool  # (B,)

            if combined is None:
                combined = reduced
            else:
                if hasattr(xp, "logical_or"):
                    combined = xp.logical_or(combined, reduced)
                else:
                    combined = (combined | reduced)

        return be_backend.asarray(combined, dtype=be_backend.bool_dtype)

    def _eval_expr_source(
        self,
        source: TerminationSourceExpr,
        core_state: CoreState,
        events: Events,
        be: Any,
    ) -> Array:
        """Evaluate expression source."""
        try:
            from envkit.engines.expr_eval import evaluate_expression

            result = evaluate_expression(
                source.expr, core_state, events, self.layout, be
            )
            return be.asarray(result, dtype=be.bool_dtype)
        except ImportError:
            return be.zeros((self.layout.B,), dtype=be.bool_dtype)

    def _eval_impl_source(
        self,
        source: TerminationSourceImpl,
        core_state: CoreState,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """Evaluate implementation source."""
        impl_fn = REGISTRY.get_termination_impl(source.impl_ref)
        result = impl_fn(core_state, events, be, self.layout, B, source.params)
        return be.asarray(result, dtype=be.bool_dtype)

    # ------------------------------------------------------------------
    # Time limit checking
    # ------------------------------------------------------------------

    def _check_time_limit(
        self,
        t: int,
        be: Any,
        B: int,
    ) -> Array:
        """
        Check if time limit reached.

        Returns:
            (B,) bool array, True where time limit reached
        """
        max_steps = self.episode_config.max_steps

        if max_steps <= 0:
            return be.zeros((B,), dtype=be.bool_dtype)

        if t >= max_steps:
            return be.ones((B,), dtype=be.bool_dtype)
        else:
            return be.zeros((B,), dtype=be.bool_dtype)
