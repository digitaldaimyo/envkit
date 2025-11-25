
from __future__ import annotations

from typing import Dict, Any, List
from enum import Enum

from envkit.core.compiled_layout import CompiledLayout, CoreState, Events, Array
from envkit.core.registry import REGISTRY
from envkit.ir.schema import (
    RewardChannelSpec,
    RewardSourceField,
    RewardSourceEvent,
    RewardSourceExpr,
    RewardSourceImpl,
    BuiltinAggregatorSpec,
    ImplAggregatorSpec,
    BuiltinAggStrategy,
)


# ----------------------------------------------------------------------
# Reward Engine
# ----------------------------------------------------------------------

class RewardEngine:
    """
    Reward computation engine (v2.1 Section 8).

    Evaluates reward channels and aggregates them per agent group:
    1. Evaluate all channels to (B, N_group) arrays
    2. Apply weights
    3. Aggregate via builtin or custom aggregators
    4. Distribute to target groups
    """

    def __init__(self, layout: CompiledLayout):
        self.layout = layout
        self.task = layout.get_active_task()
        self.reward_config = self.task.reward

        # Build channel lookup
        self.channels = {str(ch.id): ch for ch in self.reward_config.channels}

        # Build aggregator lookup
        self.aggregators = {str(agg.id): agg for agg in self.reward_config.aggregators}

    def compute(
        self,
        core_state: CoreState,
        events: Events,
        layout: CompiledLayout,
        B: int,
    ) -> Dict[str, Array]:
        """
        Compute rewards for all agent groups.
        """
        be = layout.backend

        # Step 1: Evaluate all channels
        channel_values: Dict[str, Array] = {}
        for channel_id, channel in self.channels.items():
            try:
                value = self._evaluate_channel(channel, core_state, events, be, B)
                channel_values[channel_id] = value  # (B, N_group)
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating reward channel '{channel_id}': {e}"
                ) from e

        # Step 2: Aggregate per group
        group_rewards: Dict[str, Array] = {}

        for agg in self.reward_config.aggregators:
            # Get channels for this aggregator
            agg_channels = {
                str(ch_id): channel_values[str(ch_id)]
                for ch_id in agg.channels
                if str(ch_id) in channel_values
            }

            if not agg_channels:
                continue

            # Aggregate
            try:
                if isinstance(agg, BuiltinAggregatorSpec):
                    agg_value = self._aggregate_builtin(
                        agg.strategy, agg_channels, be, B
                    )
                elif isinstance(agg, ImplAggregatorSpec):
                    # Custom aggregator; use first target group to get N_group
                    if not agg.output.groups:
                        continue
                    group_id = str(agg.output.groups[0])
                    if group_id not in layout.groups:
                        raise ValueError(
                            f"Aggregator '{agg.id}' references unknown group '{group_id}'"
                        )
                    group = layout.groups[group_id]
                    max_agents = group.count

                    impl_fn = REGISTRY.get_aggregator(agg.impl_ref)
                    agg_value = impl_fn(
                        agg_channels, be, B, max_agents, agg.params
                    )
                else:
                    raise ValueError(f"Unknown aggregator type: {type(agg)}")
            except Exception as e:
                raise RuntimeError(
                    f"Error executing aggregator '{agg.id}': {e}"
                ) from e

            # Distribute to target groups
            target_groups = (
                [str(g) for g in agg.output.groups]
                if hasattr(agg, "output")
                else []
            )

            for group_id in target_groups:
                if group_id not in layout.groups:
                    continue

                group = layout.groups[group_id]

                # Handle different output targets
                if hasattr(agg, "output") and agg.output.target == "global":
                    # Broadcast scalar to (B, N_group)
                    if agg_value.shape == (B,):
                        agg_value_group = be.broadcast_to(
                            be.expand_dims(agg_value, 1),
                            (B, group.count),
                        )
                    else:
                        agg_value_group = agg_value
                else:
                    # For 'per_group' / 'per_agent', we trust the aggregator
                    agg_value_group = agg_value

                # Accumulate to group
                if group_id not in group_rewards:
                    group_rewards[group_id] = agg_value_group
                else:
                    group_rewards[group_id] = group_rewards[group_id] + agg_value_group

        # Step 3: Fill in missing groups with zeros
        for group_id, group in layout.groups.items():
            if group_id not in group_rewards:
                group_rewards[group_id] = be.zeros(
                    (B, group.count),
                    dtype=be.float_dtype,
                )

        return group_rewards

    # ------------------------------------------------------------------
    # Channel evaluation
    # ------------------------------------------------------------------

    def _evaluate_channel(
        self,
        channel: RewardChannelSpec,
        core_state: CoreState,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """
        Evaluate a reward channel to (B, N_group).
        """
        group_id = str(channel.group)
        if group_id not in self.layout.groups:
            raise ValueError(
                f"Reward channel '{channel.id}' references unknown group '{group_id}'"
            )

        group = self.layout.groups[group_id]
        N_group = group.count

        # Evaluate source
        if isinstance(channel.source, RewardSourceField):
            value = self._eval_field_source(channel.source, core_state, be)
        elif isinstance(channel.source, RewardSourceEvent):
            value = self._eval_event_source(channel.source, events, be, B)
        elif isinstance(channel.source, RewardSourceExpr):
            value = self._eval_expr_source(channel.source, core_state, events, be)
        elif isinstance(channel.source, RewardSourceImpl):
            value = self._eval_impl_source(channel.source, core_state, events, be, B)
        else:
            raise ValueError(f"Unknown reward source type: {type(channel.source)}")

        # Normalize to (B, N_group) based on target
        value = self._normalize_to_group(value, channel.target, be, B, N_group)

        # Apply weight
        value = value * channel.weight

        return value

    def _eval_field_source(
        self,
        source: RewardSourceField,
        core_state: CoreState,
        be: Any,
    ) -> Array:
        """Evaluate field source."""
        field_id = str(source.field)
        if field_id not in self.layout.field_index:
            raise KeyError(f"Unknown state field in reward source: '{field_id}'")

        idx = self.layout.field_index[field_id]
        arr = core_state[idx]  # (B, *shape)

        # Reduce to (B,) by mean over non-batch dims
        if len(arr.shape) > 1:
            arr = be.mean(be.reshape(arr, (arr.shape[0], -1)), axis=1)

        return arr  # (B,)

    def _eval_event_source(
        self,
        source: RewardSourceEvent,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """
        Evaluate event source from tensor events.

        Semantics:
          - events[event_id] is a dict: field_name -> Array
          - If source.field is provided, we take that field.
          - If source.field is None and the channel has exactly one field,
            we use that field.
          - Arrays must have shape (B, ...) (first dim = batch).
          - Output is typically (B,) but can be (B, K) if the event field
            already encodes per-agent / per-slot signals.
        """
        event_id = str(source.event)
        field_name = str(source.field) if source.field else None

        if event_id not in events:
            return be.zeros((B,), dtype=be.float_dtype)

        chan = events[event_id]
        if not isinstance(chan, dict) or not chan:
            return be.zeros((B,), dtype=be.float_dtype)

        if field_name is None:
            if len(chan) == 1:
                arr = next(iter(chan.values()))
            else:
                raise ValueError(
                    f"RewardSourceEvent for event '{event_id}' must specify 'field' "
                    f"when the event channel has multiple fields"
                )
        else:
            if field_name not in chan:
                # Missing field → zero reward
                return be.zeros((B,), dtype=be.float_dtype)
            arr = chan[field_name]

        arr = be.asarray(arr, dtype=be.float_dtype)

        # Basic sanity: first dimension must match batch size
        if arr.shape[0] != B:
            raise ValueError(
                f"Event '{event_id}' field '{field_name}' produced shape {arr.shape}, "
                f"but batch size is B={B}"
            )

        # Allow (B,), (B,1), (B, N_group, ...); downstream normalization will handle it
        if arr.ndim == 1:
            return arr  # (B,)
        if arr.ndim == 2:
            # (B, 1) or (B, N_group) both fine, return as-is
            return arr

        # If more dims, flatten everything after batch into a single feature dim
        return be.reshape(arr, (B, -1))

    def _eval_expr_source(
        self,
        source: RewardSourceExpr,
        core_state: CoreState,
        events: Events,
        be: Any,
    ) -> Array:
        """Evaluate expression source."""
        try:
            from envkit.engines.expr_eval import evaluate_expression

            return evaluate_expression(
                source.expr, core_state, events, self.layout, be
            )
        except ImportError:
            # Fallback: return zeros
            return be.zeros((self.layout.B,), dtype=be.float_dtype)

    def _eval_impl_source(
        self,
        source: RewardSourceImpl,
        core_state: CoreState,
        events: Events,
        be: Any,
        B: int,
    ) -> Array:
        """Evaluate implementation source."""
        impl_fn = REGISTRY.get_reward_impl(source.impl_ref)
        return impl_fn(core_state, events, be, self.layout, B, source.params)

    def _normalize_to_group(
        self,
        value: Array,
        target: Any,     # RewardTarget
        be: Any,
        B: int,
        N_group: int,
    ) -> Array:
        """
        Normalize value to (B, N_group) according to target.mode.

        Modes:
          - per_env / per_group:
              input should be per-env scalar → (B,) or (B, 1);
              we broadcast to (B, N_group).
          - per_agent:
              input must already be (B, N_group).
        """
        mode = getattr(target, "mode", None)
        if hasattr(mode, "value"):
            mode = mode.value
        if isinstance(mode, str):
            mode = mode.lower()

        def _broadcast_per_env(v: Array) -> Array:
            if v.shape == (B,):
                v = be.expand_dims(v, 1)  # (B, 1)
            if v.shape == (B, 1):
                return be.broadcast_to(v, (B, N_group))
            raise ValueError(
                f"Expected per-env value with shape (B,) or (B,1), "
                f"got {v.shape} (B={B}, N_group={N_group}, mode={mode})"
            )

        if mode == "per_agent":
            if value.shape != (B, N_group):
                raise ValueError(
                    f"RewardTarget(mode='per_agent') expects value shape (B, N_group) "
                    f"= ({B}, {N_group}), got {value.shape}"
                )
            return value

        if mode in ("per_env", "per_group", None):
            if value.shape == (B, N_group):
                return value
            return _broadcast_per_env(value)

        raise ValueError(
            f"Unknown reward target mode: {mode!r} "
            f"for value shape {value.shape}, B={B}, N_group={N_group}"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_builtin(
        self,
        strategy: BuiltinAggStrategy,
        channels: Dict[str, Array],
        be: Any,
        B: int,
    ) -> Array:
        """
        Aggregate channels using builtin strategy.
        """
        if not channels:
            raise ValueError("No channels to aggregate")

        channel_list = list(channels.values())

        if strategy == BuiltinAggStrategy.SUM:
            result = channel_list[0]
            for ch in channel_list[1:]:
                result = result + ch
            return result

        elif strategy == BuiltinAggStrategy.MEAN:
            result = channel_list[0]
            for ch in channel_list[1:]:
                result = result + ch
            return result / len(channel_list)

        elif strategy == BuiltinAggStrategy.DOT_WEIGHT:
            # Weights already baked into channels via channel.weight
            result = channel_list[0]
            for ch in channel_list[1:]:
                result = result + ch
            return result

        else:
            raise ValueError(f"Unknown builtin strategy: {strategy}")
