
# envkit/core/extractors.py
"""
Sensor extraction utilities for building observations from CoreState.

These functions help construct agent observations from raw state fields.
They are optional convenience utilities - users can build observations
however they prefer.

Usage:
    >>> from envkit.core.extractors import extract_sensor
    >>>
    >>> obs = extract_sensor(layout, core_state, sensor_spec, eval_expr)
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from envkit.core.compiled_layout import CompiledLayout, CoreState, Array
from envkit.ir.schema import (
    SensorSpec,
    SensorView,
    SensorExpr,
    SensorTopK,
    SensorMapBins,
    SensorImpl,
)

# Type alias for expression evaluator used by expr / topk sensors
ExprEvaluator = Callable[[str, CompiledLayout, CoreState], Array]
ImplResolver = Callable[[str], Callable[[CompiledLayout, CoreState, dict], Array]]


# ----------------------------------------------------------------------
# View sensor - flatten and concatenate fields
# ----------------------------------------------------------------------

def extract_view(
    layout: CompiledLayout,
    core_state: CoreState,
    field_ids: List[str],
    normalize: bool = False,
) -> Array:
    """
    Concatenate and flatten fields into observation vector.

    Returns:
        (B, F_total) float32 array where F_total = sum of all field dimensions
    """
    be = layout.backend

    if not field_ids:
        # Empty observation
        return be.zeros((layout.B, 0), dtype=be.float_dtype)

    # Collect and flatten each field
    flats = []
    for field_id in field_ids:
        if field_id not in layout.field_index:
            raise ValueError(f"Unknown field: '{field_id}'")

        idx = layout.field_index[field_id]
        arr = core_state[idx]  # (B, *shape)

        # Flatten all non-batch dimensions
        arr_flat = be.reshape(arr, (layout.B, -1))  # (B, n)
        flats.append(arr_flat)

    # Concatenate along feature dimension
    if flats:
        result = be.concat(flats, axis=-1)  # (B, F_total)
    else:
        result = be.zeros((layout.B, 0), dtype=be.float_dtype)

    if normalize:
        result = _normalize(result, be)

    return result


# ----------------------------------------------------------------------
# Top-K sensor - nearest neighbor features (IR-aligned)
# ----------------------------------------------------------------------

def extract_topk(
    layout: CompiledLayout,
    core_state: CoreState,
    pos_field: str,
    extra_fields: List[str],
    k: int,
    center_expr: str,
    evaluator: ExprEvaluator,
    mask_empty: bool = True,
    normalize: bool = True,
) -> Array:
    """
    IR-aligned top-k extractor.

    Matches SensorTopK:
      - pos_field:    entity positions field id
      - extra_fields: additional per-entity fields
      - k:            number of neighbors
      - center_expr:  expression that yields center positions
      - mask_empty:   if True, empty neighborhoods produce zeros
      - normalize:    if True, nan/inf cleaned

    Output shape:
      (B, K, feat_dim) where feat_dim = 2 + len(extra_fields)
      (x, y) + one scalar per extra field
    """
    be = layout.backend
    xp = be.xp

    # Positions
    if pos_field not in layout.field_index:
        raise ValueError(f"Unknown position field: '{pos_field}'")

    idx_pos = layout.field_index[pos_field]
    pos = core_state[idx_pos]  # (B, N, D_pos)

    if len(pos.shape) < 3:
        raise ValueError(f"Position field '{pos_field}' must be 3D: (B, N, D_pos)")

    B = layout.B
    N = int(pos.shape[1])
    k = min(k, N)
    feat_dim = 2 + len(extra_fields)

    # Center from expression
    center = evaluator(center_expr, core_state, {}, layout, layout.backend)
    center = be.asarray(center, dtype=be.float_dtype)

    # Expect (B, D_pos) or (B, 1, D_pos)
    if center.ndim == 2:
        center = be.expand_dims(center, 1)  # (B, 1, D_pos)
    elif center.ndim == 3:
        if center.shape[1] != 1:
            raise ValueError(
                f"center_expr must produce shape (B, D_pos) or (B, 1, D_pos), "
                f"got {center.shape}"
            )
    else:
        center = be.reshape(center, (B, 1, -1))

    # Distances
    diff = pos - center  # (B, N, D_pos)
    d2 = be.sum(diff * diff, axis=-1)  # (B, N)

    feats_per_env: List[Array] = []

    for b in range(B):
        d2_b = d2[b]  # (N,)
        finite_mask = be.isfinite(d2_b)

        # No valid neighbors
        if not xp.any(finite_mask):
            # Current semantics: mask_empty always yields zeros.
            # If we ever want different behavior when mask_empty=False,
            # this is the place to branch.
            feats_per_env.append(
                be.zeros((k, feat_dim), dtype=be.float_dtype)
            )
            continue

        # Top-k nearest
        idx_b = be.argsort_stable(d2_b)[:k]  # (k,)

        pos_b = pos[b]  # (N, D_pos)
        x_b = pos_b[idx_b, 0]
        y_b = pos_b[idx_b, 1]
        feat_cols = [x_b, y_b]

        # Extra scalar features per entity
        for field_id in extra_fields:
            if field_id not in layout.field_index:
                raise ValueError(f"Unknown extra field: '{field_id}'")

            idx_extra = layout.field_index[field_id]
            extra = core_state[idx_extra]  # (B, ...) or (B, N, ...)
            extra_b = extra[b]

            if extra_b.ndim == 1:
                # Scalar per entity
                col = extra_b[idx_b]
            else:
                # Vector per entity - flatten and take first component
                extra_sel = extra_b[idx_b, ...]  # (k, ...)
                extra_flat = be.reshape(extra_sel, (k, -1))
                col = extra_flat[:, 0]

            feat_cols.append(be.asarray(col, dtype=be.float_dtype))

        # Stack features
        if xp.__name__ == "torch":
            stacked_b = xp.stack(feat_cols, dim=-1)  # (k, feat_dim)
        else:
            stacked_b = xp.stack(feat_cols, axis=-1)  # (k, feat_dim)

        stacked_b = be.asarray(stacked_b, dtype=be.float_dtype)
        if normalize:
            stacked_b = _normalize(stacked_b, be)

        feats_per_env.append(stacked_b)

    # Stack across batch
    if xp.__name__ == "torch":
        result = xp.stack(feats_per_env, dim=0)  # (B, k, feat_dim)
    else:
        result = xp.stack(feats_per_env, axis=0)  # (B, k, feat_dim)

    return be.asarray(result, dtype=be.float_dtype)


# ----------------------------------------------------------------------
# Map bins sensor - spatial binned features
# ----------------------------------------------------------------------

def extract_map_bins(
    layout: CompiledLayout,
    core_state: CoreState,
    fields: List[str],
    normalize: bool = True,
) -> Array:
    """
    Extract map-bin features from spatial fields.

    Returns:
        (B, K, F) float32 array where F = len(fields)
    """
    be = layout.backend
    xp = be.xp

    if not fields:
        return be.zeros((layout.B, 0, 0), dtype=be.float_dtype)

    # Get first field to determine shape
    if fields[0] not in layout.field_index:
        raise ValueError(f"Unknown field: '{fields[0]}'")

    idx0 = layout.field_index[fields[0]]
    first = core_state[idx0]  # (B, K) or (B, K, ...)

    if len(first.shape) < 2:
        raise ValueError("Map-bin fields must be at least 2D: (B, K, ...)")

    B = layout.B
    K = int(first.shape[1])

    # Extract and stack columns
    cols = []
    for field_id in fields:
        if field_id not in layout.field_index:
            raise ValueError(f"Unknown field: '{field_id}'")

        idx = layout.field_index[field_id]
        arr = core_state[idx]  # (B, K) or (B, K, ...)

        if arr.ndim == 2:
            # Already (B, K)
            col = be.asarray(arr, dtype=be.float_dtype)
        else:
            # (B, K, ...) - flatten and take first component
            arr_flat = be.reshape(arr, (B, K, -1))  # (B, K, d)
            col = arr_flat[..., 0]  # (B, K)
            col = be.asarray(col, dtype=be.float_dtype)

        cols.append(col)

    # Stack along feature dimension
    if xp.__name__ == "torch":
        M = xp.stack(cols, dim=-1)  # (B, K, F)
    else:
        M = xp.stack(cols, axis=-1)  # (B, K, F)

    M = be.asarray(M, dtype=be.float_dtype)
    M = _normalize(M, be)

    # Normalize per field (scaling)
    if normalize:
        if xp.__name__ == "torch":
            abs_M = xp.abs(M)  # (B, K, F)
            denom = xp.amax(abs_M, dim=1, keepdim=True)  # (B, 1, F)
            denom = xp.clamp(denom, min=1e-6)
            M = M / denom
        else:
            abs_M = xp.abs(M)
            denom = xp.max(abs_M, axis=1, keepdims=True)  # (B, 1, F)
            denom = xp.maximum(denom, 1e-6)
            M = M / denom

        M = _normalize(M, be)

    return M


# ----------------------------------------------------------------------
# Dispatch from IR sensors
# ----------------------------------------------------------------------

def extract_sensor(
    layout: CompiledLayout,
    core_state: CoreState,
    sensor: SensorSpec,
    eval_expr: ExprEvaluator,
    impl_resolver: Optional[ImplResolver] = None,
) -> Array:
    """
    IR-aligned entry point: build an observation from a SensorSpec.

    Args:
        layout:      compiled layout
        core_state:  current state
        sensor:      SensorSpec (discriminated by .kind)
        eval_expr:   function to evaluate safe expressions → Array
        impl_resolver:
            optional function mapping impl_ref → callable(layout, state, params)
    """
    be = layout.backend

    if isinstance(sensor, SensorView):
        # sensor.from_fields is the attribute name; JSON uses "from"
        return extract_view(
            layout,
            core_state,
            sensor.from_fields,
            normalize=sensor.normalize,
        )

    if isinstance(sensor, SensorExpr):
        # eval_expr must be backend-aware and return either
        # (B, *sensor.shape) or (*sensor.shape,) (constant across envs).
        arr = eval_expr(sensor.expr, layout, core_state)

        arr = _ensure_sensor_shape_from_ir(
            arr,
            per_env_shape=sensor.shape,
            layout=layout,
            backend=be,
            sensor_id=sensor.id,
        )

        if sensor.normalize:
            arr = _normalize(arr, be)
        return arr

    if isinstance(sensor, SensorTopK):
        return extract_topk(
            layout=layout,
            core_state=core_state,
            pos_field=sensor.pos_field,
            extra_fields=sensor.extra_fields,
            k=sensor.k,
            center_expr=sensor.center_expr,
            eval_expr=eval_expr,
            mask_empty=sensor.mask_empty,
            normalize=sensor.normalize,
        )

    if isinstance(sensor, SensorMapBins):
        return extract_map_bins(
            layout,
            core_state,
            sensor.fields,
            normalize=sensor.do_normalize,
        )

    if isinstance(sensor, SensorImpl):
        if impl_resolver is None:
            raise ValueError(
                f"No impl_resolver provided for sensor impl_ref='{sensor.impl_ref}'"
            )
        impl_fn = impl_resolver(sensor.impl_ref)

        # Impl is allowed to return (B, *sensor.shape) or (*sensor.shape,)
        arr = impl_fn(layout, core_state, sensor.params)

        arr = _ensure_sensor_shape_from_ir(
            arr,
            per_env_shape=sensor.shape,
            layout=layout,
            backend=be,
            sensor_id=sensor.id,
        )

        if sensor.normalize:
            arr = _normalize(arr, be)
        return arr

    raise TypeError(f"Unsupported sensor kind: {sensor}")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _normalize(arr: Array, backend: Any, fill: float = 0.0) -> Array:
    """Replace NaN/Inf with fill value."""
    be = backend
    arr = be.asarray(arr, dtype=be.float_dtype)
    arr = be.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


def _ensure_sensor_shape_from_ir(
    arr: Array,
    per_env_shape: list[int],
    layout: CompiledLayout,
    backend: Any,
    sensor_id: str,
) -> Array:
    """
    Ensure a sensor output matches the IR-declared per-env shape.

    IR declares shapes without batch; runtime arrays must be:
      (B, *per_env_shape)

    We accept:
      - (B, *per_env_shape)  -> returned as-is
      - (*per_env_shape,)    -> broadcast over B

    Anything else is an error.
    """
    be = backend
    arr = be.asarray(arr, dtype=be.float_dtype)

    expected = (layout.B, *per_env_shape)
    per_env_tuple = tuple(per_env_shape)

    if arr.shape == expected:
        return arr

    if arr.shape == per_env_tuple:
        # Constant across envs → broadcast to (B, *shape)
        return be.broadcast_to(arr, expected)

    raise ValueError(
        f"Sensor '{sensor_id}' produced shape {arr.shape}, "
        f"but IR declares per-env shape {per_env_tuple} "
        f"(expected either {expected} or {per_env_tuple})."
    )
