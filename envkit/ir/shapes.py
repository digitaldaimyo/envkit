from __future__ import annotations

from typing import Dict, List, Tuple, Any

from .schema import (
    IR,
    StateField,
    ShapeDim,
    ShapeLike,
    SensorView,
    SensorTopK,
    SensorMapBins,
    SensorExpr,
    SensorImpl,
)


class ShapeResolutionError(ValueError):
    """Raised when a shape or sensor shape cannot be resolved."""
    pass


class _ShapeNotReady(Exception):
    """Internal sentinel: a shape depends on unresolved symbols/aliases."""
    pass


class ShapeService:
    """
    Centralized shape resolution and query service.

    Responsibilities:
    - Resolve state field shapes (symbols + ShapeLike aliases) to concrete
      per-env shapes: (d0, d1, ...)
    - Resolve sensor shapes to concrete per-env shapes (no batch dim).
    - Validate multi-agent group-axis bindings against resolved field shapes.

    All shapes are resolved once at construction and cached.
    """

    def __init__(self, ir: IR):
        self.ir = ir
        # Copy symbols so later mutation of IR.symbols does not affect us
        self.symbols: Dict[str, int] = dict(ir.state_schema.symbols or {})

        # Resolve state shapes
        self.state_shapes: Dict[str, Tuple[int, ...]] = self._resolve_state_shapes()

        # Resolve sensor shapes (across ALL groups; ids must be globally unique)
        self.sensor_shapes: Dict[str, Tuple[int, ...]] = self._resolve_sensor_shapes()

        # Validate group-axis bindings using resolved shapes
        self._validate_group_axes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def field_shape(self, field_id: str) -> Tuple[int, ...]:
        """Return resolved per-env shape for a state field (no batch dim)."""
        if field_id not in self.state_shapes:
            raise KeyError(f"Unknown field: '{field_id}'")
        return self.state_shapes[field_id]

    def sensor_shape(self, sensor_id: str) -> Tuple[int, ...]:
        """Return resolved per-env shape for a sensor (no batch dim)."""
        if sensor_id not in self.sensor_shapes:
            raise KeyError(f"Unknown sensor id: '{sensor_id}'")
        return self.sensor_shapes[sensor_id]

    # ------------------------------------------------------------------
    # State shape resolution
    # ------------------------------------------------------------------

    def _resolve_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        resolved: Dict[str, Tuple[int, ...]] = {}

        # After bundle expansion, all entries are StateField
        pending: Dict[str, StateField] = {
            str(f.id): f for f in self.ir.state_schema.fields
        }

        if not pending:
            return resolved

        max_iterations = len(pending) + 4
        iteration = 0

        while pending and iteration < max_iterations:
            iteration += 1
            made_progress = False

            for field_id in list(pending.keys()):
                field = pending[field_id]
                try:
                    shape = self._resolve_shape_dims(field.shape, resolved)
                    resolved[field_id] = shape
                    del pending[field_id]
                    made_progress = True
                except _ShapeNotReady:
                    continue

            if not made_progress:
                break

        if pending:
            unresolved = ", ".join(sorted(pending.keys()))
            raise ShapeResolutionError(
                f"Could not resolve shapes for fields: {unresolved}"
            )

        return resolved

    def _resolve_shape_dims(
        self,
        dims: List[ShapeDim],
        resolved: Dict[str, Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        result: List[int] = []

        for dim in dims:
            # Literal int
            if isinstance(dim, int):
                if dim < 0:
                    raise ValueError(f"Shape dimension must be >= 0, got {dim}")
                result.append(dim)

            # Symbol reference (backed by SymbolId; here we just see str)
            elif isinstance(dim, str):
                if dim not in self.symbols:
                    raise ShapeResolutionError(f"Unknown shape symbol: '{dim}'")
                val = self.symbols[dim]
                if not isinstance(val, int) or val <= 0:
                    raise ShapeResolutionError(
                        f"Symbol '{dim}' must map to a positive int, got {val!r}"
                    )
                result.append(int(val))

            # ShapeLike alias
            elif isinstance(dim, ShapeLike):
                like_id = str(dim.like)
                if like_id not in resolved:
                    raise _ShapeNotReady(f"Shape for '{like_id}' not resolved yet")

                base_shape = resolved[like_id]
                if dim.dims is None:
                    result.extend(int(d) for d in base_shape)
                else:
                    for idx in dim.dims:
                        if idx < 0 or idx >= len(base_shape):
                            raise ShapeResolutionError(
                                f"ShapeLike dims index {idx} out of range for "
                                f"field '{like_id}' with shape {base_shape}"
                            )
                        result.append(int(base_shape[idx]))
            else:
                raise TypeError(f"Unsupported ShapeDim type: {type(dim)!r}")

        return tuple(result)

    # ------------------------------------------------------------------
    # Sensor shapes
    # ------------------------------------------------------------------

    def _resolve_sensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Compute output shapes for all sensors across ALL agent groups.

        Notes:
        - Sensor ids must be globally unique across groups.
        - Shapes are per-env (no batch dim).
        """
        sensor_shapes: Dict[str, Tuple[int, ...]] = {}

        if not getattr(self.ir, "agents", None) or not self.ir.agents.groups:
            return sensor_shapes

        for group in self.ir.agents.groups:
            for sensor in group.sensors:
                sid = str(sensor.id)
                if sid in sensor_shapes:
                    raise ShapeResolutionError(
                        f"Duplicate sensor id '{sid}' across groups; "
                        "sensor ids must be globally unique."
                    )

                if isinstance(sensor, SensorView):
                    shape = self._shape_sensor_view(sensor)
                elif isinstance(sensor, SensorTopK):
                    shape = self._shape_sensor_topk(sensor)
                elif isinstance(sensor, SensorMapBins):
                    shape = self._shape_sensor_map_bins(sensor)
                elif isinstance(sensor, SensorExpr):
                    shape = self._shape_sensor_expr(sensor)
                elif isinstance(sensor, SensorImpl):
                    shape = self._shape_sensor_impl(sensor)
                else:
                    raise NotImplementedError(
                        f"Sensor kind '{getattr(sensor, 'kind', '?')}' "
                        f"not supported by ShapeService"
                    )

                sensor_shapes[sid] = shape

        return sensor_shapes

    def _shape_sensor_view(self, sensor: SensorView) -> Tuple[int, ...]:
        """
        SensorView: concatenate flattened fields into a 1D vector.
        Output: (F_total,)
        """
        field_ids = list(sensor.from_fields or [])
        if not field_ids:
            return (0,)

        total = 0
        for fid in field_ids:
            key = str(fid)
            if key not in self.state_shapes:
                raise ShapeResolutionError(
                    f"Sensor '{sensor.id}' references unknown field '{key}'"
                )
            fshape = self.state_shapes[key]
            total += self._flat_size(fshape)

        return (int(total),)

    def _shape_sensor_topk(self, sensor: SensorTopK) -> Tuple[int, ...]:
        """
        SensorTopK: (K, feat_dim) where feat_dim = 2 + len(extra_fields).
        """
        pos_field = str(sensor.pos_field)
        extra_fields = [str(f) for f in sensor.extra_fields]
        K = int(sensor.k)

        if pos_field not in self.state_shapes:
            raise ShapeResolutionError(
                f"TopK sensor '{sensor.id}' pos field '{pos_field}' has no shape"
            )
        pos_shape = self.state_shapes[pos_field]
        if len(pos_shape) < 2:
            raise ShapeResolutionError(
                f"TopK sensor '{sensor.id}' expects pos_field '{pos_field}' "
                f"to have at least 2 dims (N, D_pos); got {pos_shape}"
            )
        if K <= 0:
            raise ShapeResolutionError(
                f"TopK sensor '{sensor.id}' must have k > 0, got {K}"
            )

        feat_dim = 2 + len(extra_fields)
        return (K, int(feat_dim))

    def _shape_sensor_map_bins(self, sensor: SensorMapBins) -> Tuple[int, ...]:
        """
        SensorMapBins: (K, F) where F = len(fields).
        """
        fields = [str(f) for f in sensor.fields or []]
        if not fields:
            return (0, 0)

        K: int | None = None
        for fid in fields:
            if fid not in self.state_shapes:
                raise ShapeResolutionError(
                    f"MapBins sensor '{sensor.id}' references unknown field '{fid}'"
                )
            fshape = self.state_shapes[fid]
            if len(fshape) == 0:
                raise ShapeResolutionError(
                    f"MapBins sensor '{sensor.id}' expects field '{fid}' "
                    f"to be at least 1D; got shape {fshape}"
                )
            this_K = int(fshape[0])
            if K is None:
                K = this_K
            elif K != this_K:
                raise ShapeResolutionError(
                    f"MapBins sensor '{sensor.id}' expects all fields to share "
                    f"the same first dim, but '{fid}' has {this_K} vs {K}"
                )

        if K is None:
            K = 0

        F = len(fields)
        return (int(K), int(F))

    def _shape_sensor_expr(self, sensor: SensorExpr) -> Tuple[int, ...]:
        """
        SensorExpr must provide an explicit shape.
        """
        if not sensor.shape:
            raise ShapeResolutionError(
                f"Expr sensor '{sensor.id}' must define explicit shape"
            )
        return tuple(int(d) for d in sensor.shape)

    def _shape_sensor_impl(self, sensor: SensorImpl) -> Tuple[int, ...]:
        """
        SensorImpl must provide an explicit shape.
        """
        if not sensor.shape:
            raise ShapeResolutionError(
                f"Impl sensor '{sensor.id}' must define explicit shape"
            )
        return tuple(int(d) for d in sensor.shape)

    # ------------------------------------------------------------------
    # Group axis validation
    # ------------------------------------------------------------------

    def _validate_group_axes(self) -> None:
        """
        Validate bind_axes for multi-agent groups:

        For each group:
          - Each bound field must exist.
          - axis must be valid for that field's shape.
          - field.shape[axis] must equal group.count.
        """
        if not getattr(self.ir, "agents", None):
            return

        if not getattr(self.ir.agents, "groups", None):
            return

        for group in self.ir.agents.groups:
            count = int(getattr(group, "count", 1))
            if count <= 1:
                # Single-agent group: no special constraint
                continue

            for binding in getattr(group, "bind_axes", []) or []:
                field_id = str(binding.field)
                axis = int(binding.axis)

                if field_id not in self.state_shapes:
                    raise ShapeResolutionError(
                        f"Group '{group.id}' bind_axes references unknown field "
                        f"'{field_id}'"
                    )

                shape = self.state_shapes[field_id]
                if axis < 0 or axis >= len(shape):
                    raise ShapeResolutionError(
                        f"Group '{group.id}' bind_axes for field '{field_id}' "
                        f"uses axis {axis}, but field shape is {shape}"
                    )

                dim = int(shape[axis])
                if dim != count:
                    raise ShapeResolutionError(
                        f"Group '{group.id}' has count={count}, but "
                        f"field '{field_id}' has dim[{axis}]={dim}"
                    )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_size(shape: Tuple[int, ...]) -> int:
        size = 1
        for d in shape:
            size *= int(d)
        return int(size)
