
# envkit/engines/expr_eval.py
from __future__ import annotations

import ast
import operator as op
from typing import Dict, Any, Callable

from envkit.core.compiled_layout import CompiledLayout, CoreState, Events, Array


# ----------------------------------------------------------------------
# Allowed operators (work on all backend arrays)
# ----------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARY = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_ALLOWED_CMPS = {
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
}


# ----------------------------------------------------------------------
# Expression evaluator
# ----------------------------------------------------------------------

def evaluate_expression(
    expr: str,
    core_state: CoreState,
    events: Events,
    layout: CompiledLayout,
    backend: Any,
) -> Array:
    """
    Evaluate a safe mathematical expression over state and events.

    Supported operations:
    - Arithmetic: +, -, *, /, %, **
    - Comparisons: <, <=, >, >=, ==, !=
    - Boolean: and, or
    - Functions: relu, clip, sum, mean, norm, abs, sqrt
    - Indexing: field[0], field[:, 1], field[..., :2]

    Variables:
    - State fields by name: red_pos, ball_pos, etc.
    - Event counts per env: event_<id> has shape (B,), counting
      active entries for that event in each environment.
    """
    be = backend

    # Build environment: state fields + event counts
    env = _build_environment(core_state, events, layout, be)

    # Build function table
    funcs = _build_functions(be)

    # Parse and evaluate
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast(tree.body, env, funcs, be.xp)
    except Exception as e:
        raise ValueError(
            f"Error evaluating expression '{expr}': {e}\n"
            f"Available variables: {', '.join(sorted(env.keys()))}\n"
            f"Available functions: {', '.join(sorted(funcs.keys()))}"
        ) from e

    # Convert to backend array and ensure finite
    arr = be.asarray(result, dtype=be.float_dtype)
    arr = _normalize(arr, be)

    return arr


# ----------------------------------------------------------------------
# Environment building
# ----------------------------------------------------------------------

def _build_environment(
    core_state: CoreState,
    events: Events,
    layout: CompiledLayout,
    backend: Any,
) -> Dict[str, Any]:
    """
    Build variable environment for expression evaluation.

    Returns dict with:
    - Field names -> backend arrays (B, *shape)
    - Event names -> per-env counts (B,) int:
        event_<id>[b] = number of active entries for that event in env b.
    """
    be = backend
    xp = be.xp
    env: Dict[str, Any] = {}

    # Add all state fields
    for field_id in layout.field_ids:
        idx = layout.field_index[field_id]
        env[field_id] = core_state[idx]

    # Add event counts per env
    for event_id, chan in events.items():
        if not isinstance(chan, dict) or not chan:
            counts = be.zeros((layout.B,), dtype=be.int_dtype)
            env[f"event_{event_id}"] = counts
            continue

        counts = be.zeros((layout.B,), dtype=be.int_dtype)

        for arr in chan.values():
            a = be.asarray(arr)
            if a.shape[0] != layout.B:
                raise ValueError(
                    f"Event '{event_id}' field has shape {a.shape}, "
                    f"but batch size is B={layout.B}"
                )

            if xp.__name__ == "torch":
                a_bool = a != 0
            else:
                a_bool = a != 0

            if a_bool.ndim > 1:
                axes = tuple(range(1, a_bool.ndim))
                if hasattr(xp, "any"):
                    mask = xp.any(a_bool, axis=axes)
                else:
                    mask = be.sum(a_bool, axis=axes) > 0
            else:
                mask = a_bool

            counts = counts + mask.astype(be.int_dtype)

        env[f"event_{event_id}"] = counts

    return env

def _build_functions(backend: Any) -> Dict[str, Callable[..., Any]]:
    """Build backend-aware function table."""
    be = backend
    xp = be.xp

    def _relu(x):
        """ReLU: max(0, x)"""
        return be.clip(x, 0.0, None)

    def _clip(x, lo, hi):
        """Clip values to [lo, hi]"""
        return be.clip(x, lo, hi)

    def _abs(x):
        """Absolute value"""
        return be.abs(x)

    def _sqrt(x):
        """Square root"""
        return be.sqrt(x)

    def _sum(x, axis=None):
        """Sum along axis"""
        return be.sum(x, axis=axis)

    def _mean(x, axis=None):
        """Mean along axis"""
        return be.mean(x, axis=axis)

    def _max(x, axis=None):
        """Max along axis"""
        return be.max(x, axis=axis)

    def _min(x, axis=None):
        """Min along axis"""
        return be.min(x, axis=axis)

    def _norm(x, ord=None, axis=None):
        """Vector/matrix norm"""
        return be.norm(x, ord=ord, axis=axis)

    return {
        "relu": _relu,
        "clip": _clip,
        "abs": _abs,
        "sqrt": _sqrt,
        "sum": _sum,
        "mean": _mean,
        "max": _max,
        "min": _min,
        "norm": _norm,
    }

# ----------------------------------------------------------------------
# AST evaluation
# ----------------------------------------------------------------------

def _eval_ast(
    node: ast.AST,
    env: Dict[str, Any],
    funcs: Dict[str, Callable],
    xp: Any,
) -> Any:
    """
    Evaluate a restricted AST node in backend space.
    """
    # Literals
    if isinstance(node, ast.Constant):
        return node.value

    # Names (variables)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unknown variable: '{node.id}'")
        return env[node.id]

    # Unary operations (+x, -x)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY:
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
        return _ALLOWED_UNARY[type(node.op)](
            _eval_ast(node.operand, env, funcs, xp)
        )

    # Binary operations (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError(f"Binary operator not allowed: {type(node.op).__name__}")
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_ast(node.left, env, funcs, xp),
            _eval_ast(node.right, env, funcs, xp),
        )

    # Boolean operations (x and y, x or y)
    if isinstance(node, ast.BoolOp):
        vals = [_eval_ast(v, env, funcs, xp) for v in node.values]
        result = vals[0]
        for v in vals[1:]:
            if isinstance(node.op, ast.And):
                result = xp.logical_and(result, v)
            elif isinstance(node.op, ast.Or):
                result = xp.logical_or(result, v)
            else:
                raise ValueError(
                    f"Boolean operator not allowed: {type(node.op).__name__}"
                )
        return result

    # Comparisons (x < y, x == y, etc.)
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, env, funcs, xp)
        result = xp.ones_like(left, dtype=bool)
        for op_node, comparator in zip(node.ops, node.comparators):
            if type(op_node) not in _ALLOWED_CMPS:
                raise ValueError(
                    f"Comparison operator not allowed: {type(op_node).__name__}"
                )
            right = _eval_ast(comparator, env, funcs, xp)
            cmp_res = _ALLOWED_CMPS[type(op_node)](left, right)
            result = xp.logical_and(result, cmp_res)
            left = right
        return result

    # Function calls (sum(x), norm(x, axis=-1))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only simple function calls allowed (no methods or attributes)"
            )

        fname = node.func.id
        if fname not in funcs:
            raise ValueError(f"Function not allowed: '{fname}'")

        # Evaluate arguments
        args = [_eval_ast(a, env, funcs, xp) for a in node.args]
        kwargs = {
            kw.arg: _eval_ast(kw.value, env, funcs, xp)
            for kw in node.keywords
        }

        return funcs[fname](*args, **kwargs)

    # Slicing (x[1:5], x[:, 0])
    if isinstance(node, ast.Slice):
        lower = _eval_ast(node.lower, env, funcs, xp) if node.lower else None
        upper = _eval_ast(node.upper, env, funcs, xp) if node.upper else None
        step = _eval_ast(node.step, env, funcs, xp) if node.step else None
        return slice(lower, upper, step)

    # Tuple (for multi-dimensional indexing)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(elt, env, funcs, xp) for elt in node.elts)

    # Indexing (x[0], x[:, 1], x[..., :2])
    if isinstance(node, ast.Subscript):
        base = _eval_ast(node.value, env, funcs, xp)
        idx = _eval_ast(node.slice, env, funcs, xp)
        return base[idx]

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _normalize(arr: Any, backend: Any, fill: float = 0.0) -> Array:
    """
    Replace NaN/Inf with fill value and ensure float32.
    """
    be = backend
    arr = be.asarray(arr, dtype=be.float_dtype)
    arr = be.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


# ----------------------------------------------------------------------
# Testing utilities
# ----------------------------------------------------------------------

def validate_expression(expr: str) -> bool:
    """
    Check if an expression is syntactically valid (without evaluating).
    """
    try:
        ast.parse(expr, mode="eval")
        return True
    except SyntaxError:
        return False


def get_expression_variables(expr: str) -> set[str]:
    """
    Extract all variable names used in an expression.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()

    variables = set()

    def _visit(node):
        if isinstance(node, ast.Name):
            variables.add(node.id)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree.body)
    return variables
