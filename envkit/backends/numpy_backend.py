# envkit/backends/numpy_backend.py

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Any

import numpy as np

from .base import Backend


class NumpyBackend(Backend):
    xp = np
    float_dtype = np.float32
    int_dtype = np.int32
    bool_dtype = np.bool_

    # ------------------------------------------------------------------
    # creation / conversion
    # ------------------------------------------------------------------
    def asarray(self, x, dtype: Optional[Any] = None):
        return np.asarray(x, dtype=dtype)

    def set_requires_grad(self, x, requires_grad: bool):
        # No autograd flags for NumPy
        return np.asarray(x)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return np.zeros(shape, dtype=dtype or self.float_dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return np.ones(shape, dtype=dtype or self.float_dtype)

    def zeros_like(self, x):
        return np.zeros_like(np.asarray(x))

    def ones_like(self, x):
        return np.ones_like(np.asarray(x))

    def copy(self, x):
        return np.array(np.asarray(x), copy=True)

    # ------------------------------------------------------------------
    # shape / indexing helpers
    # ------------------------------------------------------------------
    def reshape(self, x, shape: Tuple[int, ...]):
        return np.reshape(np.asarray(x), shape)

    def squeeze(self, x, axis: Optional[int] = None):
        return np.squeeze(np.asarray(x), axis=axis)

    def expand_dims(self, x, axis: int):
        return np.expand_dims(np.asarray(x), axis=axis)

    def concat(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return np.zeros((0,), dtype=self.float_dtype)
        return np.concatenate([np.asarray(a) for a in arrays], axis=axis)

    def stack(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return np.zeros((0,), dtype=self.float_dtype)
        return np.stack([np.asarray(a) for a in arrays], axis=axis)

    def broadcast_to(self, x, shape: Tuple[int, ...]):
        return np.broadcast_to(np.asarray(x), shape)

    # ------------------------------------------------------------------
    # math
    # ------------------------------------------------------------------
    def sum(self, x, axis=None, keepdims: bool = False):
        return np.sum(np.asarray(x), axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims: bool = False):
        return np.mean(np.asarray(x), axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims: bool = False):
        return np.max(np.asarray(x), axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims: bool = False):
        return np.min(np.asarray(x), axis=axis, keepdims=keepdims)

    def norm(self, x, ord=None, axis=None, keepdims: bool = False):
        return np.linalg.norm(np.asarray(x), ord=ord, axis=axis, keepdims=keepdims)

    def clip(self, x, lo, hi):
        return np.clip(np.asarray(x), lo, hi)

    def nan_to_num(self, x, nan=0.0, posinf=0.0, neginf=0.0):
        return np.nan_to_num(np.asarray(x), nan=nan, posinf=posinf, neginf=neginf)

    def maximum(self, x, y):
        return np.maximum(np.asarray(x), np.asarray(y))

    def minimum(self, x, y):
        return np.minimum(np.asarray(x), np.asarray(y))

    def abs(self, x):
        return np.abs(np.asarray(x))

    def sqrt(self, x):
        return np.sqrt(np.asarray(x))

    # ------------------------------------------------------------------
    # comparisons / logic
    # ------------------------------------------------------------------
    def where(self, cond, a, b):
        return np.where(np.asarray(cond), np.asarray(a), np.asarray(b))

    def logical_and(self, a, b):
        return np.logical_and(np.asarray(a), np.asarray(b))

    def logical_or(self, a, b):
        return np.logical_or(np.asarray(a), np.asarray(b))

    def logical_not(self, a):
        return np.logical_not(np.asarray(a))

    def isfinite(self, x):
        return np.isfinite(np.asarray(x))

    def isinf(self, x):
        return np.isinf(np.asarray(x))

    def isnan(self, x):
        return np.isnan(np.asarray(x))

    # ------------------------------------------------------------------
    # reductions / indexing
    # ------------------------------------------------------------------
    def any(self, x, axis=None, keepdims: bool = False):
        return np.any(np.asarray(x), axis=axis, keepdims=keepdims)

    def all(self, x, axis=None, keepdims: bool = False):
        return np.all(np.asarray(x), axis=axis, keepdims=keepdims)

    def argsort(self, x, axis: int = -1):
        return np.argsort(np.asarray(x), axis=axis, kind="stable")

    def argmax(self, x, axis=None):
        result = np.argmax(np.asarray(x), axis=axis)
        return int(result) if axis is None else result

    def argmin(self, x, axis=None):
        result = np.argmin(np.asarray(x), axis=axis)
        return int(result) if axis is None else result

    def nonzero(self, x):
        return np.nonzero(np.asarray(x))

    # ------------------------------------------------------------------
    # rng (seed-based)
    # ------------------------------------------------------------------
    def random(self, seed: int, size):
        """Sample uniform[0, 1) with a stateless integer seed."""
        rng = np.random.default_rng(int(seed))
        return rng.random(size).astype(self.float_dtype)

    def choice(self, seed: int, n, p=None):
        rng = np.random.default_rng(int(seed))
        return rng.choice(int(n), p=p)

    def randint(self, seed: int, low: int, high: int, size):
        rng = np.random.default_rng(int(seed))
        return rng.integers(int(low), int(high), size=size, dtype=np.int64)

    def normal(self, seed: int, mean: float, std: float, size):
        rng = np.random.default_rng(int(seed))
        return rng.normal(loc=mean, scale=std, size=size).astype(self.float_dtype)

    # ------------------------------------------------------------------
    # host interop
    # ------------------------------------------------------------------
    def to_numpy(self, x):
        return np.asarray(x)

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------
    def jit_compile(self, fn, **kwargs):
        """NumPy has no JIT, return function as-is."""
        return fn
