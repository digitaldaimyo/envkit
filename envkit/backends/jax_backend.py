
# envkit/backends/jax_backend.py

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp

from .base import Backend


class JaxBackend(Backend):
    """
    JAX-based backend.

    - Arrays are jax.numpy DeviceArrays.
    - All math runs in JAX space.
    - RNG integration is done via integer seeds (from RNGStreams.next_seed),
      which are converted to JAX PRNGKeys per call.
    """
    xp = jnp
    float_dtype = jnp.float32
    int_dtype = jnp.int32
    bool_dtype = jnp.bool_

    # ------------------------------------------------------------------
    # creation / conversion
    # ------------------------------------------------------------------
    def asarray(self, x, dtype: Optional[Any] = None):
        return jnp.asarray(x, dtype=dtype)

    def set_requires_grad(self, x, requires_grad: bool):
        # No autograd flags in JAX at the array level
        return x

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return jnp.zeros(shape, dtype=dtype or self.float_dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return jnp.ones(shape, dtype=dtype or self.float_dtype)

    def zeros_like(self, x):
        return jnp.zeros_like(jnp.asarray(x))

    def ones_like(self, x):
        return jnp.ones_like(jnp.asarray(x))

    def copy(self, x):
        # JAX arrays are immutable, but this gives a "fresh" array value-wise.
        return jnp.array(jnp.asarray(x), copy=True)

    # ------------------------------------------------------------------
    # shape / indexing helpers
    # ------------------------------------------------------------------
    def reshape(self, x, shape: Tuple[int, ...]):
        return jnp.reshape(jnp.asarray(x), shape)

    def squeeze(self, x, axis: Optional[int] = None):
        return jnp.squeeze(jnp.asarray(x), axis=axis)

    def expand_dims(self, x, axis: int):
        return jnp.expand_dims(jnp.asarray(x), axis=axis)

    def concat(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return jnp.zeros((0,), dtype=self.float_dtype)
        arrs = [jnp.asarray(a) for a in arrays]
        return jnp.concatenate(arrs, axis=axis)

    def stack(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return jnp.zeros((0,), dtype=self.float_dtype)
        arrs = [jnp.asarray(a) for a in arrays]
        return jnp.stack(arrs, axis=axis)

    def broadcast_to(self, x, shape: Tuple[int, ...]):
        return jnp.broadcast_to(jnp.asarray(x), shape)

    # ------------------------------------------------------------------
    # math
    # ------------------------------------------------------------------
    def sum(self, x, axis=None, keepdims: bool = False):
        return jnp.sum(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims: bool = False):
        return jnp.mean(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims: bool = False):
        return jnp.max(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims: bool = False):
        return jnp.min(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def norm(self, x, ord=None, axis=None, keepdims: bool = False):
        return jnp.linalg.norm(jnp.asarray(x), ord=ord, axis=axis, keepdims=keepdims)

    def clip(self, x, lo, hi):
        return jnp.clip(jnp.asarray(x), lo, hi)

    def nan_to_num(self, x, nan=0.0, posinf=0.0, neginf=0.0):
        return jnp.nan_to_num(jnp.asarray(x), nan=nan, posinf=posinf, neginf=neginf)

    def maximum(self, x, y):
        return jnp.maximum(jnp.asarray(x), jnp.asarray(y))

    def minimum(self, x, y):
        return jnp.minimum(jnp.asarray(x), jnp.asarray(y))

    def abs(self, x):
        return jnp.abs(jnp.asarray(x))

    def sqrt(self, x):
        return jnp.sqrt(jnp.asarray(x))

    # ------------------------------------------------------------------
    # comparisons / logic
    # ------------------------------------------------------------------
    def where(self, cond, a, b):
        return jnp.where(jnp.asarray(cond), jnp.asarray(a), jnp.asarray(b))

    def logical_and(self, a, b):
        return jnp.logical_and(jnp.asarray(a), jnp.asarray(b))

    def logical_or(self, a, b):
        return jnp.logical_or(jnp.asarray(a), jnp.asarray(b))

    def logical_not(self, a):
        return jnp.logical_not(jnp.asarray(a))

    def isfinite(self, x):
        return jnp.isfinite(jnp.asarray(x))

    def isinf(self, x):
        return jnp.isinf(jnp.asarray(x))

    def isnan(self, x):
        return jnp.isnan(jnp.asarray(x))

    # ------------------------------------------------------------------
    # reductions / indexing
    # ------------------------------------------------------------------
    def any(self, x, axis=None, keepdims: bool = False):
        return jnp.any(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def all(self, x, axis=None, keepdims: bool = False):
        return jnp.all(jnp.asarray(x), axis=axis, keepdims=keepdims)

    def argsort(self, x, axis: int = -1):
        arr = jnp.asarray(x)
        return jnp.argsort(arr, axis=axis, stable=True)

    def argmax(self, x, axis=None):
        result = jnp.argmax(jnp.asarray(x), axis=axis)
        return int(result) if axis is None else result

    def argmin(self, x, axis=None):
        result = jnp.argmin(jnp.asarray(x), axis=axis)
        return int(result) if axis is None else result

    def nonzero(self, x):
        nz = jnp.nonzero(jnp.asarray(x))
        # return numpy-style index arrays for convenience
        return tuple(np.asarray(t) for t in nz)

    # ------------------------------------------------------------------
    # rng (seed-based)
    #
    # seed is an integer from RNGStreams.next_seed(...)
    # ------------------------------------------------------------------
    def random(self, seed: int, size):
        key = jax.random.PRNGKey(int(seed))
        return jax.random.uniform(key, shape=size, dtype=self.float_dtype)

    def choice(self, seed: int, n, p=None):
        key = jax.random.PRNGKey(int(seed))
        n_int = int(n)
        if p is None:
            # uniform over [0, n)
            idx = jax.random.randint(key, shape=(), minval=0, maxval=n_int, dtype=jnp.int32)
            return int(idx)
        probs = jnp.asarray(p, dtype=jnp.float32)
        probs = probs / probs.sum()
        # categorical expects log-probs
        idx = jax.random.categorical(key, jnp.log(probs), shape=())
        return int(idx)

    def randint(self, seed: int, low: int, high: int, size):
        key = jax.random.PRNGKey(int(seed))
        return jax.random.randint(
            key,
            shape=size,
            minval=int(low),
            maxval=int(high),
            dtype=jnp.int64,
        )

    def normal(self, seed: int, mean: float, std: float, size):
        key = jax.random.PRNGKey(int(seed))
        z = jax.random.normal(key, shape=size, dtype=jnp.float32)
        return z * std + mean

    # ------------------------------------------------------------------
    # host interop
    # ------------------------------------------------------------------
    def to_numpy(self, x):
        # Best-effort conversion to host numpy array
        return np.asarray(x)

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------
    def jit_compile(self, fn, **kwargs):
        """
        Compile function with jax.jit.

        Args:
            fn: Function to compile
            **kwargs: Passed to jax.jit (static_argnums, donate_argnums, etc.)

        Returns:
            JIT-compiled function
        """
        return jax.jit(fn, **kwargs)
