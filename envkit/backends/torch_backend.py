
# envkit/backends/torch_backend.py

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Any

import numpy as np
import torch

from .base import Backend


class TorchBackend(Backend):
    xp = torch

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.float_dtype = torch.float32
        self.int_dtype = torch.int32
        self.bool_dtype = torch.bool

    # ------------------------------------------------------------------
    # creation / conversion
    # ------------------------------------------------------------------
    def _tensor(self, x, dtype=None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t.to(self.device)

    def asarray(self, x, dtype: Optional[Any] = None):
        return self._tensor(x, dtype=dtype)

    def set_requires_grad(self, x, requires_grad: bool):
        t = self._tensor(x)
        t.requires_grad_(requires_grad)
        return t

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return torch.zeros(shape, dtype=dtype or self.float_dtype, device=self.device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None):
        return torch.ones(shape, dtype=dtype or self.float_dtype, device=self.device)

    def zeros_like(self, x):
        t = self._tensor(x)
        return torch.zeros_like(t)

    def ones_like(self, x):
        t = self._tensor(x)
        return torch.ones_like(t)

    def copy(self, x):
        return self._tensor(x).clone()

    # ------------------------------------------------------------------
    # shape / indexing helpers
    # ------------------------------------------------------------------
    def reshape(self, x, shape: Tuple[int, ...]):
        return self._tensor(x).reshape(shape)

    def squeeze(self, x, axis: Optional[int] = None):
        t = self._tensor(x)
        if axis is None:
            return t.squeeze()
        return t.squeeze(dim=axis)

    def expand_dims(self, x, axis: int):
        return self._tensor(x).unsqueeze(dim=axis)

    def concat(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return torch.zeros((0,), dtype=self.float_dtype, device=self.device)
        ts = [self._tensor(a) for a in arrays]
        return torch.cat(ts, dim=axis)

    def stack(self, arrays: Iterable, axis: int = 0):
        arrays = list(arrays)
        if not arrays:
            return torch.zeros((0,), dtype=self.float_dtype, device=self.device)
        ts = [self._tensor(a) for a in arrays]
        return torch.stack(ts, dim=axis)

    def broadcast_to(self, x, shape: Tuple[int, ...]):
        t = self._tensor(x)
        return t.expand(shape)

    # ------------------------------------------------------------------
    # math
    # ------------------------------------------------------------------
    def sum(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x)
        if axis is None:
            return torch.sum(t)
        return torch.sum(t, dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x)
        if axis is None:
            return torch.mean(t)
        return torch.mean(t, dim=axis, keepdim=keepdims)

    def max(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x)
        if axis is None:
            return torch.max(t)
        result = torch.max(t, dim=axis, keepdim=keepdims)
        return result.values

    def min(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x)
        if axis is None:
            return torch.min(t)
        result = torch.min(t, dim=axis, keepdim=keepdims)
        return result.values

    def norm(self, x, ord=None, axis=None, keepdims: bool = False):
        t = self._tensor(x, dtype=self.float_dtype)
        if axis is None:
            return torch.linalg.norm(t, ord=ord)
        return torch.linalg.norm(t, ord=ord, dim=axis, keepdim=keepdims)

    def clip(self, x, lo, hi):
        t = self._tensor(x)
        return torch.clamp(t, lo, hi)

    def nan_to_num(self, x, nan=0.0, posinf=0.0, neginf=0.0):
        t = self._tensor(x)
        return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)

    def maximum(self, x, y):
        return torch.maximum(self._tensor(x), self._tensor(y))

    def minimum(self, x, y):
        return torch.minimum(self._tensor(x), self._tensor(y))

    def abs(self, x):
        return torch.abs(self._tensor(x))

    def sqrt(self, x):
        return torch.sqrt(self._tensor(x))

    # ------------------------------------------------------------------
    # comparisons / logic
    # ------------------------------------------------------------------
    def where(self, cond, a, b):
        return torch.where(
            self._tensor(cond, dtype=torch.bool),
            self._tensor(a),
            self._tensor(b)
        )

    def logical_and(self, a, b):
        return torch.logical_and(self._tensor(a), self._tensor(b))

    def logical_or(self, a, b):
        return torch.logical_or(self._tensor(a), self._tensor(b))

    def logical_not(self, a):
        return torch.logical_not(self._tensor(a))

    def isfinite(self, x):
        return torch.isfinite(self._tensor(x))

    def isinf(self, x):
        return torch.isinf(self._tensor(x))

    def isnan(self, x):
        return torch.isnan(self._tensor(x))

    # ------------------------------------------------------------------
    # reductions / indexing
    # ------------------------------------------------------------------
    def any(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x, dtype=self.bool_dtype)
        if axis is None:
            return torch.any(t)
        return torch.any(t, dim=axis, keepdim=keepdims)

    def all(self, x, axis=None, keepdims: bool = False):
        t = self._tensor(x, dtype=self.bool_dtype)
        if axis is None:
            return torch.all(t)
        return torch.all(t, dim=axis, keepdim=keepdims)

    def argsort(self, x, axis: int = -1):
        t = self._tensor(x)
        return torch.argsort(t, dim=axis, stable=True)

    def argmax(self, x, axis=None):
        t = self._tensor(x)
        if axis is None:
            return int(torch.argmax(t).item())
        return torch.argmax(t, dim=axis)

    def argmin(self, x, axis=None):
        t = self._tensor(x)
        if axis is None:
            return int(torch.argmin(t).item())
        return torch.argmin(t, dim=axis)

    def nonzero(self, x):
        t = self._tensor(x, dtype=self.bool_dtype)
        nz = torch.nonzero(t, as_tuple=True)
        return tuple(n.cpu().numpy() for n in nz)

    # ------------------------------------------------------------------
    # rng (seed-based, torch-native)
    # ------------------------------------------------------------------
    def _make_generator(self, seed: int) -> torch.Generator:
        """Create a per-call torch.Generator seeded from an integer seed."""
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def random(self, seed: int, size):
        """Sample uniform[0, 1) with a stateless integer seed."""
        g = self._make_generator(seed)
        return torch.rand(size, generator=g, device=self.device, dtype=torch.float32)

    def choice(self, seed: int, n, p=None):
        """Sample a single index in [0, n) given optional probabilities p."""
        g = self._make_generator(seed)
        n_int = int(n)
        if p is None:
            idx = torch.randint(0, n_int, (1,), generator=g, device=self.device, dtype=torch.int64)[0]
            return int(idx.item())
        probs = torch.as_tensor(p, dtype=torch.float32, device=self.device)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)[0]
        return int(idx.item())

    def randint(self, seed: int, low: int, high: int, size):
        g = self._make_generator(seed)
        return torch.randint(int(low), int(high), size, generator=g, device=self.device, dtype=torch.int64)

    def normal(self, seed: int, mean: float, std: float, size):
        g = self._make_generator(seed)
        t = torch.randn(size, generator=g, device=self.device, dtype=torch.float32)
        return t * std + mean

    # ------------------------------------------------------------------
    # host interop
    # ------------------------------------------------------------------
    def to_numpy(self, x):
        t = self._tensor(x)
        return t.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------
    def jit_compile(self, fn, **kwargs):
        """
        Compile function with torch.compile.

        Args:
            fn: Function to compile
            **kwargs: Passed to torch.compile (mode, fullgraph, etc.)

        Returns:
            Compiled function
        """
        try:
            # torch.compile available in PyTorch 2.0+
            return torch.compile(fn, **kwargs)
        except AttributeError:
            # Fall back to uncompiled if torch.compile not available
            import warnings
            warnings.warn("torch.compile not available, returning uncompiled function")
            return fn
