"""Half-band rounding rules.

- 'nearest' (Paper 2 main rule): min(9, max(0, floor(2x + 0.5) / 2)),
  ties (.25/.75) round UP by construction.
- 'ceil' (Paper 1 rule, kept for the ablation switch): ceil(2x) / 2.

A tiny epsilon guards against float noise at exact grid points; predictions
from Ridge/LGBM almost never sit exactly on a boundary, and the anchor test
(exact quarter means) is unaffected.
"""
import numpy as np

_EPS = 1e-9


def round_half_nearest(x):
    x = np.asarray(x, dtype=float)
    return np.clip(np.floor(2.0 * x + 0.5 + _EPS) / 2.0, 0.0, 9.0)


def round_half_ceil(x):
    x = np.asarray(x, dtype=float)
    return np.clip(np.ceil(2.0 * x - _EPS) / 2.0, 0.0, 9.0)


def apply_rounding(x, mode: str):
    if mode == "nearest":
        return round_half_nearest(x)
    if mode == "ceil":
        return round_half_ceil(x)
    raise ValueError(f"unknown rounding mode: {mode}")
