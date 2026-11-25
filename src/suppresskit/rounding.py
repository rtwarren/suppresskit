"""Base rounding of counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _check(values: pd.DataFrame, base: int) -> NDArray[np.int64]:
    if base < 2:
        raise ValueError("base must be at least 2")
    arr = values.to_numpy()
    if np.isnan(arr.astype(float)).any():
        raise ValueError("values contain missing data")
    if (arr < 0).any() or (arr != np.floor(arr)).any():
        raise ValueError("values must be non-negative whole numbers")
    result: NDArray[np.int64] = arr.astype(np.int64)
    return result


def round_to_base(values: pd.DataFrame, base: int = 5) -> pd.DataFrame:
    """Round every count to the nearest multiple of ``base`` (halves round up).

    Deterministic rounding is easy to reproduce but can be reversed when the
    rounding rule and totals are both known. Prefer :func:`random_round` for release.
    """
    arr = _check(values, base)
    rounded = ((arr + base // 2) // base) * base
    return pd.DataFrame(rounded, index=values.index, columns=values.columns)


def random_round(values: pd.DataFrame, base: int = 5, seed: int | None = None) -> pd.DataFrame:
    """Unbiased random rounding to a multiple of ``base``.

    A count ``n`` with remainder ``r = n % base`` rounds up with probability
    ``r / base`` and down otherwise, so the expected value equals ``n``. Counts that
    are already multiples of ``base`` are unchanged. Pass ``seed`` for a
    reproducible release.
    """
    arr = _check(values, base)
    rng = np.random.default_rng(seed)
    remainder = arr % base
    floor = arr - remainder
    round_up = rng.random(arr.shape) < (remainder / base)
    out = np.where(round_up, floor + base, floor)
    return pd.DataFrame(out, index=values.index, columns=values.columns)
