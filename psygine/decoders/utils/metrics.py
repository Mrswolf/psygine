# -*- coding: utf-8 -*-

# Authors: swolf <swolfforever@gmail.com>
# Date: 2025/03/02
# License: MIT License
"""Metrics.
"""
import numpy as np


def mse(x, y):
    """Mean Squared Error."""
    return np.mean(np.abs(x - y) ** 2, axis=tuple(range(1, x.ndim)))


def psnr(x, y):
    """Peak Signa-to-Noise Ratio."""
    if np.issubdtype(y.dtype, np.integer):
        return 20 * np.log10(255) - 10 * np.log10(mse(x, y))
    elif np.issubdtype(y.dtype, np.floating) or np.issubdtype(
        y.dtype, np.Complexfloating
    ):
        return 20 * np.log10(1) - 10 * np.log10(mse(x, y))
    else:
        raise NotImplementedError(f"not implemented for current dtype {y.dtype}")


def nrmse(x, y):
    """Normalized Root-mean Square Error."""
    return np.sqrt(mse(x, y) / mse(0, y))
