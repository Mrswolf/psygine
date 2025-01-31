# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/10/05
# License: MIT License
"""Basic operators related with proximal mapping.

"""
import numpy as np
import warnings
# Suppress only ComplexWarning
from numpy.exceptions import ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)

__all__ = ["proxmap_l1", "proxmap_nuclear"]


def proxmap_l1(X, tau):
    r"""Proximal mapping of L-1 norm.

    Parameters
    ----------
    X : array_like
        Input data array.
    tau : float
        Non-negative soft-thresholding value.


    Returns
    -------
    Y : array_like
        Output data array.

    Notes
    -----
    Proximal mapping of L-1 norm on vectorized data is defined as follows:

    .. math:: \mathbf{y}^* = \mathop{\mathrm{argmin}}_{\mathbf{y}} \ \frac{1}{2\tau} \|\mathbf{x}-\mathbf{y}\|_2^2 + \|\mathbf{y}\|_1
    """
    if tau < 0:
        raise ValueError("tau should be an non-negative value.")
    Y = np.exp(1j * np.angle(X)) * np.maximum(np.abs(X) - tau, 0)
    return Y.astype(X.dtype)

def proxmap_nuclear(X, tau):
    r"""Proximal mapping of nuclear norm.

    Parameters
    ----------
    X : (m, n) array_like
        Input data array.
    tau : float
        Non-negative singular value threshold value.


    Returns
    -------
    Y : (m, n) array_like
        Output data array.

    Notes
    -----
    Proximal mapping of nuclear norm on matrices is defined as follows:

    .. math:: \mathbf{Y}^* = \mathop{\mathrm{argmin}}_{\mathbf{Y}} \ \frac{1}{2\tau} \|\mathbf{X}-\mathbf{Y}\|_F^2 + \|\mathbf{Y}\|_*
    """
    # TODO: replace it with fastsvd in utils
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    Y = (U * proxmap_l1(s, tau)) @ Vh
    return Y
