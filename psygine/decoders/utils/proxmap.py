# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/10/05
# License: MIT License
"""Basic operators related with proximal mapping.

"""
import numpy as np


def proxmap_l1(X, tau):
    if tau < 0:
        raise ValueError("tau should be an non-negative value.")
    X_hat = np.exp(1j * np.angle(X)) * np.maximum(np.abs(X) - tau, 0)
    return X_hat.astype(X.dtype)


def proxmap_nuclear(X, tau):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    X_hat = (U * proxmap_l1(s, tau)) @ Vh
    return X_hat
