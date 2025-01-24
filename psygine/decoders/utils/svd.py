# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Basic methods related with SVD.
"""
import os

os.environ["SCIPY_USE_PROPACK"] = "True"

import numpy as np
from scipy.integrate import quad
from scipy.linalg import svd
from scipy.sparse.linalg import svds


__all__ = ["sign_flip", "optimal_svht", "fastsvd"]


def sign_flip(U, V, X):
    r"""Flip signs of SVD.

    Parameters
    ----------
    U : (m, k) array_like
        Left singular vectors, :math:`k \leq min(m, n)`.
    V : (n, k) array_like
        Right singular vectors.
    X : (m, n) array_like
        Data array, ``[U, s, Vh] = np.linalg.svd(X)``

    Returns
    -------
    U : (m, k) array_like
        Left singular vectors.
    V : (n, k) array_like
        Right singular vectors.

    Notes
    -----
    The SVD itself provides no means for assessing the sign of each singular vector. In actual algorithmic implementations of SVD, the individual singular vectors have an 'arbitrary' sign. Mathematically, there is no way to avoid the sign ambiguity of a multiplicative term such as the pair of singular vectors. However, data analysis is more than algebra. In order to identify the sign of a singular vector, it is suggested that it would be similar to the sign of the majority of vectors it is representing. Geometrically, it should point in the same, not the opposite, direction as the points it is representing.

    The sign can be determined from the sign of the inner product of the singular vector and the individual data vectors, more details can be found in [1]_ and [2]_.

    The implementation here is a modified version of the authors' code, the sign of the kth pair of singular vectors would be:

    .. math:: s_k = \mathop{\mathrm{argmax}}_{s_k \in \{1, -1\}} s_k \left( \frac{1}{m} \sum_{i=1}^{m} \frac{X_{i,.}v_k}{\| X_{i,.} \|_2} + \frac{1}{n} \sum_{j=1}^{n} \frac{u_k^T X_{.,j}}{\| X_{.,j} \|_2} \right)

    Examples
    --------
    >>> X = np.random.randn(50, 6)
    >>> [U, s, Vh] = np.linalg.svd(X, full_matrices=False)
    >>> U, V = sign_flip(U, Vh.T, X)

    References
    ----------
    .. [1] Bro, Rasmus, Evrim Acar, and Tamara G. Kolda. "Resolving the sign ambiguity in the singular value decomposition." Journal of Chemometrics: A Journal of the Chemometrics Society 22.2 (2008): 135-140.
    .. [2] https://www.mathworks.com/matlabcentral/fileexchange/22118-sign-correction-in-svd-and-pca
    """
    M = X / np.linalg.norm(X, axis=1, keepdims=True)
    right_projections = np.mean(M @ V, axis=0)
    M = X / np.linalg.norm(X, axis=0, keepdims=True)
    left_projections = np.mean(U.T @ M, axis=1)
    total_projections = right_projections + left_projections
    signs = np.sign(total_projections)

    random_sign_idx = signs == 0
    if np.any(random_sign_idx):
        # arbitrary signs due to nearly zero magnitudes
        signs[random_sign_idx] = 1

    U = U * signs
    V = V * signs

    return U, V


def optimal_svht(m, n, sigma, noise_known=False):
    r"""Optimal SVD hard thresholding.

    Parameters
    ----------
    m : int
        The first dimension of the matrix.
    n : int
        The second dimension of the matrix.
    sigma : float
        Gaussian noise variance if known, otherwise the median of singular values.
    known_noise : bool, default False
        Whether the noise level is known.

    Returns
    -------
    tau : float
        Optimal SVD hard thresholding.

    Notes
    -----
    SVD threshold choosing method based on [1]_ and [2]_.

    Examples
    --------
    Generate noisy data matrix:

    >>> t = np.arange(-3, 3, 1e-2)
    >>> Utrue = np.stack([np.cos(17*t)*np.exp(-t**2), np.sin(11*t)]).T
    >>> Strue = np.array([[2, 0], [0, 0.5]])
    >>> Vtrue = np.stack([np.sin(5*t)*np.exp(-t**2), np.cos(13*t)]).T
    >>> X = np.matmul(Utrue, np.matmul(Strue, Vtrue.T))
    >>> sigma = 1
    >>> Xnoisy = X + sigma*np.random.randn(*X.shape)

    With known noise level:

    >>> [U, S, Vt] = np.linalg.svd(Xnoisy, full_matrices=False)
    >>> [m, n] = Xnoisy.shape
    >>> tau = optimal_svht(m, n, sigma=1, known_noise=True)
    >>> S[S<tau] = 0
    >>> Xhat_1 = np.matmul(U, np.matmul(np.diag(S), Vt))

    With unknown noise level:

    >>> [U, S, Vt] = np.linalg.svd(Xnoisy, full_matrices=False)
    >>> [m, n] = Xnoisy.shape
    >>> tau = optimal_svht(m, n, sigma=np.median(S), known_noise=False)
    >>> S[S<tau] = 0
    >>> Xhat_2 = np.matmul(U, np.matmul(np.diag(S), Vt))

    References
    ----------
    .. [1] Gavish, Matan, and David L. Donoho. "The optimal hard threshold for singular values is $4/\sqrt {3} $." IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
    .. [2] https://purl.stanford.edu/vg705qn9070.
    """
    if m < n:
        m, n = n, m

    beta = n / m

    if noise_known:
        weight = np.sqrt(
            2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
        )
        tau = weight * np.sqrt(m) * sigma
    else:
        weight = np.sqrt(
            2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
        )
        # approximate the median of the Marcenko-Pastur distribution numerically
        lower_bound = (1 - np.sqrt(beta)) ** 2
        upper_bound = (1 + np.sqrt(beta)) ** 2
        # the correct median equation
        int_func = (
            lambda t: np.sqrt((upper_bound - t) * (t - lower_bound))
            / (2 * np.pi * beta * t)
            if (upper_bound - t) * (t - lower_bound) > 0
            else 0
        )

        lo = lower_bound
        hi = upper_bound
        while (hi - lo) > 1e-3:
            t_vec = np.linspace(lo, hi, 5)
            y_vec = [quad(int_func, lower_bound, t)[0] for t in t_vec]
            if y_vec[2] > 0.5:
                hi = t_vec[2]
            elif y_vec[2] < 0.5:
                lo = t_vec[2]
            else:
                break
        miu_beta = (lo + hi) / 2
        weight = weight / np.sqrt(miu_beta)
        tau = weight * sigma
    return tau


def fastsvd(A, k=1, method="scipy", random_state=None):
    U, s, Vh = None, None, None
    rs = np.random.RandomState(random_state)
    if method == "matlab":
        [U, s, Vh] = svd(A, full_matrices=False, lapack_driver="gesvd")
        U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
    elif method == "scipy":
        [U, s, Vh] = svd(A, full_matrices=False, lapack_driver="gesdd")
        U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
    elif method == "propack":
        [U, s, Vh] = svds(A, k=k, which="LM", solver="propack")
    elif method == "arpack":
        v0 = rs.uniform(-1, 1, min(A.shape))
        [U, s, Vh] = svds(A, k=k, which="LM", v0=v0, solver="arpack")
    elif method == "lobpcg":
        [U, s, Vh] = svds(A, k=k, which="LM", solver="lobpcg")
    else:
        raise NotImplementedError(
            "Unknown svd method:{:s}, available methods include matlab, scipy, propack, arpack, lobpcg.".format(
                method
            )
        )
    return U, s, Vh
