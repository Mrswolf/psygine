# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Robust PCA.
"""
import numpy as np
from .proxmap import *


def rpca_admm(M, lamda, rho=None, tol=1e-4, max_iter=200, verbose=False):
    r"""Robust Principle Componet Analysis with Alternating Direction Method of Multipliers (ADMM)

    Parameters
    ----------
    X : (m, n) array_like
        Measured signals in matrix form, m is the number of features and n is the number of observations.
    lamda : float
        Regularization parameter for controlling the sparsity of S.
    rho : float, optional
        Penalty factor.
    tol : float, default 1e-4
        Tolerance.
    max_iter : int, default 200
        Naximum number of iterations.
    verbose : bool, default False
        Display iteration information.

    Returns
    ------
    L : (m, n) array_like
        Low rank matrix.
    S : (m, n) array_like
        Sparse matrix.

    Notes
    -----
    The L+S decomposition with ADMM adopts the following augmented Lagrangian objective function:

    ..math::
        \underset{\mathbf{L}, \mathbf{S}, \mathbf{Y}}{\mathrm{argmin}}\ {\| \mathbf{L} \|}_{\ast} + \lambda_S {\| \mathrm{vec}(\mathbf{S}) \|}_1 + \mathrm{tr}(\mathbf{Y}^T(X-L-S))  + \frac{\rho}{2} {\|X-L-S\|}_F^2

    where :math:`\mathrm{vec}()` is the vectorization operator of a matrix.

    References
    ----------
    .. [1] Candès, Emmanuel J., et al. "Robust principal component analysis?." Journal of the ACM (JACM) 58.3 (2011): 1-37.
    .. [2] Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.
    """
    Lk = np.zeros_like(M)
    Sk = np.zeros_like(M)
    Yk = np.zeros_like(M)

    n_iter = 0
    converge = False
    eps = np.finfo(M.dtype).eps
    old_norm = np.linalg.norm(M, "fro")
    threshold = tol * old_norm

    # default rho in Candes' paper
    if rho is None:
        m, n = M.shape
        rho = 0.25 / (np.maximum(np.linalg.norm(M.flatten(), 1) / (m * n), eps))

    while not converge:
        n_iter += 1
        if n_iter > max_iter:
            print("reach the maximum number of iteration")
            break

        # step 1
        old_Lk = Lk
        Lk = proxmap_nuclear(M - Sk + Yk / rho, 1 / rho)

        # step 2
        old_Sk = Sk
        Sk = proxmap_l1(M - Lk + Yk / rho, lamda / rho)

        # step 3
        Yk = Yk + rho * (M - Lk - Sk)

        # Candes's stopping criterion
        primal_residual = np.linalg.norm(M - Lk - Sk, "fro")
        dual_residual = rho * np.linalg.norm(Lk - old_Lk, "fro")

        # Boyd's stopping criterion
        # primal_residual = np.linalg.norm(M - Lk - Sk, 'fro')
        # # not sure which one is right
        # # dual_residual = rho * np.linalg.norm(Sk - old_Sk, 'fro')
        # dual_residual = rho * np.linalg.norm(Lk - old_Lk, 'fro')
        # if primal_residual < threshold and dual_residual < threshold:
        #     converge = True
        # else:
        #     if primal_residual > 10 * dual_residual:
        #         rho *= 2
        #     elif dual_residual > 10 * primal_residual:
        #         rho /= 2

        if verbose:
            disp_info = "iter: {:d} rank: {:d} sparsity: {:.2f}% rho: {:.4f} primal: {:.4f} dual: {:.4f}"
            print(
                disp_info.format(
                    n_iter,
                    np.linalg.matrix_rank(Lk),
                    100 * (1 - np.count_nonzero(Sk) / float(Sk.size)),
                    rho,
                    primal_residual,
                    dual_residual,
                )
            )

        if primal_residual < threshold:
            print("reach the convergence at iter:{:d}".format(n_iter))
            converge = True
    return Lk, Sk


def rpca_ista(
    M, miu, lamda, t=0.99, tol=1e-5, max_iter=200, continuation=False, verbose=False
):
    r"""L+S decomposition with the iterative soft thresholding algorithm (ISTA)

    Parameters
    ----------
    X : (m, n) array_like
        Measured signals in matrix form, m is the number of features and n is the number of observations.
    lambda_l : float
        Regularization parameter for controlling the rank of L.
    lambda_s : float
        Regularization parameter for controlling the sparsity of S.
    t : float, default 1
        Updating step size.
    tol : float, default 1e-4
        Tolerance.
    max_iter : int, default 200
        Naximum number of iterations.
    verbose : bool, default False
        Display iteration information.

    Returns
    ------
    L : (m, n) array_like
        Low rank matrix.
    S : (m, n) array_like
        Sparse matrix.

    Notes
    -----
    The L+S decomposition with ISTA adopts the following objective function:

    ..math::
        \underset{\mathbf{L}, \mathbf{S}}{\mathrm{argmin}}\ \lambda_L {\| \mathbf{L} \|}_{\ast} + \lambda_S {\| \mathrm{vec}(\mathbf{S}) \|}_1 + \frac{1}{2} {\|X-(L+S)\|}_F^2

    where :math:`\mathrm{vec}()` is the vectorization operator of a matrix. ISTA solves the problem by following steps:

    1. Update :math:`\hat{\mathbf{L}}_k` and :math:`\hat{\mathbf{S}}_k`:

    .. math::
        \begin{split}
        \hat{\mathbf{L}}_k &= \mathbf{L}_k + t\left( X-(\mathbf{L}_k+\mathbf{S}_k)\right)\\
        \hat{\mathbf{S}}_k &= \mathbf{S}_k + t\left( X-(\mathbf{L}_k+\mathbf{S}_k)\right)
        \end{split}

    2. Solve the proximal mapping of the nuclear norm of L

    .. math::
        \mathbf{L}_{k+1} = \underset{\mathbf{L}}{\mathrm{argmin}}\ \lambda_L {\| \mathbf{L} \|}_{\ast} + \frac{1}{2t} {\|L-\hat{\mathbf{L}}_k\|}_F^2

    3. Solve the proximal mapping of the L1 norm of S

    .. math::
        \mathbf{S}_{k+1} = \underset{\mathbf{S}}{\mathrm{argmin}}\ \lambda_S {\| \mathrm{vec}(\mathbf{S}) \|}_1 + \frac{1}{2t} {\|S-\hat{\mathbf{S}}_k\|}_F^2

    4. Repeat steps 1-3 until convergence.

    References
    ----------
    .. [1] TODO
    """
    Lk = np.zeros_like(M)
    Sk = np.zeros_like(M)

    n_iter = 0
    converge = False
    eps = np.finfo(M.dtype).eps
    threshold = tol * np.linalg.norm(M, "fro")
    residual = M - Lk - Sk

    if continuation:
        miu_lower_bound = 1e-5 * miu

    while not converge:
        n_iter += 1
        if n_iter > max_iter:
            print("reach the maximum number of iteration")
            break

        # step 1
        Lk_hat = Lk + t * residual
        Lk = proxmap_nuclear(Lk_hat, miu * t)

        # step 2
        Sk_hat = Sk + t * residual
        Sk = proxmap_l1(Sk_hat, miu * lamda * t)

        # step 3
        residual = M - (Lk + Sk)

        if continuation:
            miu = np.maximum(miu * 0.9, miu_lower_bound)

        if verbose:
            disp_info = (
                "iter: {:d} miu: {:.4f} rank: {:d} sparsity: {:.2f}% primal: {:.4f}"
            )
            print(
                disp_info.format(
                    n_iter,
                    miu,
                    np.linalg.matrix_rank(Lk),
                    100 * (1 - np.count_nonzero(np.abs(Sk)) / float(Sk.size)),
                    np.linalg.norm(residual, "fro"),
                )
            )

        if np.linalg.norm(residual, "fro") < threshold:
            print("reach the convergence at iter:{:d}".format(n_iter))
            converge = True
    return Lk, Sk


def rpca_fista(
    M, miu, lamda, t=0.5, tol=1e-5, max_iter=200, continuation=False, verbose=False
):
    Lk = np.zeros_like(M)
    Sk = np.zeros_like(M)
    Lk_1 = np.zeros_like(M)
    Sk_1 = np.zeros_like(M)

    ak = 1
    ak_1 = 1

    n_iter = 0
    converge = False
    eps = np.finfo(M.dtype).eps
    # threshold = tol * np.linalg.norm(M, 'fro')

    if continuation:
        miu_lower_bound = 1e-5 * miu

    while not converge:
        n_iter += 1
        if n_iter > max_iter:
            print("reach the maximum number of iteration")
            break

        YL = Lk + (ak_1 - 1) / ak * (Lk - Lk_1)
        YS = Sk + (ak_1 - 1) / ak * (Sk - Sk_1)

        Lk_1 = Lk
        Sk_1 = Sk

        residual = M - YL - YS

        # step 1
        Lk_hat = YL + t * residual
        Lk = proxmap_nuclear(Lk_hat, miu * t)

        # step 2
        Sk_hat = YS + t * residual
        Sk = proxmap_l1(Sk_hat, miu * lamda * t)

        # step 3
        ak_1 = ak
        ak = 0.5 * (1 + np.sqrt(1 + 4 * ak * ak))
        if continuation:
            miu = np.maximum(miu * 0.9, miu_lower_bound)

        primal_residual = np.linalg.norm(Sk + Lk - Lk_1 - Sk_1, "fro")
        gradient_norm = np.linalg.norm(
            np.concatenate((YL - Lk + Sk - YS, YS - Sk + Lk - YL), axis=0), "fro"
        )
        threshold = (
            np.maximum(np.linalg.norm(np.concatenate((Lk, Sk), axis=0), "fro"), 1)
            * 2
            * tol
        )

        if verbose:
            disp_info = "iter: {:d} miu: {:.4f} rank: {:d} sparsity: {:.2f} gradient: {:.4f} primal: {:.4f}"
            print(
                disp_info.format(
                    n_iter,
                    miu,
                    np.linalg.matrix_rank(Lk),
                    100 * (1 - np.count_nonzero(np.abs(Sk)) / float(Sk.size)),
                    gradient_norm,
                    primal_residual,
                )
            )

        if gradient_norm < threshold:
            print("reach the convergence at iter:{:d}".format(n_iter))
            converge = True

    return Lk, Sk
