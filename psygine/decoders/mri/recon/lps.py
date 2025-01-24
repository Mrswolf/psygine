# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2025/01/24
# License: MIT License
"""Otazo's L+S Reconstruction.

https://github.com/JeffFessler/reproduce-l-s-dynamic-mri/tree/main/data
"""
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from psygine.decoders.utils import fft1c, ifft1c, proxmap_l1


# only for reproducible purpose
# Otazo used the wrong fft transform, causing inversion of axes
# though it's mathmatically equvilent
def _fft2c_mri(X, axes=(0, 1)):
    K = fftshift(
        ifft2(fftshift(X, axes=axes), axes=axes, norm="ortho", workers=-1), axes=axes
    )
    return K


def _ifft2c_mri(K, axes=(0, 1)):
    X = fftshift(
        fft2(fftshift(K, axes=axes), axes=axes, norm="ortho", workers=-1), axes=axes
    )
    return X


def _convert2kspace(X, sens=None, mask=None):
    if sens is not None:
        X = X[:, :, :, np.newaxis] * sens[:, :, np.newaxis, :]
    else:
        X = X[:, :, :, np.newaxis]

    K = _fft2c_mri(X, axes=(0, 1))

    if mask is not None:
        K *= mask[:, :, :, np.newaxis]
    return K


def _convert2image(K, sens=None, mask=None):
    if mask is not None:
        K *= mask[:, :, :, np.newaxis]

    K = _ifft2c_mri(K, axes=(0, 1))

    # be aware of normalize of sens
    if sens is not None:
        sens = sens.conj() / np.sum(np.square(np.abs(sens)), axis=-1, keepdims=True)
        X = np.sum(K * sens[:, :, np.newaxis, :], axis=-1)
    else:
        X = np.mean(K, axis=-1)
    return X


def proxmap_nuclear_otazo(X, tau):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    X_hat = (U * proxmap_l1(s, tau * s[0])) @ Vh
    return X_hat


# TODO: to be replaced with pylops
class GArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, forward_func, backward_func, adjoint=False, **kwargs):
        self._forward_func = forward_func
        self._backward_func = backward_func
        self.adjoint = adjoint
        self._kwargs = kwargs

    def __repr__(self):
        pass

    def __array__(self, dtype=None):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            output = None
            # print(ufunc.__name__)
            for input in inputs:
                if isinstance(input, np.ndarray):
                    if self.adjoint:
                        output = self._backward_func(input, **self._kwargs)
                    else:
                        output = self._forward_func(input, **self._kwargs)
            return output

    def __array_function__(self, func, types, args, kwargs):
        return func(*args, **kwargs)

    @property
    def T(self):
        return self.__class__(
            self._forward_func,
            self._backward_func,
            adjoint=not self.adjoint,
            **self._kwargs,
        )

    @property
    def H(self):
        return self.__class__(
            self._forward_func,
            self._backward_func,
            adjoint=not self.adjoint,
            **self._kwargs,
        )


def lps_otazo_ista(
    d,
    miu=1e-2,
    lamda=1,
    sens=None,
    mask=None,
    t=1,
    tol=2.5e-3,
    max_iter=50,
    continuation=False,
    verbose=False,
):
    r"""L+S MR reconstruction used by Otazo."""
    E = GArray(_convert2kspace, _convert2image, sens=sens, mask=mask)
    T = GArray(fft1c, ifft1c, axis=-1)

    [m, n, nt, nc] = d.shape
    M0 = E.T @ d
    Lk = np.copy(M0)
    Sk = np.zeros((m, n, nt), dtype=d.dtype)

    n_iter = 0
    converge = False

    residual = d - E @ (Lk + Sk)

    if continuation:
        miu_lower_bound = 1e-5 * miu

    while not converge:
        n_iter += 1
        if n_iter > max_iter:
            print("reach the maximum number of iteration")
            break

        M0 = Lk + Sk
        threshold = np.linalg.norm(np.ravel(M0), 2) * tol
        # step 1
        Lk_hat = Lk + t * (E.T @ residual)
        Lk_hat = np.reshape(Lk_hat, (m * n, nt))
        # otazo's way
        Lk = proxmap_nuclear_otazo(Lk_hat, miu * t)

        Lk = np.reshape(Lk, (m, n, nt))

        # step 2
        Sk_hat = T @ (Sk + t * (E.T @ residual))
        Sk = T.T @ (proxmap_l1(Sk_hat, miu * lamda * t))

        # step 3
        residual = d - E @ (Lk + Sk)

        residual_norm = np.linalg.norm(np.ravel(Lk + Sk - M0), 2)

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
                    np.linalg.matrix_rank(Lk.reshape((m * n, nt))),
                    100 * (1 - np.count_nonzero(np.abs(Sk)) / float(Sk.size)),
                    residual_norm,
                )
            )

        if residual_norm < threshold:
            print("reach the convergence at iter:{:d}".format(n_iter))
            converge = True
    return Lk, Sk


def lps_otazo_fista(
    d,
    miu=1e-2,
    lamda=1,
    sens=None,
    mask=None,
    t=0.5,
    tol=2.5e-3,
    max_iter=50,
    continuation=False,
    verbose=False,
):
    r"""L+S MR reconstruction used by Otazo."""
    E = GArray(_convert2kspace, _convert2image, sens=sens, mask=mask)
    T = GArray(fft1c, ifft1c, axis=-1)

    [m, n, nt, nc] = d.shape
    M0 = E.T @ d

    Lk = np.copy(M0)
    # Lk = np.zeros_like(M0)
    Sk = np.zeros_like(M0)
    Lk_1 = np.copy(M0)
    # Lk_1 = np.zeros_like(M0)
    Sk_1 = np.zeros_like(M0)

    ak = 1
    ak_1 = 1

    n_iter = 0
    converge = False

    if continuation:
        miu_lower_bound = 1e-5 * miu

    while not converge:
        n_iter += 1
        if n_iter > max_iter:
            print("reach the maximum number of iteration")
            break
        YL = Lk + (ak_1 - 1) / ak * (Lk - Lk_1)
        YS = Sk + (ak_1 - 1) / ak * (Sk - Sk_1)
        M0 = YL + YS
        residual = d - E @ (YL + YS)
        threshold = np.linalg.norm(np.ravel(Lk + Sk), 2) * tol

        Lk_1 = Lk
        Sk_1 = Sk

        # step 1
        Lk_hat = YL + t * (E.T @ residual)
        Lk_hat = np.reshape(Lk_hat, (m * n, nt))
        # otazo's way
        Lk = proxmap_nuclear_otazo(Lk_hat, miu * t)
        Lk = np.reshape(Lk, (m, n, nt))

        # step 2
        Sk_hat = T @ (YS + t * (E.T @ residual))
        Sk = T.T @ (proxmap_l1(Sk_hat, miu * lamda * t))

        # step 3
        ak_1 = ak
        ak = 0.5 * (1 + np.sqrt(1 + 4 * ak * ak))

        if continuation:
            miu = np.maximum(miu * 0.9, miu_lower_bound)

        residual_norm = np.linalg.norm(np.ravel(Lk - Lk_1 + Sk - Sk_1), 2)

        if verbose:
            disp_info = (
                "iter: {:d} miu: {:.4f} rank: {:d} sparsity: {:.2f}% primal: {:.4f}"
            )
            print(
                disp_info.format(
                    n_iter,
                    miu,
                    np.linalg.matrix_rank(Lk.reshape((m * n, nt))),
                    100 * (1 - np.count_nonzero(np.abs(Sk)) / float(Sk.size)),
                    residual_norm,
                )
            )

        if residual_norm < threshold:
            print("reach the convergence at iter:{:d}".format(n_iter))
            converge = True
    return Lk, Sk
