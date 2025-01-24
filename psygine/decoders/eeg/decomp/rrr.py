# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2025/01/24
# License: MIT License
"""Reduced-Rank Regression.

"""
import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import Parallel, delayed
from functools import partial
from psygine.decoders.utils import sqrtm, invsqrtm

from .base import pearsonr

__all__ = ["rrr", "ECCAR3", "TRCAR3"]


def rrr(X, Y, k=1, Wx=None, Wy=None):
    r"""Reduced-Rank Regression.

    .. math::
        \argmin_{A, B} \frac{1}{2}\|Y - A@B.T@X\|_F^2, s.t. rank(A@B.T) = k

    Parameters
    ----------
    X : ndarray
        (Nx, Ns)
    Y : ndarray
        (Ny, Ns)
    k : int, optional
        the reduced rank k, by default 1
    Wx : ndarray, optional
        weight matrix, by default None
    Wy : ndarray, optional
        weight matrix, by default None

    Returns
    -------
    A: ndarray
        (Ny, k)
    B: ndarray
        (Nx, k)
    s: ndarray
        (k,)

    Raises
    ------
    ValueError
        if X and Y are not consistent
    """
    if Y.shape[1] != X.shape[1]:
        raise ValueError(f"X{X.shape[-1]} should be equal to Y{Y.shape[-1]}")
    Ny, Nx = Y.shape[0], X.shape[0]
    Ns = X.shape[1]

    if Wx is None:
        Gx = np.eye(Ns)
    else:
        Gx = Wx.T @ Wx

    if Wy is None:
        Gy = np.eye(Ny)
    else:
        Gy = Wy @ Wy.T

    # svd
    isqrt_xGxxt = invsqrtm(X @ Gx @ X.T)
    sqrt_Gy = sqrtm(Gy)
    isqrt_Gy = invsqrtm(Gy)
    Z = sqrt_Gy @ Y @ Gx @ X.T @ isqrt_xGxxt
    [U, s, Vh] = svd(Z, full_matrices=False)
    ind = np.argsort(s)[::-1]
    U, s, Vh = U[:, ind], s[ind], Vh[ind, :]
    A = isqrt_xGxxt @ Vh[:k, :].T
    B = isqrt_Gy @ U[:, :k]
    s = s[:k]

    return A, B, s


def _ecca_r3_feature(X, T, Yf, n_components=1):
    X = X - np.mean(X, axis=-1, keepdims=True)
    Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
    T = T - np.mean(T, axis=-1, keepdims=True)
    rhos = []
    # 14a, 14d
    Wy = invsqrtm(Yf @ Yf.T)
    A, B, _ = rrr(X, Yf, k=n_components, Wy=Wy)
    C = A @ B.T
    a = Wy.T @ C.T @ X
    b = Wy.T @ Yf
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    a = Wy.T @ C.T @ X
    b = Wy.T @ C.T @ T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    # 14b, 14e
    Wy = invsqrtm(T @ T.T)
    A, B, _ = rrr(X, T, k=n_components, Wy=Wy)
    C = A @ B.T
    a = Wy.T @ C.T @ X
    b = Wy.T @ C.T @ T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    a = Wy.T @ C.T @ T
    b = Wy.T @ T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    # 14c
    Wy = invsqrtm(Yf @ Yf.T)
    A, B, _ = rrr(T, Yf, k=n_components, Wy=Wy)
    C = A @ B.T
    a = Wy.T @ C.T @ X
    b = Wy.T @ C.T @ T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    rhos = np.array(rhos)
    r = np.sum(np.sign(rhos) * np.square(rhos))
    return r


def ecca_r3_feature(X, T, Yf, n_components=1, n_jobs=None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
    T = np.reshape(T, (-1, *T.shape[-2:]))

    rhos = Parallel(n_jobs=n_jobs)(
        delayed(partial(_ecca_r3_feature, n_components=n_components))(x, t, yf)
        for x in X
        for t, yf in zip(T, Yf)
    )
    rhos = np.reshape(rhos, (X.shape[0], Yf.shape[0]))
    return rhos


class ECCAR3(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, Yf, n_components=1, n_jobs=None):
        self.Yf = Yf
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.T_ = np.stack([np.mean(X[y == label], axis=0) for label in self.classes_])
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        rhos = ecca_r3_feature(
            X, self.T_, self.Yf, n_components=self.n_components, n_jobs=self.n_jobs
        )
        return rhos

    def predict(self, X):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


def trca_r3_kernel(X, k=1, n_jobs=None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X_bar = np.sum(X, axis=0)
    N = X.shape[0]
    Wy = invsqrtm(X_bar @ X_bar.T)
    A, B, _ = rrr(
        np.reshape(np.transpose(X, [1, 0, 2]), (X.shape[1], -1)),
        np.kron(np.ones((1, N)), X_bar),
        k=k,
        Wy=Wy,
    )
    C = A @ B.T
    return C, Wy, X_bar


def trca_r3_feature(X, Ts, Cs, Ws, ensemble=True):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Ts = np.reshape(Ts, (-1, *Ts.shape[-2:]))
    Ws = np.reshape(Ws, (-1, *Ws.shape[-2:]))
    Cs = np.reshape(Cs, (-1, *Cs.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)

    rhos = []
    if not ensemble:
        for T, C, W in zip(Ts, Cs, Ws):
            a = W.T @ C.T @ X
            b = W.T @ C.T @ T
            # b = W.T@T
            a = np.reshape(a, (a.shape[0], -1))
            b = np.reshape(b, (1, -1))
            rhos.append(pearsonr(a, b))
    else:
        C = np.concatenate([C @ W for W, C in zip(Ws, Cs)], axis=-1)
        W = np.concatenate(Ws, axis=-1)
        for T in Ts:
            a = C.T @ X
            b = C.T @ T
            # b = W.T@T
            a = np.reshape(a, (a.shape[0], -1))
            b = np.reshape(b, (1, -1))
            rhos.append(pearsonr(a, b))
    rhos = np.stack(rhos).T
    return rhos


class TRCAR3(BaseEstimator, TransformerMixin, ClassifierMixin):

    def __init__(self, n_components=1, ensemble=True, n_jobs=None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Cs, Ws, Ts = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(
                    partial(trca_r3_kernel, k=self.n_components, n_jobs=self.n_jobs)
                )(X[y == label])
                for label in self.classes_
            )
        )
        self.templates_ = np.stack(Ts)
        self.Ws_ = np.stack(Ws)
        self.Cs_ = np.stack(Cs)
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        rhos = trca_r3_feature(
            X,
            self.templates_,
            self.Cs_,
            self.Ws_,
            ensemble=self.ensemble,
        )
        return rhos

    def predict(self, X):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels
