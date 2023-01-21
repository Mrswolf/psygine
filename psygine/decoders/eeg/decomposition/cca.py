# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/20
# License: MIT License
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.cross_decomposition import CCA
from joblib import Parallel, delayed
from .base import pearsonr

def cca_kernel(X, Yf, n_components=1):
    r"""Naive CCA kernel.

    Parameters
    ----------
    X : (n_channels, n_samples) array_like
        EEG signal.
    Yf : (2*n_harmonics, n_samples) array_like
        Reference signal.

    Returns
    -------
    U : (n_channels, n_components) array_like
        The left singular vectors.
    V : (2*n_harmonics, n_components) array_like
        The right singular vectors.
    """
    X = X - np.mean(X, axis=-1, keepdims=True)
    Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
    est = CCA(n_components=n_components)
    est.fit(X.T, Yf.T)
    U = est.x_weights_
    V = est.y_weights_
    return U, V

def _cca_feature(X, Yf, n_components=1):
    U, V = cca_kernel(X, Yf, n_components=n_components)
    a = np.matmul(U.T, X)
    b = np.matmul(V.T, Yf)
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    r = pearsonr(a, b)
    return r

def cca_feature(X, Yf, n_components=1, n_jobs=None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))

    rhos = []
    for yf in Yf:
        rs = Parallel(n_jobs=n_jobs)(delayed(partial(_cca_feature, n_components=n_components))(x, yf) for x in X)
        rhos.append(rs)
    rhos = np.stack(rhos).T
    return rhos

class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, Yf, n_components=1, n_jobs=None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.Yf = Yf        
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rhos = cca_feature(
            X, self.Yf, n_components=self.n_components, n_jobs=self.n_jobs)
        return rhos

    def predict(self, X):
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)
        return labels

def _ecca_feature(X, T, Yf, n_components=1):
    rhos = []
    # 14a, 14d
    U1, V1 = cca_kernel(X, Yf, n_components=n_components)
    a = U1.T@X
    b = V1.T@Yf
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    a = U1.T@X
    b = U1.T@T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    # 14b
    U2, _ = cca_kernel(X, T)
    a = U2.T@X
    b = U2.T@T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    # 14c
    U3, _ = cca_kernel(T, Yf)
    a = U3.T@X
    b = U3.T@T
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    rhos.append(pearsonr(a, b))
    rhos = np.array(rhos)
    r = np.sum(np.sign(rhos)*np.square(rhos))
    return r

def ecca_feature(X, T, Yf, n_components=1, n_jobs=None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
    T = np.reshape(T, (-1, *T.shape[-2:]))
    rhos = []
    for t, yf in zip(T, Yf):
        rs = Parallel(n_jobs=n_jobs)(delayed(partial(_ecca_feature, n_components=n_components))(x, t, yf) for x in X)
        rhos.append(rs)
    rhos = np.stack(rhos).T
    return rhos

class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, Yf, n_components=1, n_jobs=None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.Yf = Yf
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.T_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])
        return self

    def transform(self, X):
        rhos = ecca_feature(
            X, self.T_, self.Yf, n_components=self.n_components, n_jobs=self.n_jobs)
        return rhos

    def predict(self, X):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels
