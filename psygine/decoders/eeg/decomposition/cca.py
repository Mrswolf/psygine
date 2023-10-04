# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/20
# License: MIT License
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.cross_decomposition import CCA
from joblib import Parallel, delayed
from .base import pearsonr, FilterBank

__all__ = [
    'cca_kernel', 'cca_feature', 'SCCA', 'FBSCCA', 'ecca_feature', 'ECCA', 'FBECCA'
]

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
    X = X - np.mean(X, axis=-1, keepdims=True)
    Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
    U, V = cca_kernel(X, Yf, n_components=n_components)
    a = np.matmul(U.T, X)
    b = np.matmul(V.T, Yf)
    a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
    r = pearsonr(a, b)
    return r[0]

def cca_feature(X, Yf, n_components=1, n_jobs=None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))

    # rhos = []
    # for yf in Yf:
    #     rs = Parallel(n_jobs=n_jobs)(delayed(partial(_cca_feature, n_components=n_components))(x, yf) for x in X)
    #     rhos.append(rs)
    # rhos = np.stack(rhos).T
    
    rhos = Parallel(n_jobs=n_jobs)(delayed(partial(_cca_feature, n_components=n_components))(x, yf) for x in X for yf in Yf)
    rhos = np.reshape(rhos, (X.shape[0], Yf.shape[0]))

    return rhos

class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, Yf, n_components=1, n_jobs=None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.Yf = Yf        
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        rhos = cca_feature(
            X, self.Yf, n_components=self.n_components, n_jobs=self.n_jobs)
        return rhos

    def predict(self, X):
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)
        return labels

class FBSCCA(FilterBank, ClassifierMixin):
    r"""Filterbank SCCA.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    Yf : (n_classes, 2*n_harmonics, n_samples) array_like
        Reference signals.
    n_components : int, default 1
        The number of components.
    filter_weights : array_like, optional
        Filter weights for each sub-band.
        If None, all sub-bands have the same weight 1.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """
    def __init__(self,
        filterbank,
        Yf,
        n_components=1,
        filter_weights=None,
        n_jobs=None):
        self.filterbank = filterbank
        self.Yf = Yf
        self.n_components = n_components
        self.filter_weights = filter_weights
        self.n_jobs = n_jobs

        if self.Yf is None:
            raise ValueError("The reference signals Yf should be provided.")

        if (self.filter_weights is not None
            and len(self.filter_weights) != len(filterbank)):
            raise ValueError("Filter weights and filterbank should be the same length.")

        super().__init__(
            SCCA(Yf, n_components=n_components, n_jobs=n_jobs),
            filterbank,
            concat=False,
            n_jobs=n_jobs)

    def fit(self, X=None, y=None):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal, not used here.
        y : (n_trials,) array_like
            Labels, not used here.
        """
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        rhos : (n_trials, n_classes) array_like
            Combined DSP correlation features.
        """
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(self.filterbank))]
        features = super().transform(X, axis=-1)
        if self.filter_weights is None:
            filter_weights = np.ones(len(self.filterbank))
        else:
            filter_weights = self.filter_weights
        # features  = np.square(features) * filter_weights
        features = features * filter_weights
        features = np.sum(features, axis=-1)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        features = self.transform(X)
        labels = np.argmax(features, axis=-1)
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
    # rhos = []
    # for t, yf in zip(T, Yf):
    #     rs = Parallel(n_jobs=n_jobs)(delayed(partial(_ecca_feature, n_components=n_components))(x, t, yf) for x in X)
    #     rhos.append(rs)
    # rhos = np.stack(rhos).T
    
    rhos = Parallel(n_jobs=n_jobs)(delayed(partial(_ecca_feature, n_components=n_components))(x, t, yf) for x in X for t, yf in zip(T, Yf))
    rhos = np.reshape(rhos, (X.shape[0], Yf.shape[0]))
    
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
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        rhos = ecca_feature(
            X, self.T_, self.Yf, n_components=self.n_components, n_jobs=self.n_jobs)
        return rhos

    def predict(self, X):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBECCA(FilterBank, ClassifierMixin):
    r"""Filterbank ECCA.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    Yf : (n_classes, 2*n_harmonics, n_samples) array_like
        Reference signals.
    n_components : int, default 1
        The number of components.
    filter_weights : array_like, optional
        Filter weights for each sub-band.
        If None, all sub-bands have the same weight 1.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """
    def __init__(self,
        filterbank,
        Yf,
        n_components=1,
        filter_weights=None,
        n_jobs=None):
        self.filterbank = filterbank
        self.Yf = Yf
        self.n_components = n_components
        self.filter_weights = filter_weights
        self.n_jobs = n_jobs

        if self.Yf is None:
            raise ValueError("The reference signals Yf should be provided.")

        if (self.filter_weights is not None
            and len(self.filter_weights) != len(filterbank)):
            raise ValueError("Filter weights and filterbank should be the same length.")

        super().__init__(
            ECCA(Yf, n_components=n_components, n_jobs=n_jobs),
            filterbank,
            concat=False,
            n_jobs=n_jobs)

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels, not used here.
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, axis=-1)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        rhos : (n_trials, n_classes) array_like
            Combined DSP correlation features.
        """
        features = super().transform(X, axis=-1)
        if self.filter_weights is None:
            filter_weights = np.ones(len(self.filterbank))
        else:
            filter_weights = self.filter_weights
        # features  = np.square(features) * filter_weights
        features = features * filter_weights
        features = np.sum(features, axis=-1)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        features = self.transform(X)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels