# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/12/31
# License: MIT License
"""TRCA-related Methods.
"""
from functools import partial
import numpy as np
from scipy.linalg import eigh
from psygine.decoders.utils import covariances
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import Parallel, delayed
from .base import pearsonr, FilterBank

__all__ = ["trca_kernel", "trca_feature", "TRCA", "FBTRCA"]


def trca_kernel(X, cov_estimator="cov"):
    r"""TRCA kernel.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.

    Returns
    -------
    D : (n_channels,) array_like
        Eigenvalues in descending order.
    W : (n_channels, n_channels) array_like
        Eigenvectors corresponding to eigenvalues.
    T : (n_channels, n_samples) array_like
        Template signal.

    Notes
    -----
    Fast TRCA method based on [1]_, which improves original TRCA method.

    References
    ----------
    .. [1] Chiang, Kuan-Jung, et al. "Reformulating Task-Related Component Analysis for Reducing its Computational Complexity." (2022).
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    T = np.mean(X, axis=0)
    S_bar = covariances(T, estimator=cov_estimator, n_jobs=1)
    Q = covariances(X, estimator=cov_estimator, n_jobs=1)
    Q = np.mean(Q, axis=0)
    S = S_bar - Q
    D, W = eigh(S, Q)
    D, W = D[::-1], W[:, ::-1]
    return D, W, T


def trca_feature(X, Ts, Ws, n_components=1, ensemble=True):
    r"""TRCA feature.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    Ts : (n_templates, n_channels, n_samples) array_like
        Template signal.
    Ws : (n_templates, n_channels, n_channels) array_like
        Spatial filters.
    n_components : int, default 1
        The number of components.
    ensemble : bool, default True
        Return features with the ensemble method if True.

    Returns
    -------
    (n_trials, n_templates) array_like
        TRCA features.

    Notes
    -----
    TRCA features based on [1]_.

    References
    ----------
    .. [1] Nakanishi, Masaki, et al. "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis." IEEE Transactions on Biomedical Engineering 65.1 (2017): 104-112.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    Ts = np.reshape(Ts, (-1, *Ts.shape[-2:]))
    Ws = np.reshape(Ws, (-1, *Ws.shape[-2:]))

    if len(Ts) != len(Ws):
        raise ValueError("Ts and Ws should be the same length.")

    rhos = []
    if not ensemble:
        for T, W in zip(Ts, Ws):
            a = np.matmul(W[:, :n_components].T, X)
            b = np.matmul(W[:, :n_components].T, T)
            a = np.reshape(a, (a.shape[0], -1))
            b = np.reshape(b, (1, -1))
            rhos.append(pearsonr(a, b))
    else:
        Ws = Ws[..., :n_components]
        W = np.concatenate(Ws, axis=-1)
        for T in Ts:
            a = np.matmul(W.T, X)
            b = np.matmul(W.T, T)
            a = np.reshape(a, (a.shape[0], -1))
            b = np.reshape(b, (1, -1))
            rhos.append(pearsonr(a, b))
    rhos = np.stack(rhos).T
    return rhos


class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Task-related component analysis.

    Parameters
    ----------
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    ensemble : bool, default True
        Return features with the ensemble method if True.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.

    Notes
    -----
    TRCA [1]_ method for SSVEP-BCI.

    References
    ----------
    .. [1] Nakanishi, Masaki, et al. "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis." IEEE Transactions on Biomedical Engineering 65.1 (2017): 104-112.
    """

    def __init__(self, n_components=1, cov_estimator="cov", ensemble=True, n_jobs=None):
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        _, Ws, Ts = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(partial(trca_kernel, cov_estimator=self.cov_estimator))(
                    X[y == label]
                )
                for label in self.classes_
            )
        )
        self.templates_ = np.stack(Ts)
        self.Ws_ = np.stack(Ws)
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
            TRCA correlation features.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        rhos = trca_feature(
            X,
            self.templates_,
            self.Ws_,
            n_components=self.n_components,
            ensemble=self.ensemble,
        )
        return rhos

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
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCA(FilterBank, ClassifierMixin):
    r"""Filterbank TRCA.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    ensemble : bool, default True
        Return features with the ensemble method if True.
    filter_weights : array_like, optional
        Filter weights for each sub-band.
        If None, all sub-bands have the same weight 1.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """

    def __init__(
        self,
        filterbank,
        n_components=1,
        cov_estimator="cov",
        ensemble=True,
        filter_weights=None,
        n_jobs=None,
    ):
        self.filterbank = filterbank
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.ensemble = ensemble
        self.filter_weights = filter_weights
        self.n_jobs = n_jobs

        if self.filter_weights is not None and len(self.filter_weights) != len(
            filterbank
        ):
            raise ValueError("Filter weights and filterbank should be the same length.")

        super().__init__(
            TRCA(
                n_components=n_components,
                cov_estimator=cov_estimator,
                ensemble=ensemble,
            ),
            filterbank,
            concat=False,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
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
            Combined TRCA correlation features.
        """
        features = super().transform(X, axis=-1)
        if self.filter_weights is None:
            filter_weights = np.ones(len(self.filterbank))
        else:
            filter_weights = self.filter_weights

        # bad performance for square
        # features  = np.square(features) * filter_weights

        # nakanishi's features
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
