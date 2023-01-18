# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/18
# License: MIT License
"""CSP-related methods.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from psygine.decoders.utils import covariances
from .base import FilterBank

__all__ = [
    'csp_kernel', 'csp_feature', 'CSP', 'FBCSP'
]

def csp_kernel(X, y, cov_estimator='cov'):
    r"""CSP kernel.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    y : (n_trials,) array_like
        Labels.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.

    Returns
    -------
    D : (n_channels,) array_like
        Eigenvalues in descending order.
    W : (n_channels, n_channels) array_like
        Eigenvectors corresponding to eigenvalues.

    Notes
    -----
    Instead of choosing the first and last m components in [1]_, this implementation sorted W with |D-0.5| in descending order, so that the first 2m components can be chosed.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError("csp kernel is only for 2-class classification problem.")

    C1 = covariances(
        X[y==labels[0]],
        estimator=cov_estimator,
        n_jobs=1)
    C2 = covariances(
        X[y==labels[1]],
        estimator=cov_estimator,
        n_jobs=1)

    # trace normalization
    C1 = C1 / np.trace(C1, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]
    C2 = C2 / np.trace(C2, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]

    C1 = np.mean(C1, axis=0)
    C2 = np.mean(C2, axis=0)
    Cc = C1 + C2

    D, W = eigh(C1, Cc)
    # the eigenvalues range from 0 to 1
    D = np.abs(D - 0.5)
    idx = np.argsort(D)[::-1]

    D, W = D[idx], W[:,idx]
    return D, W

def csp_feature(X, W, n_components=2):
    r"""CSP feature.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    W : (n_channels, n_channels) array_like
        Spatial filters.
    n_components : int, default 2
        The number of components.

    Returns
    -------
    (n_trials, n_templates) array_like
        CSP features.

    Notes
    -----
    CSP features based on [1]_.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    eps = np.finfo(X.dtype).eps
    feat = np.matmul(W[:,:n_components].T, X)
    # variance normalization
    feat = np.mean(np.square(feat), axis=-1)
    feat = feat / (np.sum(feat, axis=-1, keepdims=True) + eps)
    # log transformation
    feat = np.log(np.clip(feat, eps, None))
    return feat

class CSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Common spatial pattern.

    Parameters
    ----------
    n_components : int, default 2
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.

    Notes
    -----
    CSP [1]_ method for MI-BCI.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    def __init__(self,
            n_components=2,
            cov_estimator='cov',
            classifier=None):
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.classifier = classifier
        self.classifier_ = None

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        _, W = csp_kernel(
            X, y, cov_estimator=self.cov_estimator)
        self.W_ = W

        if self.classifier is not None:
            self.classifier_ = clone(self.classifier)
            feat = csp_feature(
                X, W, n_components=self.n_components)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_components) array_like
            CSP features.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        features = csp_feature(X, self.W_, n_components=self.n_components)
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
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")

class FBCSP(FilterBank, ClassifierMixin):
    r"""Filterbank CSP.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.
    n_mi_components : int, default 1
        The number of mutual information selection components.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """
    def __init__(self,
        filterbank,
        n_components=2,
        cov_estimator='cov',
        classifier=None,
        n_mi_components=4,
        n_jobs=None):
        self.filterbank = filterbank
        self.n_components = n_components
        self.n_mi_components = n_mi_components
        self.cov_estimator = cov_estimator
        self.classifier = classifier
        self.classifier_ = None
        self.n_jobs = n_jobs

        super().__init__(
            CSP(
                n_components=n_components,
                cov_estimator=cov_estimator,
                classifier=None),
            filterbank,
            concat=True,
            n_jobs=n_jobs)

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        super().fit(X, y, axis=-1)

        # feature selection step
        feat = super().transform(X)
        self.selector_ = SelectKBest(
            score_func=mutual_info_classif,
            k=self.n_mi_components)
        self.selector_.fit(feat, y)

        if self.classifier is not None:
            feat = self.selector_.transform(feat)
            self.classifier_ = clone(self.classifier)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_mi_components) array_like
            CSP features with mutual information selection.
        """
        features = super().transform(X, axis=-1)
        features = self.selector_.transform(features)
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
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")