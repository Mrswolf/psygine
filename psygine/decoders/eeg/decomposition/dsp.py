# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/14
# License: MIT License
"""DSP-related Methods.
"""
from functools import partial
import numpy as np
from scipy.linalg import eigh
from psygine.decoders.utils import covariances
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import Parallel, delayed
from .base import pearsonr, FilterBank

__all__ = [
    'dsp_kernel', 'dsp_feature', 'DSP', 'FBDSP'
]

def dsp_kernel(X, y, cov_estimator='cov'):
    r"""DSP kernel.

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
    M : (n_channels, n_samples) array_like
        Template signal for all classes.

    Notes
    -----
    The implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    labels = np.unique(y)
    # the normalized weight of each class
    weights = np.array([np.sum(y==label) for label in labels])
    weights = weights / np.sum(weights)
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms = np.stack([np.mean(X[y==label], axis=0) for label in labels])
    Ss = np.stack([
        np.sum(covariances(X[y==label] - Ms[i], estimator=cov_estimator, n_jobs=1), axis=0) for i, label in enumerate(labels)], axis=-1)

    # within-class scatter matrix
    Sw = np.sum(Ss * weights, axis=-1)
    # between-class scatter matrix
    Sb = covariances(Ms-M, estimator=cov_estimator, n_jobs=1)
    Sb = np.sum(Sb * weights[:, np.newaxis, np.newaxis], axis=0)

    D, W = eigh(Sb, Sw)
    D, W = D[::-1], W[:,::-1]
    return D, W, M

def dsp_feature(W, M, X, n_components=1):
    """DSP feature.

    Parameters
    ----------
    W : (n_channels, n_channels) array_like
        Spatial filters.
    M: (n_channels, n_samples) array_like
        Template signal for all classes.
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    n_components : int, default 1
        The number of components.

    Returns
    -------
    (n_trials, n_components, n_samples) array_like
        DSP features.
 
    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    feat = np.matmul(W[:, :n_components].T, X - M)
    return feat

class DSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Discriminative spatial patterns.

    Parameters
    ----------
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    transform_method : str, default 'corr'
        If 'corr', return correlation features with class templates.
        If 'mean', return mean features described in [1]_.

    Notes
    -----
    DSP [1]_ method for SSVEP-BCI.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    def __init__(self,
            n_components=1,
            cov_estimator='cov',
            transform_method='corr'):
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.transform_method = transform_method

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

        _, W, M = dsp_kernel(X, y, cov_estimator=self.cov_estimator)
        self.W_ = W
        self.M_ = M
        self.templates_ = np.stack([
            np.mean(dsp_feature(W, M, X[y==label], n_components=W.shape[1]), axis=0) for label in self.classes_], axis=0)
        return self

    def _pearson_transform(self, X, T):
        X = np.reshape(X, (X.shape[0], -1))
        T = np.reshape(T, (T.shape[0], -1))
        rhos = []
        for t in T:
            rhos.append(pearsonr(X, t))
        rhos = np.stack(rhos).T
        return rhos

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        rhos : (n_trials, n_classes) array_like
            DSP correlation features if transform_method is 'corr'.
            Otherwise, return mean features of shape (n_trials, n_components).
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)

        features = dsp_feature(self.W_, self.M_, X, n_components=self.n_components)

        if self.transform_method == 'mean':
            return np.mean(features, axis=-1)
        elif self.transform_method == 'corr':
            T = self.templates_[:,:self.n_components,:]
            return self._pearson_transform(features, T)
        else:
            raise ValueError("non-supported transform method")

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
        if self.transform_method == 'corr':
            labels = self.classes_[np.argmax(feat, axis=-1)]
        else:
            raise NotImplementedError
        return labels

class FBDSP(FilterBank, ClassifierMixin):
    r"""Filterbank DSP.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    transform_method : str, default 'corr'
        If 'corr', return correlation features with class templates.
        If 'mean', return mean features described in [1]_.
    filter_weights : array_like, optional
        Filter weights for each sub-band.
        If None, all sub-bands have the same weight 1.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """
    def __init__(self,
        filterbank,
        n_components=1,
        cov_estimator='cov',
        transform_method='corr',
        filter_weights=None,
        n_jobs=None):
        self.filterbank = filterbank
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.transform_method = transform_method
        self.filter_weights = filter_weights
        self.n_jobs = n_jobs

        if (self.filter_weights is not None
            and len(self.filter_weights) != len(filterbank)):
            raise ValueError("Filter weights and filterbank should be the same length.")

        super().__init__(
            DSP(n_components=n_components, cov_estimator=cov_estimator, transform_method=transform_method),
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
            Combined DSP correlation features.
        """
        features = super().transform(X, axis=-1)
        if self.filter_weights is None:
            filter_weights = np.ones(len(self.filterbank))
        else:
            filter_weights = self.filter_weights
        if self.transform_method == 'corr':
            features  = np.square(features) * filter_weights
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
