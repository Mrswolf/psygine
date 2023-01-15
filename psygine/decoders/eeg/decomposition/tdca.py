# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/15
# License: MIT License
"""TDCA-related methods.
"""
import numpy as np
from scipy.linalg import eigh, qr
from .dsp import dsp_kernel, dsp_feature
from .base import pearsonr, FilterBank
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

__all__ = [
    'tdca_kernel', 'tdca_feature', 'TDCA', 'FBTDCA'
]


def _proj_ref(Yf):
    r"""Get projection matrix from reference signals.

    Parameters
    ----------
    Yf : (n_channels, n_samples) array_like
        Reference signal.

    Returns
    -------
    P : (n_samples, n_samples) array_like
        Projection matrix.
    """
    Q, _ = qr(Yf.T, mode='economic')
    P = Q@Q.T
    return P

def _aug_2(X, n_samples, l, P, training=True):
    r"""Augment EEG signal.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples+l) array_like
        EEG signals.
    n_samples : int
        The number of samples without delay.
    l : int
        The number of delay points.
    training : bool, default True
        Return training augmented data or testing augmented data.

    Returns
    -------
    aug_X : (n_trials, (l+1)*n_channels, 2*n_samples) array_like
        Augmented EEG signals.
    """
    X = X.reshape((-1, *X.shape[-2:]))
    n_trials, n_channels, n_points = X.shape
    if n_points < l+n_samples:
        raise ValueError("the length of X should be larger than l+n_samples.")
    aug_X = np.zeros((n_trials, (l+1)*n_channels, n_samples))
    if training:
        for i in range(l+1):
            aug_X[:, i*n_channels:(i+1)*n_channels, :] = X[..., i:i+n_samples]
    else:
        for i in range(l+1):
            aug_X[:, i*n_channels:(i+1)*n_channels, :n_samples-i] = X[..., i:n_samples]
    aug_Xp = aug_X@P
    aug_X = np.concatenate([aug_X, aug_Xp], axis=-1)
    return aug_X

def tdca_kernel(
    X, y, Yf, l,
    cov_estimator='cov'):
    r"""TDCA kernel.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples+l) array_like
        EEG signal.
    y : (n_trials,) array_like
        Labels.
    Yf : (n_freqs, 2*n_harmonics, n_samples) array_like
        Reference signals. Note the order of the first dimension should match the order of classes (in ascending order) and n_freqs == n_classes.
    l : int
        The number of delay points.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.

    Returns
    -------
    D : ((l+1)*n_channels,) array_like
        Eigenvalues in descending order.
    W : ((l+1)*n_channels, (l+1)*n_channels) array_like
        Eigenvectors corresponding to eigenvalues.
    M : ((l+1)*n_channels, 2*n_samples) array_like
        Template signal for all classes.
    P : (n_freqs, n_samples, n_samples) array_like
        Project matrices computed from Yf.
    Mc : (n_freqs, (l+1)*n_channels, 2*n_samples) array_like
        Transformed template signal of each class.

    References
    ----------
    .. [1] Liu, Bingchuan, et al. "Improving the performance of individually calibrated ssvep-bci by task-discriminant component analysis." IEEE Transactions on Neural Systems and Rehabilitation Engineering 29 (2021): 1998-2007.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    n_samples = Yf.shape[-1]
    labels = np.unique(y)
    P = np.stack([_proj_ref(Yf[i]) for i in range(len(labels))])
    aug_X, aug_Y = [], []
    for i, label in enumerate(labels):
        aug_X.append(
            _aug_2(
                X[y==label], n_samples, l, P[i], training=True))
        aug_Y.append(y[y==label])

    aug_X = np.concatenate(aug_X, axis=0)
    aug_Y = np.concatenate(aug_Y, axis=0)

    D, W, M = dsp_kernel(aug_X, aug_Y, cov_estimator=cov_estimator)

    Mc = np.stack([
            np.mean(dsp_feature(W, M, aug_X[aug_Y==label], n_components=W.shape[-1]), axis=0) for label in labels
            ])
    return D, W, M, P, Mc

def tdca_feature(
    X, Mc, W, M, P, l,
    n_components=1):
    """TDCA feature.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signals.
    Mc: (n_freqs, (l+1)*n_channels, 2*n_samples) array_like
        Template signal of each class.
    W : ((l+1)*n_channels, (l+1)*n_channels) array_like
        Spatial filters.
    M: ((l+1)*n_channels, 2*n_samples) array_like
        Template signal for all classes.
    P : (n_freqs, n_samples, n_samples) array_like
        Project matrices computed from Yf.
    l : int
        The number of delay points.
    n_components : int, default 1
        The number of components.

    Returns
    -------
    (n_trials, n_freqs) array_like
        TDCA correlation features.

    References
    ----------
    .. [1] Liu, Bingchuan, et al. "Improving the performance of individually calibrated ssvep-bci by task-discriminant component analysis." IEEE Transactions on Neural Systems and Rehabilitation Engineering 29 (2021): 1998-2007.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    n_samples = P.shape[-1]
    rhos = []
    for Xk, Pi in zip(Mc, P):
        a = dsp_feature(
                W, M, _aug_2(X, n_samples, l, Pi, training=False), n_components=n_components)
        b = Xk[:n_components, :]
        a = np.reshape(a, (a.shape[0], -1))
        b = np.reshape(b, (1, -1))
        rhos.append(pearsonr(a, b))
    rhos = np.stack(rhos).T
    return rhos

class TDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Task-discriminant component analysis.

    Parameters
    ----------
    l : int
        The number of delay points.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.

    Notes
    -----
    TDCA [1]_ method for SSVEP-BCI.

    References
    ----------
    .. [1] Liu, Bingchuan, et al. "Improving the performance of individually calibrated ssvep-bci by task-discriminant component analysis." IEEE Transactions on Neural Systems and Rehabilitation Engineering 29 (2021): 1998-2007.
    """
    def __init__(self,
            l,
            n_components=1,
            cov_estimator='cov'):
        self.l = l
        self.n_components = n_components
        self.cov_estimator = cov_estimator

    def fit(self, X, y, Yf=None):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples+l) array_like
            EEG signal with delay points.
        y : (n_trials,) array_like
            Labels.
        Yf : (n_freqs, n_channels, n_samples) array_like
            Reference signals. Note the order of the first dimension should match the order of classes (in ascending order) and n_freqs == n_classes.
        """
        if Yf is None:
            raise ValueError("Yf must be provided.")
        self.classes_ = np.unique(y)
        _, self.W_, self.M_, self.P_, self.Mc_ = tdca_kernel(
            X, y, Yf, self.l, cov_estimator=self.cov_estimator)
        return self
 
    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        rhos : (n_trials, n_freqs) array_like
            TDCA correlation features.
        """
        rhos = tdca_feature(
            X, self.Mc_, self.W_, self.M_, self.P_, self.l,
            n_components=self.n_components)
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

class FBTDCA(FilterBank, ClassifierMixin):
    r"""Filterbank TDCA.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    l : int
        The number of delay points.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    filter_weights : array_like, optional
        Filter weights for each sub-band.
        If None, all sub-bands have the same weight 1.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """
    def __init__(self,
        filterbank,
        l,
        n_components=1,
        cov_estimator='cov',
        filter_weights=None,
        n_jobs=None):
        self.filterbank = filterbank
        self.l = l
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.filter_weights = filter_weights
        self.n_jobs = n_jobs

        if (self.filter_weights is not None
            and len(self.filter_weights) != len(filterbank)):
            raise ValueError("Filter weights and filterbank should be the same length.")

        super().__init__(
            TDCA(l, n_components=n_components, cov_estimator=cov_estimator),
            filterbank,
            concat=False,
            n_jobs=n_jobs)

    def fit(self, X, y, Yf=None):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples+l) array_like
            EEG signal with delay points.
        y : (n_trials,) array_like
            Labels.
        Yf : (n_freqs, n_channels, n_samples) array_like
            Reference signals. Note the order of the first dimension should match the order of classes (in ascending order) and n_freqs == n_classes.
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, axis=-1, Yf=Yf)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        rhos : (n_trials, n_freqs) array_like
            Combined TDCA correlation features.
        """
        features = super().transform(X, axis=-1)
        if self.filter_weights is None:
            filter_weights = np.ones(len(self.filterbank))
        else:
            filter_weights = self.filter_weights
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
